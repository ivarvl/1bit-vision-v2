"""Train Faster R-CNN with a BinaryAttention-T/S/B backbone on Pascal VOC 2012.

Optimization recipe follows the BinaryAttention paper's detection setup:
AdamW (betas 0.9/0.999, weight decay 0.1), base LR 1e-3, 250-iter linear warm-up,
linear decay to zero across the remaining iterations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import VOC2012Detection
from engine import evaluate, log_compute_stats, train_one_epoch
from model import build_faster_rcnn, load_pretrained_backbone


class HFlip:
    """Random horizontal flip on (image_tensor, target_dict). Boxes are xyxy."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: torch.Tensor, target: dict) -> tuple[torch.Tensor, dict]:
        if torch.rand(1).item() >= self.p:
            return image, target
        image = TF.hflip(image)
        if target["boxes"].numel():
            w = image.shape[-1]
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="dataset/voc2012")
    p.add_argument("--output-dir", default="runs/binattn_voc")
    p.add_argument("--variant", choices=["tiny", "small", "base"], default="tiny")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8,
                   help="paper uses 64 (multi-GPU); default 8 for a single GPU")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-iters", type=int, default=250)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--no-attn-quant", action="store_true",
                   help="disable binary QK quantization (debug / fp baseline)")
    p.add_argument("--no-pv-quant", action="store_true",
                   help="disable 8-bit P/V quantization")
    p.add_argument("--pretrained", type=str, default=None,
                   help="path to ImageNet-1K pretrained BinaryAttention backbone (.pth)")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug-subset", type=int, default=0,
                   help="if >0, use only this many train+val images (smoke test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = VOC2012Detection(args.data_root, split="train", transforms=HFlip(0.5))
    val_ds   = VOC2012Detection(args.data_root, split="val", transforms=None)

    if args.debug_subset > 0:
        train_ds.ids = train_ds.ids[: args.debug_subset]
        val_ds.ids   = val_ds.ids[: args.debug_subset]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )

    model = build_faster_rcnn(
        variant=args.variant,
        num_classes=train_ds.num_classes,
        img_size=args.img_size,
        attn_quant=not args.no_attn_quant,
        pv_quant=not args.no_pv_quant,
        drop_path_rate=args.drop_path,
    ).to(device)

    if args.pretrained:
        missing, unexpected = load_pretrained_backbone(model.backbone, args.pretrained)
        print(f"loaded pretrained backbone from {args.pretrained}")
        print(f"  missing ({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
        print(f"  unexpected ({len(unexpected)}): {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )

    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_iters)

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    writer.add_text("config", "```\n" + json.dumps(vars(args), indent=2) + "\n```")

    start_epoch, global_step, best_ap = 0, 0, -1.0
    if args.resume and Path(args.resume).is_file():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]
        best_ap = ck.get("best_ap", -1.0)
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    log_compute_stats(model, model.backbone, args.img_size, device, writer)
    print(f"model={args.variant} train={len(train_ds)} val={len(val_ds)} "
          f"steps/epoch={len(train_loader)}")

    for epoch in range(start_epoch, args.epochs):
        global_step = train_one_epoch(
            model, optimizer, scheduler, train_loader, device,
            epoch, writer, global_step,
            log_interval=args.log_interval, grad_clip=args.grad_clip,
        )

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = evaluate(model, val_loader, device, epoch, writer)
            ap = metrics.get("AP", -1.0)
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_ap": max(best_ap, ap),
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "last.pt")
            if ap > best_ap:
                best_ap = ap
                torch.save(ckpt, out_dir / "best.pt")

    writer.close()
    print(f"done. best AP={best_ap:.4f}")


if __name__ == "__main__":
    main()
