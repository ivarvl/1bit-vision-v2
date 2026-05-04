"""Train one epoch and evaluate (COCO mAP) for the BinaryAttention detector.

All scalars are streamed to a ``torch.utils.tensorboard.SummaryWriter`` provided
by the caller — per-iteration training losses + LR + grad-norm, peak GPU memory
per epoch, and per-epoch COCO metrics.
"""

from __future__ import annotations

import contextlib
import io
import math
import time
from typing import Iterable

import torch
import torchvision.transforms.functional as TF
from pycocotools.cocoeval import COCOeval
from torch.utils.flop_counter import FlopCounterMode
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes, make_grid

from dataset import VOC_CLASSES

# COCOeval.stats indices.
COCO_METRICS = [
    ("AP",       0),  # mAP @ [0.5:0.95]
    ("AP50",     1),
    ("AP75",     2),
    ("AP_small", 3),
    ("AP_med",   4),
    ("AP_large", 5),
    ("AR_1",     6),
    ("AR_10",    7),
    ("AR_100",   8),
    ("AR_small", 9),
    ("AR_med",   10),
    ("AR_large", 11),
]


def log_compute_stats(
    model: torch.nn.Module,
    backbone: torch.nn.Module,
    img_size: int,
    device: torch.device,
    writer: SummaryWriter,
) -> dict[str, float]:
    """One-shot static stats: total/trainable params, model size (MB), backbone GFLOPs.

    FLOPs are counted on the backbone alone with a single ``[1, 3, img, img]``
    input — matching how ViT detection papers report backbone compute.  The
    counter treats the binary QK matmul as full-precision; the paper reports
    BOPs (binary ops) separately.
    """
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    was_training = backbone.training
    backbone.eval()
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fcm:
        backbone(x)
    backbone.train(was_training)
    gflops = fcm.get_total_flops() / 1e9

    writer.add_scalar("compute/params_M", n_params / 1e6, 0)
    writer.add_scalar("compute/trainable_params_M", n_trainable / 1e6, 0)
    writer.add_scalar("compute/model_size_MB", size_mb, 0)
    writer.add_scalar("compute/backbone_GFLOPs", gflops, 0)
    print(
        f"compute: params={n_params/1e6:.2f}M trainable={n_trainable/1e6:.2f}M "
        f"size={size_mb:.1f}MB backbone={gflops:.2f}GFLOPs @ {img_size}x{img_size}"
    )
    return {
        "params_M": n_params / 1e6,
        "trainable_params_M": n_trainable / 1e6,
        "model_size_MB": size_mb,
        "backbone_GFLOPs": gflops,
    }


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader: Iterable,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    log_interval: int = 20,
    grad_clip: float | None = 1.0,
) -> int:
    """Train for one epoch.  Returns the updated global step counter."""
    model.train()
    n_batches = len(loader)
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for it, (images, targets) in enumerate(loader):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at step {global_step}: {loss_dict}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip or float("inf"))
        optimizer.step()
        scheduler.step()

        if it % log_interval == 0 or it == n_batches - 1:
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/loss", loss.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)
            writer.add_scalar("train/lr", lr, global_step)
            writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
            print(
                f"epoch {epoch} [{it:>4d}/{n_batches}] "
                f"loss={loss.item():.4f} lr={lr:.2e} gn={float(grad_norm):.2f}"
            )

        global_step += 1

    writer.add_scalar("train/epoch_time_s", time.time() - t0, epoch)
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        writer.add_scalar("compute/train_peak_alloc_MB", peak_mb, epoch)
        writer.add_scalar("compute/train_peak_reserved_MB", reserved_mb, epoch)
        print(f"epoch {epoch} peak GPU mem: alloc={peak_mb:.0f}MB reserved={reserved_mb:.0f}MB")
    return global_step


_VIS_SCORE_THRESH = 0.3
_VIS_SIZE = 512


def _vis_detections(
    vis_data: list[tuple],
    epoch: int,
    writer: SummaryWriter,
) -> None:
    """Draw GT (green) and predicted (red) boxes on images and log a grid to TensorBoard."""
    imgs = []
    for img_cpu, pred_boxes, pred_scores, pred_labels, gt_boxes in vis_data:
        img_u8 = (img_cpu.clamp(0, 1) * 255).to(torch.uint8)
        _, orig_h, orig_w = img_u8.shape
        img_u8 = TF.resize(img_u8, [_VIS_SIZE, _VIS_SIZE])

        sx, sy = _VIS_SIZE / orig_w, _VIS_SIZE / orig_h

        def _scale(b: torch.Tensor) -> torch.Tensor:
            b = b.clone().float()
            b[:, [0, 2]] *= sx
            b[:, [1, 3]] *= sy
            return b

        if gt_boxes.numel():
            img_u8 = draw_bounding_boxes(img_u8, _scale(gt_boxes), colors="green", width=2)

        keep = pred_scores >= _VIS_SCORE_THRESH
        if keep.any():
            label_strs = [
                f"{VOC_CLASSES[l - 1] if 1 <= l <= len(VOC_CLASSES) else '?'} {s:.2f}"
                for l, s in zip(pred_labels[keep].tolist(), pred_scores[keep].tolist())
            ]
            img_u8 = draw_bounding_boxes(
                img_u8, _scale(pred_boxes[keep]), labels=label_strs, colors="red", width=2
            )

        imgs.append(img_u8)

    grid = make_grid(torch.stack(imgs), nrow=4)
    writer.add_image("val/detections", grid, epoch)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    num_vis_images: int = 8,
) -> dict[str, float]:
    """Run COCO-style evaluation and log the standard 12 metrics."""
    model.eval()
    coco_gt = loader.dataset.coco
    results: list[dict] = []
    vis_data: list[tuple] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    for images, targets in loader:
        images_dev = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images_dev)
        for img_cpu, tgt, out in zip(images, targets, outputs):
            if len(vis_data) < num_vis_images:
                vis_data.append((
                    img_cpu,
                    out["boxes"].cpu(),
                    out["scores"].cpu(),
                    out["labels"].cpu(),
                    tgt["boxes"],
                ))
            img_id = int(tgt["image_id"].item())
            boxes = out["boxes"].cpu()
            scores = out["scores"].cpu()
            labels = out["labels"].cpu()
            # xyxy -> xywh
            boxes_xywh = boxes.clone()
            boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
            for box, score, label in zip(boxes_xywh.tolist(), scores.tolist(), labels.tolist()):
                results.append(
                    {
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [round(c, 2) for c in box],
                        "score": float(score),
                    }
                )
    eval_time = time.time() - t0

    if vis_data:
        _vis_detections(vis_data, epoch, writer)

    if not results:
        print("eval: no detections produced; skipping COCO eval")
        return {}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = list(loader.dataset.ids)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    print(buf.getvalue())

    metrics = {name: float(coco_eval.stats[idx]) for name, idx in COCO_METRICS}
    for name, value in metrics.items():
        writer.add_scalar(f"val/{name}", value, epoch)
    writer.add_scalar("val/eval_time_s", eval_time, epoch)
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        writer.add_scalar("compute/val_peak_alloc_MB", peak_mb, epoch)
    print(
        f"epoch {epoch} val: AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f} "
        f"AP75={metrics['AP75']:.4f} ({eval_time:.1f}s)"
    )
    return metrics
