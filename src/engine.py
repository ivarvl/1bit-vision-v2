"""Train one epoch and evaluate (COCO mAP) for the BinaryAttention detector.

All scalars are streamed to a ``torch.utils.tensorboard.SummaryWriter`` provided
by the caller — per-iteration training losses + LR + grad-norm, and per-epoch
COCO metrics.
"""

from __future__ import annotations

import contextlib
import io
import math
import time
from typing import Iterable

import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.tensorboard import SummaryWriter

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
    return global_step


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> dict[str, float]:
    """Run COCO-style evaluation and log the standard 12 metrics."""
    model.eval()
    coco_gt = loader.dataset.coco
    results: list[dict] = []

    t0 = time.time()
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        for tgt, out in zip(targets, outputs):
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
    print(
        f"epoch {epoch} val: AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f} "
        f"AP75={metrics['AP75']:.4f} ({eval_time:.1f}s)"
    )
    return metrics
