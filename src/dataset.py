"""PASCAL VOC 2012 detection dataset for torchvision Faster R-CNN.

Reads Pascal VOC XML annotations. Returns the same ``(image, target)`` dict
that torchvision expects: ``boxes`` (xyxy), ``labels``, ``image_id``, ``area``,
``iscrowd``.  No masks — detection only.

``difficult`` annotations are kept but exposed as ``iscrowd=1`` so the COCO
evaluator ignores them (same semantics: don't penalise missed detections).

Directory layout expected::

    <root>/
      VOC2012_train_val/VOC2012_train_val/
        Annotations/    *.xml
        JPEGImages/     *.jpg
        ImageSets/Main/
          train.txt
          val.txt
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

_NUM_CLASSES = len(VOC_CLASSES)  # 20
_CLASS_TO_ID = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}


def _parse_voc_xml(ann_path: Path) -> tuple[int, int, list[tuple]]:
    """Parse a VOC XML file.

    Returns ``(width, height, [(x1, y1, x2, y2, cat_id, difficult), ...])``.
    Only top-level ``<object>`` elements are parsed; ``<part>`` sub-elements
    are skipped.  Objects whose name is not in the 20 VOC classes are ignored.
    """
    root = ET.parse(ann_path).getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        cat_id = _CLASS_TO_ID.get(name)
        if cat_id is None:
            continue
        difficult = int(obj.find("difficult").text)
        bb = obj.find("bndbox")
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        objects.append((x1, y1, x2, y2, cat_id, difficult))
    return w, h, objects


def _build_coco_index(
    data_dir: Path, split: str, skip_empty: bool
) -> tuple[COCO, list[int]]:
    """Build an in-memory pycocotools COCO object from VOC XML annotations.

    Image dimensions come from the XML ``<size>`` block — no image headers
    need to be opened at index time.
    """
    split_file = data_dir / "ImageSets" / "Main" / f"{split}.txt"
    stems = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]

    coco_images: list[dict] = []
    coco_anns: list[dict] = []
    ann_id = 1
    valid_ids: list[int] = []

    for img_id, stem in enumerate(stems, start=1):
        ann_path = data_dir / "Annotations" / f"{stem}.xml"
        img_w, img_h, objs = _parse_voc_xml(ann_path)

        if skip_empty and not objs:
            continue

        coco_images.append(
            {"id": img_id, "file_name": f"{stem}.jpg", "width": img_w, "height": img_h}
        )
        valid_ids.append(img_id)

        for x1, y1, x2, y2, cat_id, difficult in objs:
            coco_anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": difficult,
                }
            )
            ann_id += 1

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(VOC_CLASSES)]
    coco = COCO()
    coco.dataset = {
        "images": coco_images,
        "annotations": coco_anns,
        "categories": categories,
    }
    coco.createIndex()
    return coco, valid_ids


class VOC2012Detection(Dataset):
    """PASCAL VOC 2012 detection dataset wrapper for Faster R-CNN.

    Args:
        root: directory containing the ``VOC2012_train_val/`` folder.
        split: ``"train"`` or ``"val"``.
        transforms: callable applied to ``(image_tensor, target_dict)``.
        skip_empty: drop images with no annotations (recommended for training).
    """

    # VOC category IDs 1-20 are contiguous; identity mapping.
    cat_id_map: dict[int, int] = {i: i for i in range(1, _NUM_CLASSES + 1)}
    num_classes: int = _NUM_CLASSES + 1  # +1 for background

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms=None,
        skip_empty: bool = True,
    ):
        data_dir = Path(root) / "VOC2012_train_val" / "VOC2012_train_val"
        self.image_dir = data_dir / "JPEGImages"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"image directory not found: {self.image_dir}")

        self.coco, self.ids = _build_coco_index(data_dir, split, skip_empty)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        from PIL import Image

        image = Image.open(self.image_dir / info["file_name"]).convert("RGB")

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
            areas.append(a["area"])
            iscrowd.append(a["iscrowd"])

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            area_t = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor(img_id, dtype=torch.int64),
            "area": area_t,
            "iscrowd": iscrowd_t,
        }

        image_t = to_tensor(image)  # CHW float32 in [0, 1]

        if self.transforms is not None:
            image_t, target = self.transforms(image_t, target)
        return image_t, target
