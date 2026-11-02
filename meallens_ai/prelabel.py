"""Model-assisted labelling: draft YOLO labels for new photos so annotators only correct them.

Writes <out>/images, <out>/labels and data.yaml, which Roboflow, CVAT and Label Studio all import
as a YOLO dataset. Every draft box still needs a human check before it is used for training.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from . import classes
from .yolo_io import Box, list_images, write_labels


def prelabel(weights: Path, images_dir: Path, out: Path, conf: float = 0.4, imgsz: int = 640, device=None) -> dict:
    from ultralytics import YOLO

    out = Path(out)
    images = list_images(images_dir)
    model = YOLO(str(weights))
    (out / "images").mkdir(parents=True, exist_ok=True)
    total = 0
    for image, result in zip(images, model.predict([str(p) for p in images], conf=conf, imgsz=imgsz,
                                                    stream=True, verbose=False, device=device)):
        boxes = [Box(int(c), *map(float, xywh)) for c, xywh in zip(result.boxes.cls.tolist(), result.boxes.xywhn.tolist())]
        shutil.copy2(image, out / "images" / image.name)
        write_labels(out / "labels" / f"{image.stem}.txt", boxes)
        total += len(boxes)
    (out / "data.yaml").write_text(yaml.safe_dump({"names": classes.names(), "nc": len(classes.names())}, sort_keys=False))
    return {"images": len(images), "draft_boxes": total}
