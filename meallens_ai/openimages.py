"""Download the Open Images V7 images for our classes and convert them to a YOLO source directory.

Uses the FiftyOne dataset zoo, which fetches only the images that contain the requested classes.
Every downloaded image keeps the boxes of *all* our Open Images classes, not only the one it was
fetched for. Open Images is not exhaustively labelled for our other classes (an apple photo may
show an unlabelled onion), which is one more reason the custom fridge set matters.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

from . import classes
from .yolo_io import from_top_left, valid_box, write_labels, write_source_info


def _flag(det, name: str) -> bool:
    try:
        return bool(det[name])
    except (KeyError, AttributeError):
        return False


def _place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def download(out: Path, max_per_class: int = 400, split: str = "train", cache_dir: Path | None = None,
             seed: int = 51, keep_group_boxes: bool = False) -> Counter:
    """Fetch up to `max_per_class` images per class and write <out>/images, <out>/labels, source.json."""
    import fiftyone as fo
    import fiftyone.zoo as foz

    oi_map = classes.openimages_map()
    out = Path(out)
    counts: Counter = Counter()
    skipped: Counter = Counter()
    seen: set[str] = set()

    for oi_name in oi_map:
        name = f"meallens-oi-{split}-{oi_name}".replace(" ", "-").replace("(", "").replace(")", "")
        if fo.dataset_exists(name):
            fo.delete_dataset(name)
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=[oi_name],
            max_samples=max_per_class,
            shuffle=True,
            seed=seed,
            only_matching=False,
            dataset_name=name,
            dataset_dir=str(cache_dir) if cache_dir else None,
        )
        for sample in ds:
            stem = "oi_" + Path(sample.filepath).stem
            if stem in seen:
                continue
            dets = sample["ground_truth"].detections if sample["ground_truth"] else []
            ours = [d for d in dets if d.label in oi_map]
            if not keep_group_boxes and any(_flag(d, "IsGroupOf") for d in ours):
                skipped["group-of box"] += 1
                continue
            if any(_flag(d, "IsDepiction") for d in ours):
                skipped["drawing or depiction"] += 1
                continue
            boxes = [from_top_left(oi_map[d.label], *d.bounding_box) for d in ours]
            boxes = [b for b in boxes if valid_box(b)]
            if not boxes:
                skipped["no usable boxes"] += 1
                continue
            seen.add(stem)
            _place(Path(sample.filepath), out / "images" / f"{stem}.jpg")
            write_labels(out / "labels" / f"{stem}.txt", boxes)
            counts.update(classes.names()[b.cls] for b in boxes)
        ds.delete()

    write_source_info(out, name="openimages", kind="public", dataset="open-images-v7", split=split,
                      max_per_class=max_per_class, images=len(seen), boxes=dict(counts),
                      skipped=dict(skipped))
    return counts
