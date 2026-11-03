"""Import a YOLO-format export (Roboflow, CVAT, Label Studio, Roboflow Universe) as a MealLens source.

The export's class names are mapped onto MealLens classes by name (configs/classes.yaml aliases),
so its class order does not matter. Boxes of classes we do not use are dropped; an extra mapping
can rename or drop names explicitly, e.g. {"capsicum_red": "bell_pepper", "jar": null}.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import yaml

from . import classes
from .yolo_io import Box, label_path_for, list_images, read_labels, valid_box, write_labels, write_source_info


def export_names(src: Path) -> list[str]:
    """Class names from data.yaml (Roboflow/Ultralytics) or classes.txt / obj.names (CVAT, LabelImg)."""
    for candidate in ("data.yaml", "data.yml", "dataset.yaml"):
        path = src / candidate
        if path.exists():
            names = yaml.safe_load(path.read_text())["names"]
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    for candidate in ("classes.txt", "obj.names"):
        path = src / candidate
        if path.exists():
            return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    raise FileNotFoundError(f"{src}: no data.yaml, classes.txt or obj.names with the class names")


def build_mapping(names: list[str], extra: dict[str, str | None] | None = None) -> tuple[dict[int, int], list[str]]:
    """Export class id -> MealLens index, plus the export names we could not map."""
    extra = {classes.normalize(k): v for k, v in (extra or {}).items()}
    mapping, unmapped = {}, []
    for i, name in enumerate(names):
        key = classes.normalize(name)
        if key in extra:
            if extra[key] is None:
                continue
            target = classes.resolve(extra[key])
            if target is None:
                raise ValueError(f"mapping {name!r} -> {extra[key]!r}: target is not a MealLens class")
            mapping[i] = target
        elif (target := classes.resolve(name)) is not None:
            mapping[i] = target
        else:
            unmapped.append(name)
    return mapping, unmapped


def import_yolo(src: Path, out: Path, name: str, kind: str, extra_map: dict | None = None,
                prefix: str | None = None) -> dict:
    """Copy images and remapped labels from `src` into source directory `out`."""
    src, out = Path(src), Path(out)
    if kind not in ("public", "custom"):
        raise ValueError("kind must be 'public' or 'custom'")
    export = export_names(src)
    mapping, unmapped = build_mapping(export, extra_map)
    if not mapping:
        raise ValueError(f"none of {export} map to MealLens classes; pass a mapping file")

    prefix = prefix if prefix is not None else ("" if kind == "custom" else f"{name}_")
    counts: Counter = Counter()
    images = dropped_boxes = empty = 0
    for image in list_images(src):
        boxes = []
        for b in read_labels(label_path_for(image)):
            if b.cls in mapping and valid_box(b):
                boxes.append(Box(mapping[b.cls], b.cx, b.cy, b.w, b.h))
            else:
                dropped_boxes += 1
        # Custom photos keep empty images: they are valid negatives. Public images without
        # any of our classes are just noise.
        if not boxes and kind == "public":
            empty += 1
            continue
        stem = prefix + image.stem
        dst = out / "images" / (stem + image.suffix.lower())
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, dst)
        write_labels(out / "labels" / f"{stem}.txt", boxes)
        counts.update(classes.names()[b.cls] for b in boxes)
        images += 1

    info = dict(name=name, kind=kind, imported_from=str(src), images=images, boxes=dict(counts),
                dropped_boxes=dropped_boxes, skipped_empty_images=empty, unmapped_classes=unmapped)
    write_source_info(out, **info)
    return info
