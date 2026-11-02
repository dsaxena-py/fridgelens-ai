"""Merge source directories into one YOLO dataset with leak-free splits, and check it.

Split rules (proposal §4):
* Custom fridge photos are split 70/15/15 **by scene**, so near-identical shots of one shelf never
  land in both train and test. The scene is read from the file name (see SCENE_PATTERN).
* Public images go 90/10 to train/val and never to test: the test set measures the real domain.
* A scene's split is a hash of its id, so adding new photos later never moves existing ones.
  Small datasets are the exception: if val or test would be empty, the train scene closest to
  that split's hash range is moved into it, so training always has a validation set.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import classes
from .yolo_io import label_path_for, list_images, read_labels, read_source_info, valid_box

SPLITS = ("train", "val", "test")
CUSTOM_RATIOS = (0.70, 0.15, 0.15)
PUBLIC_VAL_RATIO = 0.10
# "<initials>-<scene number>-<shot number>", e.g. ds-014-03.jpg. Roboflow keeps this prefix when it
# renames files (ds-014-03_jpg.rf.<hash>.jpg), so the scene survives annotation.
SCENE_PATTERN = re.compile(r"^([A-Za-z]+-\d+)")


@dataclass
class Item:
    image: Path
    source: str
    kind: str
    group: str
    split: str = ""


def scene_of(stem: str) -> str | None:
    m = SCENE_PATTERN.match(stem)
    return m.group(1).lower() if m else None


def unit_hash(key: str, seed: int) -> float:
    """Deterministic value in [0, 1) for a key."""
    digest = hashlib.sha1(f"{seed}:{key}".encode()).hexdigest()
    return int(digest[:12], 16) / 16 ** 12


def assign_split(kind: str, group: str, seed: int) -> str:
    u = unit_hash(group, seed)
    if kind == "public":
        return "val" if u < PUBLIC_VAL_RATIO else "train"
    train, val, _ = CUSTOM_RATIOS
    return "train" if u < train else "val" if u < train + val else "test"


def collect(sources: list[Path], seed: int) -> tuple[list[Item], list[str]]:
    items, warnings = [], []
    stems: dict[str, str] = {}
    for src in sources:
        info = read_source_info(src)
        name, kind = info["name"], info["kind"]
        no_scene = 0
        for image in list_images(Path(src) / "images"):
            if image.stem in stems:
                raise ValueError(f"duplicate image name {image.stem} in {name} and {stems[image.stem]}")
            stems[image.stem] = name
            if kind == "custom":
                group = scene_of(image.stem)
                if group is None:
                    no_scene += 1
                    group = image.stem
            else:
                group = image.stem
            item = Item(image=image, source=name, kind=kind, group=f"{name}/{group}")
            item.split = assign_split(kind, item.group, seed)
            items.append(item)
        fill_empty_splits([it for it in items if it.source == name], kind, seed)
        if no_scene:
            warnings.append(f"{name}: {no_scene} custom images do not follow the '<initials>-<scene>-<shot>' "
                            "name pattern; each is treated as its own scene, so near-duplicates may leak "
                            "between splits")
    return items, warnings


def fill_empty_splits(items: list[Item], kind: str, seed: int) -> None:
    """Move whole train groups into val (and, for custom data, test) when those would be empty."""
    wanted = ["val"] if kind == "public" else ["val", "test"]
    for split in wanted:
        if any(it.split == split for it in items):
            continue
        train_groups = sorted({it.group for it in items if it.split == "train"}, key=lambda g: -unit_hash(g, seed))
        if len(train_groups) < 2:
            continue  # keep at least one group for training
        moved = train_groups[0]
        for it in items:
            if it.group == moved:
                it.split = split


def _place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def build(sources: list[Path], out: Path, seed: int = 0, link: str = "hardlink") -> dict:
    """Write <out>/images/<split>/, <out>/labels/<split>/, data.yaml, manifest.csv and stats.json."""
    out = Path(out)
    items, warnings = collect([Path(s) for s in sources], seed)
    if not items:
        raise ValueError("no images found in the sources")
    if not any(it.split == "val" for it in items):
        raise ValueError("not enough images for a validation split; add more scenes or a public source")
    for split in SPLITS:
        for sub in ("images", "labels"):
            shutil.rmtree(out / sub / split, ignore_errors=True)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "source", "kind", "group", "split"])
        for it in items:
            _place(it.image, out / "images" / it.split / it.image.name, link)
            label = label_path_for(it.image)
            dst = out / "labels" / it.split / f"{it.image.stem}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if label.exists():
                shutil.copy2(label, dst)
            else:
                dst.write_text("")
            writer.writerow([it.image.name, it.source, it.kind, it.group, it.split])

    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
            "names": dict(enumerate(classes.names()))}
    if any(it.split == "test" for it in items):
        data["test"] = "images/test"
    else:
        warnings.append("no test split yet: it is made only of custom fridge photos")
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    stats = check(out)
    stats["warnings"] = warnings + stats["warnings"]
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def check(dataset: Path, min_train_instances: int = 50) -> dict:
    """Per-split image and box counts, label problems and class-balance warnings."""
    dataset = Path(dataset)
    names = classes.names()
    stats: dict = {"splits": {}, "warnings": [], "errors": []}
    instances: dict[str, Counter] = defaultdict(Counter)
    for split in SPLITS:
        images = list_images(dataset / "images" / split) if (dataset / "images" / split).exists() else []
        if not images:
            continue
        empty = 0
        for image in images:
            label = dataset / "labels" / split / f"{image.stem}.txt"
            try:
                boxes = read_labels(label)
            except ValueError as e:
                stats["errors"].append(str(e))
                continue
            if not boxes:
                empty += 1
            for b in boxes:
                if not 0 <= b.cls < len(names):
                    stats["errors"].append(f"{label}: class {b.cls} out of range")
                elif not valid_box(b):
                    stats["errors"].append(f"{label}: box outside the image or too small: {b.line()}")
                else:
                    instances[split][names[b.cls]] += 1
        stats["splits"][split] = {"images": len(images), "images_without_boxes": empty,
                                  "boxes": sum(instances[split].values()),
                                  "per_class": {n: instances[split][n] for n in names}}
    for n in names:
        if instances["train"][n] < min_train_instances:
            stats["warnings"].append(f"{n}: only {instances['train'][n]} training boxes (aim for {min_train_instances}+)")
        if "val" in stats["splits"] and instances["val"][n] == 0:
            stats["warnings"].append(f"{n}: no validation boxes, so its val metrics are undefined")
        if "test" in stats["splits"] and instances["test"][n] == 0:
            stats["warnings"].append(f"{n}: no test boxes in the fridge test set")
    return stats


def format_stats(stats: dict) -> str:
    names = classes.names()
    splits = [s for s in SPLITS if s in stats["splits"]]
    lines = ["class".ljust(14) + "".join(s.rjust(8) for s in splits)]
    for n in names:
        lines.append(n.ljust(14) + "".join(str(stats["splits"][s]["per_class"][n]).rjust(8) for s in splits))
    lines.append("boxes".ljust(14) + "".join(str(stats["splits"][s]["boxes"]).rjust(8) for s in splits))
    lines.append("images".ljust(14) + "".join(str(stats["splits"][s]["images"]).rjust(8) for s in splits))
    for kind in ("errors", "warnings"):
        if stats.get(kind):
            lines.append(f"\n{len(stats[kind])} {kind}:")
            lines += [f"  - {m}" for m in stats[kind][:40]]
            if len(stats[kind]) > 40:
                lines.append(f"  ... and {len(stats[kind]) - 40} more")
    return "\n".join(lines)
