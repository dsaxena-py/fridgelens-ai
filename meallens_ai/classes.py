"""The MealLens class list (configs/classes.yaml) and name resolution for external datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CLASSES_FILE = Path(__file__).resolve().parent.parent / "configs" / "classes.yaml"


@dataclass(frozen=True)
class DetectorClass:
    index: int
    slug: str
    name: str
    openimages: tuple[str, ...] = field(default_factory=tuple)
    aliases: tuple[str, ...] = field(default_factory=tuple)


def normalize(label: str) -> str:
    """'Bell_Pepper ' -> 'bell pepper'."""
    return re.sub(r"\s+", " ", re.sub(r"[_\-]", " ", label)).strip().lower()


@lru_cache(maxsize=None)
def load_classes(path: Path = CLASSES_FILE) -> tuple[DetectorClass, ...]:
    rows = yaml.safe_load(Path(path).read_text())
    classes = tuple(
        DetectorClass(
            index=r["index"],
            slug=r["slug"],
            name=r["name"],
            openimages=tuple(r.get("openimages", [])),
            aliases=tuple(r.get("aliases", [])),
        )
        for r in rows
    )
    if [c.index for c in classes] != list(range(len(classes))):
        raise ValueError(f"{path}: class indices must be 0..n-1 in order")
    return classes


def names() -> list[str]:
    """Class slugs in index order, as YOLO `names`."""
    return [c.slug for c in load_classes()]


@lru_cache(maxsize=None)
def _lookup() -> dict[str, int]:
    table: dict[str, int] = {}
    for c in load_classes():
        for label in (c.slug, c.name, *c.aliases, *c.openimages):
            key = normalize(label)
            if table.get(key, c.index) != c.index:
                raise ValueError(f"name '{label}' maps to two classes")
            table[key] = c.index
    return table


def resolve(label: str) -> int | None:
    """Map a class name from any dataset to a MealLens class index, or None if it is not ours."""
    key = normalize(label)
    table = _lookup()
    if key in table:
        return table[key]
    if key.endswith("s") and key[:-1] in table:  # simple plural
        return table[key[:-1]]
    return None


def openimages_map() -> dict[str, int]:
    """Open Images class name -> MealLens class index."""
    return {oi: c.index for c in load_classes() for oi in c.openimages}
