"""Reading and writing YOLO-format datasets.

A "source" directory (one per dataset we pull in) looks like:
    <source>/images/<stem>.jpg
    <source>/labels/<stem>.txt      # "<class> <cx> <cy> <w> <h>" per line, normalised, MealLens indices
    <source>/source.json            # {"name": ..., "kind": "public" | "custom", ...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)

    def line(self) -> str:
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def from_top_left(cls: int, x: float, y: float, w: float, h: float) -> Box:
    """Normalised [x, y, w, h] with a top-left origin (Open Images / FiftyOne) -> YOLO box, clipped to the image."""
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(1.0, x + w), min(1.0, y + h)
    return Box(cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


def read_labels(path: Path) -> list[Box]:
    if not path.exists():
        return []
    boxes = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{n}: expected 5 values, got {len(parts)} (polygon labels are not supported)")
        boxes.append(Box(int(parts[0]), *map(float, parts[1:])))
    return boxes


def write_labels(path: Path, boxes: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(b.line() + "\n" for b in boxes))


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def label_path_for(image: Path) -> Path:
    """YOLO convention: .../images/x.jpg -> .../labels/x.txt."""
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def write_source_info(source_dir: Path, **info) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "source.json").write_text(json.dumps(info, indent=2) + "\n")


def read_source_info(source_dir: Path) -> dict:
    path = Path(source_dir) / "source.json"
    if not path.exists():
        raise FileNotFoundError(f"{source_dir} has no source.json; create it with import-yolo or download-openimages")
    return json.loads(path.read_text())


def valid_box(b: Box, min_size: float = 0.002) -> bool:
    """Inside the image and not degenerate."""
    x0, y0, x1, y1 = b.to_xyxy()
    eps = 1e-6
    return (b.w > min_size and b.h > min_size and x0 >= -eps and y0 >= -eps
            and x1 <= 1 + eps and y1 <= 1 + eps)
