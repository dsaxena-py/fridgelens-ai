"""Fine-tune a pretrained YOLO model from a config file (configs/train_*.yaml)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml


def load_config(path: Path, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key in ("model", "data"):
        if key not in cfg:
            raise ValueError(f"{path}: '{key}' is required")
    return cfg


def train(config: Path, overrides: dict | None = None) -> Path:
    """Train and return the path to best.pt. The resolved config is saved next to the weights."""
    from ultralytics import YOLO

    cfg = load_config(config, overrides)
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    (save_dir / "meallens_config.json").write_text(json.dumps({"config": str(config), **cfg}, indent=2, default=str))
    shutil.copy2(config, save_dir / Path(config).name)
    print(f"best weights: {best}")
    return best
