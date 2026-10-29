"""End-to-end smoke test: synthetic data -> train 1 epoch on CPU -> export ONNX -> infer -> evaluate.

Needs ultralytics + onnxruntime (requirements.txt); skipped otherwise. Takes a minute or two on CPU.
The model learns nothing useful in one epoch; the point is that every stage runs and the
files each stage hands to the next (data.yaml, best.pt, .onnx, model_card.json, SQL) are right.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

pytest.importorskip("ultralytics")
pytest.importorskip("onnxruntime")

from meallens_ai import classes  # noqa: E402
from meallens_ai.dataset import build  # noqa: E402
from meallens_ai.evaluate import evaluate  # noqa: E402
from meallens_ai.export import export, metrics_from_report  # noqa: E402
from meallens_ai.onnx_infer import Detector  # noqa: E402
from meallens_ai.prelabel import prelabel  # noqa: E402
from meallens_ai.train import train  # noqa: E402
from meallens_ai.yolo_io import Box, write_labels, write_source_info  # noqa: E402

COLORS = {0: (220, 40, 40), 18: (240, 240, 220), 24: (190, 140, 70)}  # apple, egg, bread


def synthetic_source(root: Path, scenes: int = 12, shots: int = 3, size: int = 96) -> Path:
    rng = random.Random(0)
    for scene in range(scenes):
        for shot in range(shots):
            img = Image.new("RGB", (size, size), (90, 90, 100))
            draw = ImageDraw.Draw(img)
            boxes = []
            for cls, color in rng.sample(list(COLORS.items()), 2):
                w, h = rng.randint(20, 40), rng.randint(20, 40)
                x, y = rng.randint(0, size - w), rng.randint(0, size - h)
                draw.rectangle([x, y, x + w, y + h], fill=color)
                boxes.append(Box(cls, (x + w / 2) / size, (y + h / 2) / size, w / size, h / size))
            stem = f"zz-{scene:03d}-{shot:02d}"
            (root / "images").mkdir(parents=True, exist_ok=True)
            img.save(root / "images" / f"{stem}.jpg")
            write_labels(root / "labels" / f"{stem}.txt", boxes)
    write_source_info(root, name="synthetic", kind="custom")
    return root


def test_end_to_end(tmp_path):
    source = synthetic_source(tmp_path / "sources" / "synthetic")
    stats = build([source], tmp_path / "dataset", link="copy")
    assert not stats["errors"] and "test" in stats["splits"]
    data_yaml = tmp_path / "dataset" / "data.yaml"

    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump({
        "model": "yolo11n.yaml",  # from scratch: no weight download in CI
        "data": str(data_yaml), "epochs": 1, "imgsz": 64, "batch": 8, "device": "cpu", "workers": 0,
        "project": str(tmp_path / "runs"), "name": "smoke", "plots": False, "amp": False, "val": True,
    }))
    best = train(config)
    assert best.exists()

    out = tmp_path / "export"
    card = export(best, "meallens-smoke", out, conf=0.01, imgsz=64)
    assert (out / "meallens-smoke.onnx").exists()
    assert [c["slug"] for c in card["classes"]] == classes.names()
    assert "insert into detector_classes" in (out / "register_model.sql").read_text()

    detector = Detector(out / "meallens-smoke.onnx")
    first_image = sorted((tmp_path / "dataset" / "images" / "test").glob("*.jpg"))[0]
    for d in detector(first_image):
        assert 0 <= d.x_min < d.x_max <= 1 and 0 <= d.y_min < d.y_max <= 1 and d.label in classes.names()

    report = evaluate(best, data_yaml, "test", conf=0.01, imgsz=64, device="cpu",
                      onnx=out / "meallens-smoke.onnx", out=tmp_path / "reports")
    assert report["images"] > 0 and "cpu_ms_p50" in report["latency"]
    assert (tmp_path / "reports" / "report_test.md").read_text().startswith("# MealLens detector")
    assert set(metrics_from_report(report)) >= {"precision", "recall", "f1", "map50", "map50_95"}

    info = prelabel(best, tmp_path / "dataset" / "images" / "test", tmp_path / "prelabel", conf=0.01, imgsz=64,
                    device="cpu")
    assert info["images"] > 0 and (tmp_path / "prelabel" / "data.yaml").exists()
