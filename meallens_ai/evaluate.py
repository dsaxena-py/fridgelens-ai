"""Evaluate a trained model: mAP (Ultralytics), P/R/F1 at the deployment threshold, threshold sweep, latency."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import yaml

from . import classes
from .metrics import Pred, Truth, score, sweep
from .yolo_io import list_images, read_labels


def _split_dir(data_yaml: Path, split: str) -> tuple[Path, Path]:
    data = yaml.safe_load(Path(data_yaml).read_text())
    if split not in data:
        raise ValueError(f"{data_yaml} has no '{split}' split")
    root = Path(data.get("path", Path(data_yaml).parent))
    images = root / data[split]
    return images, root / "labels" / images.name


def collect_predictions(weights: Path, data_yaml: Path, split: str, imgsz: int = 640, nms_iou: float = 0.5,
                        min_conf: float = 0.05, device=None) -> list[tuple[list[Pred], list[Truth]]]:
    """Predictions at a low confidence plus ground truth for every image of a split, all normalised xyxy."""
    from ultralytics import YOLO

    images_dir, labels_dir = _split_dir(data_yaml, split)
    images = list_images(images_dir)
    model = YOLO(str(weights))
    pairs = []
    for image, result in zip(images, model.predict([str(p) for p in images], conf=min_conf, iou=nms_iou,
                                                    imgsz=imgsz, stream=True, verbose=False, device=device)):
        b = result.boxes
        preds = [Pred(int(c), float(s), tuple(map(float, xy)))
                 for c, s, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxyn.tolist())]
        truths = [Truth(t.cls, t.to_xyxy()) for t in read_labels(labels_dir / f"{image.stem}.txt")]
        pairs.append((preds, truths))
    return pairs


def ultralytics_map(weights: Path, data_yaml: Path, split: str, imgsz: int = 640, device=None,
                    project: str | None = None) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), split=split, imgsz=imgsz, plots=True, device=device,
                              project=project, name=f"eval-{split}", exist_ok=True, verbose=False)
    names = classes.names()
    per_class = {names[int(c)]: {"ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i])}
                 for i, c in enumerate(m.box.ap_class_index)}
    return {"map50": float(m.box.map50), "map50_95": float(m.box.map), "per_class": per_class,
            "plots_dir": str(m.save_dir)}


def benchmark_onnx(onnx_path: Path, images: list[Path], runs: int = 30) -> dict:
    """End-to-end CPU latency of onnx_infer.Detector (preprocess + model + NMS), in milliseconds."""
    from .onnx_infer import Detector

    det = Detector(onnx_path)
    sample = images[:runs] or []
    if not sample:
        return {}
    det(sample[0])  # warm-up
    times = []
    for image in sample:
        t0 = time.perf_counter()
        det(image)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"cpu_ms_p50": round(statistics.median(times), 1),
            "cpu_ms_p95": round(times[min(len(times) - 1, int(0.95 * len(times)))], 1), "runs": len(times)}


def evaluate(weights: Path, data_yaml: Path, split: str = "test", conf: float | None = None, imgsz: int = 640,
             nms_iou: float = 0.5, device=None, onnx: Path | None = None, out: Path | None = None) -> dict:
    """Full report for one split. With conf=None the best macro-F1 threshold on this split is used."""
    names = classes.names()
    pairs = collect_predictions(weights, data_yaml, split, imgsz, nms_iou, device=device)
    best_conf, table = sweep(pairs, len(names))
    conf = best_conf if conf is None else conf
    s = score(pairs, conf, len(names))
    report = {
        "split": split, "images": len(pairs), "weights": str(weights), "conf_threshold": conf,
        "nms_iou": nms_iou, "best_conf_on_this_split": best_conf,
        "macro": s["macro"], "micro": s["micro"],
        "per_class": {names[c]: v for c, v in s["per_class"].items()},
        "threshold_sweep": table,
        "map": ultralytics_map(weights, data_yaml, split, imgsz, device),
    }
    if onnx:
        images_dir, _ = _split_dir(data_yaml, split)
        report["latency"] = benchmark_onnx(onnx, list_images(images_dir))
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"report_{split}.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / f"report_{split}.md").write_text(to_markdown(report))
    return report


def to_markdown(r: dict) -> str:
    """Report tables for the paper: overall metrics, then one row per class."""
    lines = [f"# MealLens detector, {r['split']} split ({r['images']} images)", "",
             f"Confidence threshold {r['conf_threshold']}, NMS IoU {r['nms_iou']}, match IoU 0.5.", "",
             "| Metric | Value |", "| --- | --- |",
             f"| mAP@0.5 | {r['map']['map50']:.3f} |", f"| mAP@0.5:0.95 | {r['map']['map50_95']:.3f} |",
             f"| Macro precision | {r['macro']['precision']:.3f} |", f"| Macro recall | {r['macro']['recall']:.3f} |",
             f"| Macro F1 | {r['macro']['f1']:.3f} |", f"| Micro F1 | {r['micro']['f1']:.3f} |"]
    if r.get("latency"):
        lines.append(f"| CPU latency p50 / p95 (ms) | {r['latency']['cpu_ms_p50']} / {r['latency']['cpu_ms_p95']} |")
    lines += ["", "| Class | Support | Precision | Recall | F1 | AP50 | AP50-95 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, v in r["per_class"].items():
        ap = r["map"]["per_class"].get(name, {})
        ap50 = f"{ap['ap50']:.3f}" if ap else "–"
        ap95 = f"{ap['ap50_95']:.3f}" if ap else "–"
        lines.append(f"| {name} | {v['support']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {ap50} | {ap95} |")
    return "\n".join(lines) + "\n"
