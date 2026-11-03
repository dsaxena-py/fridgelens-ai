"""MealLens training pipeline command line. Run `python -m meallens_ai --help`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def _device(value: str | None):
    return None if value in (None, "", "auto") else value


def cmd_classes(args) -> None:
    from .classes import load_classes

    for c in load_classes():
        oi = ", ".join(c.openimages) or "-"
        print(f"{c.index:>2}  {c.slug:<13} Open Images: {oi}")


def cmd_download_openimages(args) -> None:
    from .openimages import download

    counts = download(Path(args.out), args.max_per_class, args.split, Path(args.cache) if args.cache else None)
    print(json.dumps(dict(counts), indent=2))


def cmd_import_yolo(args) -> None:
    from .importer import import_yolo

    extra = yaml.safe_load(Path(args.map).read_text()) if args.map else None
    info = import_yolo(Path(args.src), Path(args.out), args.name, args.kind, extra)
    print(json.dumps(info, indent=2))
    if info["unmapped_classes"]:
        print(f"\nThese classes were dropped; map them with --map if they are ours: {info['unmapped_classes']}",
              file=sys.stderr)


def cmd_build_dataset(args) -> None:
    from .dataset import build, format_stats

    stats = build([Path(s) for s in args.sources], Path(args.out), args.seed, args.link)
    print(format_stats(stats))
    print(f"\nwrote {Path(args.out) / 'data.yaml'}")
    if stats["errors"]:
        sys.exit(1)


def cmd_check_dataset(args) -> None:
    from .dataset import check, format_stats

    stats = check(Path(args.dataset), args.min_train)
    print(format_stats(stats))
    if stats["errors"]:
        sys.exit(1)


def cmd_train(args) -> None:
    from .train import train

    overrides = {"data": args.data, "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
                 "device": _device(args.device), "name": args.name, "project": args.project,
                 "fraction": args.fraction}
    train(Path(args.config), overrides)


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate

    report = evaluate(Path(args.weights), Path(args.data), args.split, args.conf, args.imgsz, args.nms_iou,
                      _device(args.device), Path(args.onnx) if args.onnx else None, Path(args.out))
    m = report["map"]
    print(f"{args.split}: mAP50 {m['map50']:.3f}  mAP50-95 {m['map50_95']:.3f}  "
          f"macro P {report['macro']['precision']:.3f} R {report['macro']['recall']:.3f} F1 {report['macro']['f1']:.3f} "
          f"at conf {report['conf_threshold']}")
    print(f"report: {Path(args.out) / f'report_{args.split}.md'}")


def cmd_tune_threshold(args) -> None:
    from .classes import names
    from .evaluate import collect_predictions
    from .metrics import sweep

    pairs = collect_predictions(Path(args.weights), Path(args.data), args.split, args.imgsz, args.nms_iou,
                                device=_device(args.device))
    best, rows = sweep(pairs, len(names()))
    print("conf   macro P  macro R  macro F1")
    for r in rows:
        mark = "  <- best" if r["conf"] == best else ""
        print(f"{r['conf']:.2f}   {r['macro_precision']:.3f}    {r['macro_recall']:.3f}    {r['macro_f1']:.3f}{mark}")
    print(f"\nbest confidence threshold on {args.split}: {best}")


def cmd_export(args) -> None:
    from .export import export, metrics_from_report

    metrics = metrics_from_report(json.loads(Path(args.report).read_text())) if args.report else None
    card = export(Path(args.weights), args.name, Path(args.out), args.conf, args.nms_iou, args.imgsz,
                  metrics, args.weights_uri)
    print(f"wrote {Path(args.out) / (args.name + '.onnx')}, model_card.json and register_model.sql")
    print(json.dumps(card["metrics"], indent=2))


def cmd_prelabel(args) -> None:
    from .prelabel import prelabel

    info = prelabel(Path(args.weights), Path(args.images), Path(args.out), args.conf, args.imgsz, _device(args.device))
    print(json.dumps(info, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m meallens_ai", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("classes", help="list the 25 classes and their Open Images names")
    s.set_defaults(func=cmd_classes)

    s = sub.add_parser("download-openimages", help="download Open Images V7 images for our classes")
    s.add_argument("--out", default="data/sources/openimages")
    s.add_argument("--max-per-class", type=int, default=400)
    s.add_argument("--split", default="train", choices=["train", "validation", "test"])
    s.add_argument("--cache", help="FiftyOne download cache (default ~/fiftyone)")
    s.set_defaults(func=cmd_download_openimages)

    s = sub.add_parser("import-yolo", help="import a YOLO export (Roboflow, CVAT, Universe) as a source")
    s.add_argument("src")
    s.add_argument("--out", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--kind", required=True, choices=["custom", "public"],
                   help="custom = your own photos (used for test); public = anything else")
    s.add_argument("--map", help="YAML of extra name mappings, e.g. {capsicum_red: bell_pepper, jar: null}")
    s.set_defaults(func=cmd_import_yolo)

    s = sub.add_parser("build-dataset", help="merge sources into one dataset with scene-level splits")
    s.add_argument("sources", nargs="+")
    s.add_argument("--out", default="data/meallens")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--link", default="hardlink", choices=["hardlink", "symlink", "copy"])
    s.set_defaults(func=cmd_build_dataset)

    s = sub.add_parser("check-dataset", help="per-class counts and label problems")
    s.add_argument("dataset")
    s.add_argument("--min-train", type=int, default=50)
    s.set_defaults(func=cmd_check_dataset)

    s = sub.add_parser("train", help="fine-tune YOLO from a config")
    s.add_argument("--config", default="configs/train_baseline.yaml")
    s.add_argument("--data")
    s.add_argument("--epochs", type=int)
    s.add_argument("--batch", type=int)
    s.add_argument("--imgsz", type=int)
    s.add_argument("--device", help="0, 0,1, cpu or auto")
    s.add_argument("--name")
    s.add_argument("--project")
    s.add_argument("--fraction", type=float, help="train on a fraction of the images (quick experiments)")
    s.set_defaults(func=cmd_train)

    for name, help_text in (("evaluate", "full report for a split"), ("tune-threshold", "pick the confidence threshold")):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--weights", required=True)
        s.add_argument("--data", default="data/meallens/data.yaml")
        s.add_argument("--split", default="test" if name == "evaluate" else "val")
        s.add_argument("--imgsz", type=int, default=640)
        s.add_argument("--nms-iou", type=float, default=0.5)
        s.add_argument("--device")
        if name == "evaluate":
            s.add_argument("--conf", type=float, help="deployment threshold (default: best macro F1 on this split)")
            s.add_argument("--onnx", help="exported model, to measure CPU latency")
            s.add_argument("--out", default="reports")
            s.set_defaults(func=cmd_evaluate)
        else:
            s.set_defaults(func=cmd_tune_threshold)

    s = sub.add_parser("export", help="ONNX + model card + registration SQL")
    s.add_argument("--weights", required=True)
    s.add_argument("--name", required=True, help="e.g. meallens-yolo11s-v1")
    s.add_argument("--out", default="exports")
    s.add_argument("--conf", type=float, required=True, help="from tune-threshold on val")
    s.add_argument("--nms-iou", type=float, default=0.5)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--report", help="report_test.json from evaluate, stored as the model's metrics")
    s.add_argument("--weights-uri", help="where the backend will load the .onnx from")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("prelabel", help="draft labels for new photos with a trained model")
    s.add_argument("--weights", required=True)
    s.add_argument("--images", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--conf", type=float, default=0.4)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--device")
    s.set_defaults(func=cmd_prelabel)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
