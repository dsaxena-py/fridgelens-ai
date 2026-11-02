"""Detection precision / recall / F1 at a fixed confidence threshold, and threshold selection.

Ultralytics reports mAP well, but its per-class P and R are taken at whichever confidence
maximises its own F1 curve. The app runs at one fixed threshold, so we also score predictions
at exactly that threshold: a prediction is a true positive when it overlaps an unmatched ground
truth box of the same class with IoU >= 0.5 (greedy, highest confidence first).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

XYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class Pred:
    cls: int
    conf: float
    box: XYXY


@dataclass(frozen=True)
class Truth:
    cls: int
    box: XYXY


def iou(a: XYXY, b: XYXY) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_image(preds: list[Pred], truths: list[Truth], conf: float, iou_thr: float = 0.5) -> dict[int, list[int]]:
    """{class: [tp, fp, fn]} for one image, using only predictions with confidence >= conf."""
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    used = [False] * len(truths)
    for p in sorted((p for p in preds if p.conf >= conf), key=lambda p: -p.conf):
        best, best_iou = -1, iou_thr
        for i, t in enumerate(truths):
            if used[i] or t.cls != p.cls:
                continue
            v = iou(p.box, t.box)
            if v >= best_iou:
                best, best_iou = i, v
        if best >= 0:
            used[best] = True
            counts[p.cls][0] += 1
        else:
            counts[p.cls][1] += 1
    for i, t in enumerate(truths):
        if not used[i]:
            counts[t.cls][2] += 1
    return counts


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(images: list[tuple[list[Pred], list[Truth]]], conf: float, num_classes: int,
          iou_thr: float = 0.5) -> dict:
    """Per-class and averaged P/R/F1 over a set of images at one threshold.

    Macro averages use only classes that have ground truth in the set, so a class missing
    from the split does not count as zero.
    """
    totals = defaultdict(lambda: [0, 0, 0])
    for preds, truths in images:
        for cls, (tp, fp, fn) in match_image(preds, truths, conf, iou_thr).items():
            totals[cls][0] += tp
            totals[cls][1] += fp
            totals[cls][2] += fn
    per_class = {}
    for cls in range(num_classes):
        tp, fp, fn = totals[cls]
        p, r, f = _prf(tp, fp, fn)
        per_class[cls] = {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "support": tp + fn}
    present = [v for v in per_class.values() if v["support"] > 0]
    macro = {k: (sum(v[k] for v in present) / len(present) if present else 0.0) for k in ("precision", "recall", "f1")}
    tp, fp, fn = (sum(totals[c][i] for c in totals) for i in range(3))
    micro = dict(zip(("precision", "recall", "f1"), _prf(tp, fp, fn)))
    return {"conf": conf, "iou": iou_thr, "per_class": per_class, "macro": macro, "micro": micro,
            "classes_with_support": len(present)}


def sweep(images: list[tuple[list[Pred], list[Truth]]], num_classes: int,
          thresholds: list[float] | None = None) -> tuple[float, list[dict]]:
    """Score every threshold; return the one with the best macro F1 (ties go to the higher threshold)."""
    thresholds = thresholds or [round(0.05 * i, 2) for i in range(1, 19)]
    rows = []
    for t in thresholds:
        s = score(images, t, num_classes)
        rows.append({"conf": t, **{f"macro_{k}": v for k, v in s["macro"].items()}})
    best = max(rows, key=lambda r: (round(r["macro_f1"], 6), r["conf"]))
    return best["conf"], rows
