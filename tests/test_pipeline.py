"""Unit tests for the pure-Python parts of the pipeline (no GPU, no downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from meallens_ai import classes
from meallens_ai.dataset import assign_split, build, check, scene_of
from meallens_ai.export import registration_sql
from meallens_ai.importer import build_mapping, import_yolo
from meallens_ai.metrics import Pred, Truth, iou, match_image, score, sweep
from meallens_ai.onnx_infer import decode, letterbox, nms
from meallens_ai.yolo_io import Box, from_top_left, label_path_for, read_labels, write_labels, write_source_info

# Must equal the `class` order in fridgelens-backend/seed_data/ingredients.yaml.
BACKEND_SLUGS = ["apple", "banana", "orange", "lemon", "strawberry", "tomato", "potato", "onion", "garlic",
                 "carrot", "cucumber", "bell_pepper", "broccoli", "cauliflower", "cabbage", "leafy_greens",
                 "mushroom", "eggplant", "egg", "milk", "cheese", "yogurt", "butter", "chicken", "bread"]


def make_image(path: Path, size=(64, 48), color=(200, 30, 30)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


# ---------- classes ----------

def test_classes_match_backend_order():
    assert classes.names() == BACKEND_SLUGS


@pytest.mark.parametrize("label, expected", [
    ("Bell_Pepper", 11), ("capsicum", 11), ("Egg (Food)", 18), ("lime", 3), ("Tomatoes", 5),
    ("baby-carrot", 9), ("lettuce", 15), ("chicken breast", 23), ("jar", None), ("Chicken", 23),
])
def test_resolve(label, expected):
    assert classes.resolve(label) == expected


def test_openimages_map_excludes_live_chicken():
    oi = classes.openimages_map()
    assert "Chicken" not in oi  # Open Images "Chicken" is the bird
    assert oi["Egg (Food)"] == 18 and len(set(oi.values())) == 17


# ---------- yolo_io ----------

def test_from_top_left_converts_and_clips():
    b = from_top_left(3, 0.9, 0.1, 0.2, 0.2)  # runs past the right edge
    assert b.cls == 3
    assert b.to_xyxy() == pytest.approx((0.9, 0.1, 1.0, 0.3))


def test_labels_roundtrip(tmp_path):
    p = tmp_path / "labels" / "a.txt"
    boxes = [Box(1, 0.5, 0.5, 0.2, 0.4), Box(24, 0.1, 0.2, 0.05, 0.05)]
    write_labels(p, boxes)
    assert read_labels(p) == [Box(1, 0.5, 0.5, 0.2, 0.4), Box(24, 0.1, 0.2, 0.05, 0.05)]
    assert label_path_for(tmp_path / "train" / "images" / "a.jpg") == tmp_path / "train" / "labels" / "a.txt"


# ---------- importer ----------

def test_build_mapping_with_extra():
    mapping, unmapped = build_mapping(["Capsicum", "tomato", "jar", "Red-Cap"], {"red cap": "bell_pepper", "jar": None})
    assert mapping == {0: 11, 1: 5, 3: 11}
    assert unmapped == []


def _roboflow_export(root: Path, names: list[str]) -> Path:
    (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}))
    make_image(root / "train" / "images" / "ds-001-01_jpg.rf.abc.jpg")
    write_labels(root / "train" / "labels" / "ds-001-01_jpg.rf.abc.txt", [Box(0, .5, .5, .2, .2), Box(1, .3, .3, .1, .1)])
    make_image(root / "valid" / "images" / "ds-002-01_jpg.rf.def.jpg")
    write_labels(root / "valid" / "labels" / "ds-002-01_jpg.rf.def.txt", [Box(1, .5, .5, .2, .2)])  # 'jar' only
    return root


def test_import_custom_keeps_empty_images(tmp_path):
    src = _roboflow_export(tmp_path / "export", ["Tomato", "jar"])
    info = import_yolo(src, tmp_path / "custom", "fridge", "custom")
    assert info["images"] == 2 and info["boxes"] == {"tomato": 1} and info["unmapped_classes"] == ["jar"]
    assert read_labels(tmp_path / "custom" / "labels" / "ds-002-01_jpg.rf.def.txt") == []


def test_import_public_drops_empty_images_and_prefixes(tmp_path):
    src = _roboflow_export(tmp_path / "export", ["Tomato", "jar"])
    info = import_yolo(src, tmp_path / "pub", "veg", "public")
    assert info["images"] == 1 and info["skipped_empty_images"] == 1
    assert (tmp_path / "pub" / "images" / "veg_ds-001-01_jpg.rf.abc.jpg").exists()


# ---------- dataset ----------

def test_scene_of_survives_roboflow_renaming():
    assert scene_of("ds-014-03") == "ds-014"
    assert scene_of("DS-014-07_jpg.rf.9f8e7d") == "ds-014"
    assert scene_of("IMG_1234") is None


def test_splits_are_deterministic_and_public_never_test():
    assert assign_split("custom", "fridge/ds-001", 0) == assign_split("custom", "fridge/ds-001", 0)
    assert {assign_split("public", f"oi/{i}", 0) for i in range(500)} == {"train", "val"}
    custom = [assign_split("custom", f"fridge/ds-{i:03d}", 0) for i in range(2000)]
    assert 0.65 < custom.count("train") / 2000 < 0.75 and custom.count("test") > 200


def test_build_keeps_scenes_together(tmp_path):
    custom = tmp_path / "custom"
    for scene in range(30):
        for shot in range(3):
            stem = f"ab-{scene:03d}-{shot:02d}"
            make_image(custom / "images" / f"{stem}.jpg")
            write_labels(custom / "labels" / f"{stem}.txt", [Box(scene % 25, .5, .5, .3, .3)])
    write_source_info(custom, name="fridge", kind="custom")
    public = tmp_path / "public"
    for i in range(20):
        make_image(public / "images" / f"oi_{i}.jpg")
        write_labels(public / "labels" / f"oi_{i}.txt", [Box(0, .5, .5, .3, .3)])
    write_source_info(public, name="openimages", kind="public")

    out = tmp_path / "ds"
    stats = build([custom, public], out, seed=0, link="copy")
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"][11] == "bell_pepper" and data["test"] == "images/test"
    rows = (out / "manifest.csv").read_text().splitlines()[1:]
    split_of_scene: dict[str, set] = {}
    for row in rows:
        _, source, kind, group, split = row.split(",")
        split_of_scene.setdefault(group, set()).add(split)
        assert not (kind == "public" and split == "test")
    assert all(len(s) == 1 for s in split_of_scene.values())
    assert sum(v["images"] for v in stats["splits"].values()) == 110


def test_check_reports_bad_boxes(tmp_path):
    make_image(tmp_path / "images" / "train" / "x.jpg")
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "x.txt").write_text("99 0.5 0.5 0.1 0.1\n2 0.95 0.5 0.3 0.1\n")
    errors = check(tmp_path)["errors"]
    assert any("out of range" in e for e in errors) and any("outside the image" in e for e in errors)


# ---------- metrics ----------

def test_iou():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    assert iou((0, 0, 2, 1), (1, 0, 3, 1)) == pytest.approx(1 / 3)


def test_match_counts_duplicates_and_wrong_class_as_fp():
    truths = [Truth(0, (0, 0, .5, .5)), Truth(1, (.5, .5, 1, 1))]
    preds = [Pred(0, .9, (0, 0, .5, .5)), Pred(0, .8, (0, 0, .5, .5)),  # duplicate -> fp
             Pred(2, .9, (.5, .5, 1, 1)),                                # wrong class -> fp, truth 1 -> fn
             Pred(0, .1, (0, 0, .5, .5))]                                # below threshold, ignored
    counts = match_image(preds, truths, conf=0.5)
    assert counts[0] == [1, 1, 0] and counts[1] == [0, 0, 1] and counts[2] == [0, 1, 0]


def test_score_macro_ignores_classes_without_support():
    images = [([Pred(0, .9, (0, 0, .5, .5))], [Truth(0, (0, 0, .5, .5))])]
    s = score(images, 0.5, 25)
    assert s["macro"]["f1"] == 1.0 and s["classes_with_support"] == 1


def test_sweep_picks_threshold_that_drops_low_confidence_false_positives():
    images = [([Pred(0, .9, (0, 0, .5, .5)), Pred(0, .3, (.6, .6, .9, .9))], [Truth(0, (0, 0, .5, .5))])]
    best, rows = sweep(images, 25, [0.2, 0.5])
    assert best == 0.5 and rows[0]["macro_f1"] < rows[1]["macro_f1"]


# ---------- onnx_infer ----------

def test_letterbox_pads_to_square():
    tensor, scale, pad = letterbox(Image.new("RGB", (200, 100)), 64)
    assert tensor.shape == (1, 3, 64, 64) and scale == pytest.approx(0.32) and pad == (0, 16)


def test_nms_and_decode():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    assert nms(boxes, np.array([.9, .8, .7]), 0.5) == [0, 2]
    nc, n = 3, 4
    out = np.zeros((1, 4 + nc, n), dtype=np.float32)
    out[0, :4, 0] = [10, 10, 4, 4]; out[0, 4 + 1, 0] = .9   # class 1
    out[0, :4, 1] = [10, 10, 4, 4]; out[0, 4 + 1, 1] = .6   # duplicate of 0, suppressed
    out[0, :4, 2] = [10, 10, 4, 4]; out[0, 4 + 2, 2] = .7   # same place, other class, kept
    out[0, :4, 3] = [50, 50, 4, 4]; out[0, 4 + 0, 3] = .1   # below conf
    dets = decode(out, conf=0.25, iou_thr=0.5)
    assert sorted(dets[:, 5].tolist()) == [1.0, 2.0]
    assert dets[0, :4].tolist() == [8, 8, 12, 12]


# ---------- export ----------

def test_registration_sql():
    card = {"name": "meallens-yolo11s-v1", "architecture": "yolo11s", "weights_uri": "x.onnx", "input_size": 640,
            "conf_threshold": 0.35, "iou_threshold": 0.5, "metrics": {"map50": 0.7, "note": "it's"},
            "classes": [{"index": c.index, "slug": c.slug} for c in classes.load_classes()]}
    sql = registration_sql(card)
    assert "insert into model_versions" in sql and "(24, 'bread')" in sql
    assert "it''s" in sql and "-- update model_versions set is_active = true" in sql
