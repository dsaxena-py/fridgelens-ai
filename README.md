# MealLens AI

Training pipeline for the MealLens ingredient detector: a YOLO11 model fine-tuned to find 25 common
kitchen ingredients in one photo (for now, ingredients laid out on a table). It produces an ONNX model that the FastAPI backend
(`fridgelens-backend`) runs on CPU, plus the SQL that registers it in Supabase.

```
Open Images (17 classes) ─┐
Roboflow Universe sets  ──┼─> import ─> build-dataset ─> train ─> tune-threshold ─> evaluate ─> export
Our table photos       ──┘             (scene split)    (YOLO11)     (val)           (test)      (ONNX + SQL)
```

## Where to train: Kaggle

Train on **Kaggle Notebooks**: they're free, give about 30 GPU hours a week on a T4 or P100, run
up to 12 hours per session, and can store private datasets. Open
[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) in Kaggle (*File → Import
Notebook*), follow the settings at the top and *Run All*. Google Colab works the same way as a
backup; its free GPU time is less predictable.

A rough budget: YOLO11s at 640 px on about 7,000 images takes a few minutes per epoch on a T4,
so the 50-epoch baseline fits comfortably in one session.

## The 25 classes

[`configs/classes.yaml`](configs/classes.yaml) is the contract with the backend: the index is
the YOLO class id, and the slug equals `ingredients.slug` in the database. Never reorder it.

| Source | Classes |
| --- | --- |
| Open Images V7 + our photos | apple, banana, orange, lemon (incl. lime), strawberry, tomato, potato, carrot, cucumber, bell_pepper, broccoli, cabbage, mushroom, egg, milk, cheese, bread |
| **Our photos + Roboflow Universe only** | onion, garlic, cauliflower, leafy_greens, eggplant, yogurt, butter, chicken |

The second row is why our own photos matter so much. See
[`docs/data_collection.md`](docs/data_collection.md) for how to shoot, name and label them.

## Commands

```bash
pip install -r requirements.txt   # plus torch, preinstalled on Kaggle/Colab

python -m meallens_ai classes
python -m meallens_ai download-openimages --out data/sources/openimages --max-per-class 400
python -m meallens_ai import-yolo ~/Downloads/roboflow-export --out data/sources/fridge --name fridge --kind custom
python -m meallens_ai build-dataset data/sources/* --out data/meallens
python -m meallens_ai check-dataset data/meallens

python -m meallens_ai train --config configs/train_baseline.yaml --device 0
python -m meallens_ai tune-threshold --weights runs/baseline-yolo11s/weights/best.pt --split val
python -m meallens_ai export --weights runs/baseline-yolo11s/weights/best.pt --name meallens-yolo11s-v1 --conf 0.35
python -m meallens_ai evaluate --weights runs/baseline-yolo11s/weights/best.pt --split test --conf 0.35 \
    --onnx exports/meallens-yolo11s-v1.onnx --out reports
python -m meallens_ai prelabel --weights runs/baseline-yolo11s/weights/best.pt --images new_photos --out prelabelled
```

## How the evaluation stays honest

- **Scene-level splits.** Custom photos are split 70/15/15 by scene, read from the file name
  (`ds-014-03.jpg` belongs to scene `ds-014`), so near-identical shots never sit in both train and
  test. Splits come from a hash of the scene id, so adding photos never moves existing ones.
- **Test set = real fridges only.** Public images go 90/10 to train/val and never to test.
- **One threshold, chosen on val.** `tune-threshold` picks the confidence with the best macro F1
  on val; `evaluate --conf` applies it unchanged to test.
- **Metrics** (proposal §4.1): per-class and macro precision, recall and F1 at that threshold
  (IoU ≥ 0.5 matching), mAP@0.5 and mAP@0.5:0.95 from Ultralytics, a confusion matrix, and
  end-to-end CPU latency of the ONNX model. `evaluate` writes `report_<split>.md`, ready for the
  paper.

## Experiments

| Config | Purpose |
| --- | --- |
| `configs/train_baseline.yaml` | Pretrained YOLO11s, default settings, 50 epochs |
| `configs/train_tuned.yaml` | Cosine LR, 120 epochs, augmentation for indoor lighting, viewpoint and clutter |

To compare model sizes, copy a config and change `model:` to `yolo11n.pt` or `yolo11m.pt`. Change
one thing at a time, so each gain can be attributed.

## Deploying a model

`export` writes three files to `exports/`:

- `<name>.onnx`: the raw YOLO head; NMS runs in [`meallens_ai/onnx_infer.py`](meallens_ai/onnx_infer.py).
- `model_card.json`: classes, input size, thresholds, metrics and the SHA-256 of the model.
- `register_model.sql`: run it in the Supabase SQL editor to fill `model_versions` and
  `detector_classes`. The two activation statements at the end are commented out on purpose.

`onnx_infer.Detector` needs only NumPy, Pillow and ONNX Runtime. The backend uses it as is:

```python
from meallens_ai.onnx_infer import Detector
detector = Detector("exports/meallens-yolo11s-v1.onnx")
detector("fridge.jpg")  # [Detection(class_index=18, label='egg', confidence=0.91, x_min=..., ...)]
```

## Tests

```bash
pytest tests/test_pipeline.py   # unit tests, seconds, no GPU
pytest tests/test_smoke.py      # trains 1 epoch on synthetic images, exports and evaluates (CPU, ~1-2 min)
```
