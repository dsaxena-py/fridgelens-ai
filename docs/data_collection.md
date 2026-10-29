# Collecting and labelling the MealLens fridge dataset

The detector learns most of what it knows from public data, but it is **judged on our own fridge
photos** (the test split is made only of them). Eight classes have no Open Images data at all, so
for those our photos are most of the training data too:

**onion, garlic, cauliflower, leafy_greens, eggplant, yogurt, butter, chicken**

## Targets

| What | Minimum | Good |
| --- | --- | --- |
| Custom images | 600 | 1,000+ |
| Scenes (one arrangement of items) | 150 | 250+ |
| Boxes per class in custom photos | 60 | 120+ |
| Boxes per class for the 8 classes above | 100 | 200+ |

The proposal's 1,500 to 3,000 images is the ideal. With model-assisted labelling (below), 600 to
1,000 is realistic by mid-November and enough for a sound evaluation. About 15% of scenes land
in the test split, so 150 scenes give about 22 test scenes.

## Naming: one rule that matters

Name every photo `<initials>-<scene>-<shot>.jpg`, for example `ds-014-03.jpg` (Dhiransh, scene 14,
shot 3).

The split is decided per scene, so all shots of one arrangement stay in the same split. If shots
of the same shelf ended up in both train and test, the test score would be inflated. Photos that
don't follow the pattern are treated as their own scene, and `build-dataset` warns about them.

Keep a one-line log per scene (fridge, lighting, items) in a shared sheet. It helps the failure
analysis later.

## Shooting

A **scene** is one arrangement of items on one shelf, drawer or counter. For each scene:

1. Take **3 to 5 shots**: straight on, slightly from above, closer, further away, and one with
   the fridge light only (no room light).
2. Then **rearrange**: move items, swap some in or out, change the shelf. That is a new scene
   number.

Across all scenes, aim for:

- **Mixed counts**: some single-item shots, most with 4 to 10 items, some crowded shelves.
- **Clutter**: jars, sauces, takeout boxes and drinks we don't detect. They teach the model what
  is *not* an ingredient.
- **Occlusion**: items half behind others, eggs partly hidden in the carton, greens in bags.
- **Variety**: at least 3 different fridges (dorm, friends, family) and at least 2 phones. Brands
  and packaging vary: different milk cartons, cheese packs, yogurt cups.
- **Hard pairs** side by side: tomato and apple, lemon and orange, onion and potato, garlic and
  onion, cabbage and lettuce.

Phone settings: on iPhone, set Settings → Camera → Formats → **Most Compatible** (JPEG, not HEIC).
No filters, no portrait mode, no digital zoom.

## Labelling rules

Use the class slugs exactly as written (copy them from `python -m meallens_ai classes`):

`apple banana orange lemon strawberry tomato potato onion garlic carrot cucumber bell_pepper broccoli
cauliflower cabbage leafy_greens mushroom eggplant egg milk cheese yogurt butter chicken bread`

**One box per thing a person would count when checking the fridge.** Box tightly around the
visible part.

| Class | Box | Notes |
| --- | --- | --- |
| apple, orange, lemon, tomato, potato, onion, carrot, cucumber, bell_pepper, eggplant | each item | `lemon` covers limes too. A bag of loose onions or potatoes: box the bag. |
| banana | each bunch | A single loose banana gets its own box. |
| strawberry, mushroom | the punnet / container, or each loose item | |
| garlic | each bulb | Loose cloves together: one box. |
| broccoli, cauliflower, cabbage | each head or bag of florets | |
| leafy_greens | each head, bag or clamshell | Lettuce, spinach, kale, salad mix. |
| egg | each visible egg; a **closed** carton gets one box | |
| milk, yogurt, butter, cheese | each package (carton, jug, cup, tub, stick, block, pack) | |
| chicken | each raw pack or tray | Not cooked dishes. |
| bread | each loaf or bag | |

Also:
- **Label partly hidden items** if about a quarter or more is visible and you can tell what it is.
- **Don't label** pictures of food on packaging (the tomato on a ketchup bottle), cooked dishes,
  or anything you can't identify.
- **Unsure?** Leave it unlabelled and note the file name. A wrong label hurts more than a missing one.

## Roboflow workflow (free plan)

1. Create a project: type **Object Detection**, name `meallens-fridge`.
2. Add the 25 classes above with exactly those names.
3. Upload photos and label them following the rules above.
4. **Generate a version** with these settings:
   - Preprocessing: **Auto-Orient on**; Resize **off** (or "Fit within 1280" to keep files small).
   - Augmentation: **none**. Training adds its own; augmenting here would leak into val and test.
   - Split: anything. The pipeline re-splits by scene.
5. Export **YOLOv8** format (the YOLO11 option is the same format) and download the zip.
6. Import it:
   ```bash
   python -m meallens_ai import-yolo path/to/export --out data/sources/fridge --name fridge --kind custom
   ```
   On Kaggle, upload the zip as a private Dataset instead; the notebook imports it.

## Model-assisted labelling

Labelling from scratch is the slowest part. Once any model exists (even the Open Images-only
baseline), let it draft the boxes:

```bash
python -m meallens_ai prelabel --weights runs/baseline-yolo11s/weights/best.pt \
    --images new_photos/ --out prelabelled/ --conf 0.4
```

Upload `prelabelled/` to Roboflow as a YOLO dataset (it includes `data.yaml`), then **check and
fix every image**: delete wrong boxes, adjust loose ones, and add what was missed. Retrain every
200 to 300 new images, and the drafts keep getting better.

## Public data for the missing classes

For the 8 classes Open Images lacks, add public datasets from
[Roboflow Universe](https://universe.roboflow.com). Search for "onion garlic", "fridge
ingredients", "eggplant", "yogurt" and similar. Pick datasets with real photos (not renders), and
check the license: CC BY 4.0 is fine, but cite it in the paper. Export them as YOLOv8 and import
with `--kind public`:

```bash
python -m meallens_ai import-yolo universe_export/ --out data/sources/veg-universe --name veg_universe \
    --kind public --map veg_universe_map.yaml
```

Classes with other names (e.g. `Aubergine`, `Spring Onion`) are matched by the aliases in
`configs/classes.yaml`. The import prints any class it could not map. Map those in a small YAML
file, using `null` to drop a class:

```yaml
Brinjal: eggplant
Spring Onion: null   # green onion is not one of our classes
```
