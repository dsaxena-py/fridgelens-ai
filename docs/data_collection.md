# Collecting and labelling the MealLens dataset

MealLens scans ingredients **laid out on an open table**: a dining table, desk or kitchen counter,
seen from above or at an angle. That is the domain the demo runs in, so it is the domain we
collect, label and test on. (Fridge scenes may come later; they would use exactly the same rules.)

## What each data source is for

| Source | Scenes | Used for | Never used for |
| --- | --- | --- | --- |
| **Our table photos** (this guide) | open tables, our lighting, our phones | training, threshold tuning, **the only test set** | |
| Open Images V7 (17 classes) | mixed: markets, close-ups, dishes | extra training data: what a tomato looks like | testing |
| Roboflow Universe (mainly the 8 classes below) | only sets that show items on a table or counter | extra training data | testing |

Every number we report is measured on our own table photos, so the test score says how the demo
will do. Public data only helps the model learn what each ingredient looks like. Only about 3 to
10% of Open Images photos of our classes show a table, which is why our photos matter.

Eight classes have no Open Images boxes, so our photos and Roboflow sets are all the model gets
for them. **Shoot these more than anything else:**

**onion, garlic, cauliflower, leafy_greens, eggplant, yogurt, butter, chicken**

## Targets

| What | Minimum | Good |
| --- | --- | --- |
| Photos | 300 | 600+ |
| Scenes (one arrangement of items) | 75 | 150+ |
| Boxes per class | 40 | 80+ |
| Boxes per class for the 8 classes above | 80 | 150+ |

A scene takes about a minute to shoot (4 shots), so the minimum is roughly **75 minutes of
shooting in total**, or 25 scenes each across three people. About 15% of scenes go to the test
set, so 75 scenes give about 11 test scenes (about 45 photos), and every class should appear in
at least 3 of them. Labelling is the bigger job: about a minute per photo, faster once the first
model drafts the boxes (see *Model-assisted labelling*).

## Shopping list

All 25 classes, so a few of each are on hand while shooting:

| Group | Items |
| --- | --- |
| Fruit | apple, banana, orange, lemon or lime, strawberries |
| Vegetables | tomato, potato, onion, garlic, carrot, cucumber, bell pepper, broccoli, cauliflower, cabbage, leafy greens (lettuce, spinach or salad bag), mushrooms, eggplant |
| Dairy and eggs | eggs (in a carton), milk, cheese, yogurt, butter |
| Other | raw chicken (in its pack; keep it sealed), bread |

Buy more than one brand of the packaged items if you can (two kinds of milk, cheese and yogurt).
Packaging changes a lot between brands, and the model needs to see that.

## Naming: one rule that matters

Name every photo `<initials>-<scene>-<shot>.jpg`, for example `ds-014-03.jpg` (Dhiransh, scene 14,
shot 3). Each person numbers their own scenes from 001.

The split is decided per scene, so all shots of one arrangement stay in the same split. If shots
of the same arrangement ended up in both train and test, the test score would be inflated. Photos
that don't follow the pattern are treated as their own scene, and `build-dataset` warns about them.

Keep a one-line log per scene in a shared sheet, with these columns: scene, surface, lighting,
items. For example: `ds-014 | wood table | window daylight | tomato, onion, garlic, eggs, butter`.
It drives the failure analysis later (which surfaces or lights the model struggles with).

## Shooting

A **scene** is one arrangement of 4 to 10 items on one table. For each scene:

1. Take **4 shots**:
   - **top-down**: phone held flat above the table, whole arrangement in frame. Most scans look
     like this, so never skip it;
   - **angled**: about 45 degrees, standing next to the table;
   - **low**: near table height, so items overlap and hide each other;
   - **close or far**: alternate between scenes, either half the items filling the frame or the
     whole table with space around it.
2. Then **rearrange**: move items around, swap at least two in or out, and every few scenes change
   the surface or the light. That is a new scene number.

Across all scenes, aim for:

- **Surfaces**: at least 3 (wood, white or plain, a patterned tablecloth or placemat, a kitchen
  counter). A detector trained on one table learns the table.
- **Lighting**: window daylight, overhead room light, a dim evening lamp, and some shots with
  strong shadows across the items. No flash.
- **Clutter**: plates, mugs, cutlery, jars, sauces, phones and drinks we don't detect. They teach
  the model what is *not* an ingredient. Put 1 to 3 in most scenes.
- **Occlusion**: items partly behind or on top of others, eggs in an open carton, greens in a bag.
- **Natural states**: in packaging, loose, in a bowl, lying down or standing, as things are when
  unpacked from a shopping bag.
- **Counts**: mostly 4 to 10 items, a few single-item scenes, and a few crowded ones (12 or more).
- **Hard pairs side by side**: tomato and apple, lemon and orange, onion and potato, garlic and
  onion, cabbage and lettuce. These are the mistakes the model will make, so it needs to see
  them together.
- **Phones**: at least 2 different phones across the team.

Phone settings: on iPhone, set Settings → Camera → Formats → **Most Compatible** (JPEG, not HEIC).
No filters, no portrait mode, no digital zoom, no live photos.

Handing photos over: upload each day's photos to the shared Roboflow project (below). If labelling
has not started yet, zip them by person and day (`ds-2026-11-04.zip`) and share the zip.

## Labelling rules

Use the class slugs exactly as written (copy them from `python -m meallens_ai classes`):

`apple banana orange lemon strawberry tomato potato onion garlic carrot cucumber bell_pepper broccoli
cauliflower cabbage leafy_greens mushroom eggplant egg milk cheese yogurt butter chicken bread`

**One box per thing a person would count when checking what they have.** Box tightly around the
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

1. Create a project: type **Object Detection**, name `meallens-table`, and invite the team.
2. Add the 25 classes above with exactly those names.
3. Upload photos and label them following the rules above.
4. **Generate a version** with these settings:
   - Preprocessing: **Auto-Orient on**; Resize **off** (or "Fit within 1280" to keep files small).
   - Augmentation: **none**. Training adds its own; augmenting here would leak into val and test.
   - Split: anything. The pipeline re-splits by scene.
5. Export **YOLOv8** format (the YOLO11 option is the same format) and download the zip.
6. Import it:
   ```bash
   python -m meallens_ai import-yolo path/to/export --out data/sources/table --name table --kind custom
   ```
   On Kaggle, upload the zip as a private Dataset instead; the notebook imports it.

## Model-assisted labelling

Labelling from scratch is the slowest part. Once any model exists (even the baseline trained on
public data only), let it draft the boxes:

```bash
python -m meallens_ai prelabel --weights runs/baseline-yolo11s/weights/best.pt \
    --images new_photos/ --out prelabelled/ --conf 0.4
```

Upload `prelabelled/` to Roboflow as a YOLO dataset (it includes `data.yaml`), then **check and
fix every image**: delete wrong boxes, adjust loose ones, and add what was missed. Retrain every
100 to 200 new images, and the drafts keep getting better.

## Public data from Roboflow Universe

For the 8 classes Open Images lacks, add public datasets from
[Roboflow Universe](https://universe.roboflow.com). **Only use a dataset if most of its photos
show items on a table or counter.** Open a page of its images before importing and skip it if
they are mostly:

- supermarket shelves, market stalls or crates,
- plants in a field or garden,
- single items on a plain white studio background,
- renders, drawings or screenshots,
- cooked or plated dishes.

Also check the license (CC BY 4.0 is fine, but cite it in the paper) and that boxes are drawn per
item as in our rules. Export as YOLOv8 and import with `--kind public`:

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
