# Project handoff: drone seal detection and thermal classification

## What this project does

The project has three related workflows:

1. Detect seals in high-resolution drone RGB images and write one annotation JSON file per image.
2. Pair each RGB image with a DJI thermal image, align the coordinate systems, measure thermal-image intensities inside each seal annotation, and assign a first-pass `alive`, `dead`, or `uncertain` prediction.
3. Prepare 640 x 640 YOLO training tiles from full-size RGB images and YOLO labels.

The preferred learned detector at handoff is `src/WAID_detect_sahi.py` with `model_weights/best_8.pt`. `src/cli.py detect` is a separate, classical computer-vision proposal generator. Both produce annotation JSON that `src/cli.py run` can consume.

The classification stage does **not** calculate calibrated physical temperature. It reads thermal pixels as grayscale intensity, compares each seal region with a nearby background ring, and applies fixed thresholds. Treat its predictions as review suggestions, not validated biological conclusions.

## End-to-end data flow

```text
RGB images
  |-- WAID_detect_sahi.py (preferred YOLO + SAHI detector)
  |        or cli.py detect (non-ML proposal generator)
  v
per-image annotation JSON + visual detection review
  |
  |     thermal images + optional RGB-to-thermal calibration JSON
  +------------------------------+
                                 v
                            cli.py run
                                 |
                                 +-- pairing QA images and report
                                 +-- per-seal thermal features
                                 +-- alive/dead/uncertain review predictions
                                 +-- label summaries/readiness report

Full RGB images + YOLO .txt labels
  -> tile_drone_dataset.py
  -> 640 x 640 training images and clipped YOLO labels

RGB + thermal images + calibration
  -> export_aligned_rgb_thermal.py
  -> paired images on the thermal pixel grid + manifests
```

## Repository orientation

| File | Role | Status |
|---|---|---|
| `src/cli.py` | Pairing, QA, classical detection, feature extraction, and heuristic classification | Main pipeline |
| `src/WAID_detect_sahi.py` | Local YOLO inference using SAHI slices and the WAID-derived model | Preferred detector |
| `src/yolo_detect_sahi.py` | Local SAHI inference using an older KSpicY-derived model | Legacy/experimental |
| `src/yolo_detect_server.py` | Hosted Roboflow workflow inference | Legacy/experimental; credential issue |
| `src/export_aligned_rgb_thermal.py` | Export aligned RGB/thermal pairs on a common pixel grid | Utility |
| `src/tile_drone_dataset.py` | Turn full images and YOLO labels into overlapping training tiles | Training-data utility; known bugs |

Model weights are under `model_weights/`. The preferred checkpoint is `best_8.pt`. Existing output directories are experiment artifacts and should not be assumed to be reproducible or canonical.

## Environment setup

This checkout contains `detector_env/` and `venv/`, but local virtual environments are not portable. Recreate an environment on the new machine and add a dependency lock file as early maintenance work.

The existing `detector_env` was observed with Python 3.12.3 and these relevant versions:

- OpenCV 4.10.0
- NumPy 2.3.5
- Pillow 12.2.0
- SAHI 0.11.36
- Ultralytics 8.4.60
- Roboflow `inference-sdk` 1.3.3

A minimal installation is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pillow opencv-python sahi ultralytics inference-sdk
```

PyTorch is installed as an Ultralytics dependency. Install the appropriate CUDA-enabled PyTorch build separately if GPU inference is required. Both SAHI scripts currently select `device="cpu"`; change that setting to `cuda` only after confirming the environment and GPU are compatible.

Run commands from the repository root. Model paths such as `model_weights/best_8.pt` are relative to the current working directory, and `export_aligned_rgb_thermal.py` imports `cli` from the `src` script directory.

## Expected image organization and naming

`cli.py` and `export_aligned_rgb_thermal.py` accept either:

```text
data/
  rgb/
    DJI_0249.jpg
  thermal/
    DJI_0248_R.JPG
```

or a single mixed directory containing both image types.

Supported extensions are `.jpg`, `.jpeg`, `.png`, `.tif`, and `.tiff`. An image is thermal when it is under a directory named `thermal` or its stem ends in `_R` (case-insensitive). Everything else is treated as RGB.

If either `<input>/rgb` or `<input>/thermal` exists, scanning is restricted to those subdirectories; images elsewhere under the input root are ignored. Generated directories named `annotations`, `outputs`, and `classifier_outputs` are also ignored during recursive scanning.

DJI pairing is based on the integer in `DJI_<number>`:

- First try `thermal number = RGB number + --thermal-offset`.
- The default offset is `-1`, so `DJI_0249.jpg` pairs with `DJI_0248_R.JPG`.
- If the offset match is absent, the same DJI number is tried.
- EXIF timestamps and GPS data only set a confidence/review flag; they do not choose the pair.
- Files without a `DJI_<number>` token remain unmatched.

Review `pairing_report.csv`, especially `missing_thermal`, `review_timestamp`, `review_gps`, and `filename_only` rows, before trusting downstream output.

## Annotation and calibration contracts

### Annotation JSON

All three detector scripts write this shape:

```json
{
  "image": "DJI_0249.jpg",
  "seals": [
    {
      "id": "seal_001",
      "label": "unknown",
      "confidence": 0.87,
      "bbox": [120, 240, 45, 28],
      "polygon": [[120, 240], [165, 240], [165, 268], [120, 268]]
    }
  ]
}
```

`bbox` is `[x, y, width, height]` in RGB pixels. `polygon` is an array of `[x, y]` RGB pixel points. An annotation needs at least one of these. The accepted known labels are `dead`, `alive`, and `unknown`; missing labels normalize to `unknown`. Detector confidence and other detector-specific fields are optional.

`cli.py run` loads only `*.json` files directly inside the annotations directory; it does not recurse. It matches files using the normalized `image` field or its stem.

### Calibration JSON

Calibration maps RGB coordinates to raw thermal coordinates with an OpenCV homography and needs at least four corresponding points:

```json
{
  "rgb_image": "DJI_0249.jpg",
  "thermal_image": "DJI_0248_R.JPG",
  "points": [
    {"rgb": [812, 436], "thermal": [143, 89]},
    {"rgb": [2194, 508], "thermal": [381, 101]},
    {"rgb": [2320, 1670], "thermal": [405, 301]},
    {"rgb": [744, 1588], "thermal": [129, 287]}
  ]
}
```

A calibration named for the RGB image/stem takes precedence; `calibration/default.json` is the fallback. See `src/CALIBRATION.md` for collection advice and the alternate `rgb_points`/`thermal_points` representation.

Without calibration, `cli.py run` resizes the thermal frame to RGB dimensions before measuring annotations. This is only an approximation and does not provide physical alignment.

## Primary workflow: `src/cli.py`

### Initialize directories

```bash
python src/cli.py init
```

This creates `data/rgb`, `data/thermal`, `annotations`, and `outputs/qa`. It does not copy data or validate dependencies.

### Option A: classical RGB candidate generation

```bash
python src/cli.py detect \
  --input IMAGES_PATH \
  --output-annotations annotations \
  --review-output classifier_outputs/detection_review \
  --sensitivity high_recall
```

This is not a YOLO command. It combines adaptive thresholding, morphology, MSER, light/dark blobs, edge/residual proposals, group splitting, overlap suppression, and heuristic rock filtering. Its output requires human review.

Important options:

- `--sensitivity {high_recall,balanced,strict}` trades false positives for recall; the default is `high_recall`.
- `--max-candidates 150` sets a per-image cap applied before the final dense-rock filter.
- `--target-rgb DJI_0249` may be repeated to select multiple filenames or stems.
- `--limit N` processes only the first N sorted RGB records.
- `--exclusions DIR` loads polygons whose interior candidate centers should be removed.
- `--manual-overrides DIR` rejects generated IDs and/or adds annotations.

An exclusion file is named `<RGB stem>.json` (or the full filename plus `.json`):

```json
{
  "exclude_polygons": [
    [[10, 10], [300, 10], [300, 200], [10, 200]]
  ]
}
```

A manual override file uses this shape:

```json
{
  "reject_ids": ["auto_seal_004", "007"],
  "add": [
    {"id": "manual_seal_001", "label": "unknown", "bbox": [120, 240, 45, 28]},
    {"label": "alive", "polygon": [[400, 200], [440, 205], [438, 230], [398, 228]]}
  ]
}
```

The command writes:

```text
annotations/<RGB stem>.json
classifier_outputs/detection_review/<RGB stem>_detected_candidates.jpg
classifier_outputs/detection_review/tiles/*.jpg
classifier_outputs/detection_review/detection_candidates.csv
```

### Pair, measure, and classify

For the preferred WAID annotations:

```bash
python src/cli.py run \
  --input IMAGES_PATH \
  --annotations ANNOTATIONS_PATH \
  --calibration calibration \
  --output OUTPUT_PATH
```

Useful targeted smoke test:

```bash
python src/cli.py run \
  --input IMAGES_PATH \
  --annotations ANNOTATIONS_PATH \
  --calibration calibration \
  --output /tmp/dead-seal-smoke \
  --target-rgb DJI_0249 \
  --limit 1
```
IMAGES_PATH can be a dataset with RGB and thermal images   

Key options:

- `--thermal-offset -1` controls the DJI number relationship described above.
- `--threshold-percentile 80` makes pixels at or above this per-image thermal percentile count as “hot.”
- `--min-label-count 5` is the number of verified `dead` and `alive` annotations required for the readiness text to say `ready_for_classifier_review`. It does not train a classifier.
- `--target-rgb` is a repeatable filename/stem filter.

For each annotated seal, the pipeline measures mean, median, minimum, and maximum intensity; fraction of thresholded hot pixels; nearby background mean; and seal-minus-background contrast. With calibration, RGB annotations are transformed to raw thermal coordinates. Without it, thermal is resized to RGB dimensions.

The current fixed prediction rules are:

| Result | Rule |
|---|---|
| `alive` | contrast >= 8 and hot fraction >= 0.45 |
| `dead` | contrast <= -2 and hot fraction <= 0.35 |
| `uncertain` | anything else, including no background pixels |

Higher-confidence sub-thresholds are embedded in `predict_dead_alive()`. These values are uncalibrated pixel units and should be revalidated for every camera/export pipeline.

Outputs under `--output`:

| Output | Purpose |
|---|---|
| `pairing_report.csv` | Pair selection, EXIF QA, dimensions, calibration, and QA paths |
| `seal_thermal_features.csv` | Complete per-seal measurements and predictions |
| `classification_report.csv` | Compact prediction report and agreement with known labels |
| `dead_alive_summary.csv` | Feature averages for verified dead/alive labels |
| `classification_readiness.txt` | Whether minimum verified-label counts were reached |
| `qa/*_overlay.jpg` | RGB/thermal blended alignment review |
| `qa/*_side_by_side.jpg` | RGB and thermal heatmap comparison |
| `thermal_masks/*_mask.jpg` | Percentile-thresholded thermal image |
| `annotation_review/` | Annotated review imagery when annotations exist |

## Preferred detector: `src/WAID_detect_sahi.py`

This script loads `model_weights/best_8.pt` as a YOLOv8 model through SAHI, slices every source image into 640 x 640 windows with 30% overlap, and uses a confidence threshold of 0.4. It converts every prediction to a rectangular polygon and labels it `unknown` for later review/classification.

Before running, edit these module-level settings:

```python
model_path = "model_weights/best_8.pt"
images_dir = Path("/path/to/rgb/images")
output_dir = Path("/path/to/detection_outputs")
# ...
device="cpu"  # or "cuda"
```
Assumes detection_outputs is a folder with structure
```text   
detection_outputs/
  annotated/
    DJI__****_annotated.jpg
  annotations/
    DJI_****.json
```
Then run from the repository root:

```bash
python src/WAID_detect_sahi.py
```

It processes direct children of `images_dir` with `.jpg`, `.jpeg`, or `.png` extensions, excluding stems ending in `_R`. It does not recurse and has no command-line flags. Outputs are:

```text
<output_dir>/annotated/<stem>_annotated.jpg
<output_dir>/annotations/<stem>.json
```

Important behavior:

- Model construction, output-directory creation, and the image loop occur at module import time. Do not import this file as a library.
- Existing output files with the same names are overwritten.
- Inference is CPU-only until the source setting is changed.
- `height` and `width` are passed into `run_workflow()` but the current slice dimensions remain fixed at 640.

## Legacy local detector: `src/yolo_detect_sahi.py`

This is an older SAHI experiment around the KSpicY seal model. It uses a 0.2 confidence threshold and slices at half of the full image height and width with 30% overlap.

It is not ready to run unchanged:

- Input and output paths are hardcoded to the original developer's home directory.
- It accesses `result.object_prediction_list[0]` for debugging and crashes on an image with zero predictions.
- Like the WAID script, it runs inference at import time and has no CLI.

If this experiment must be revived, fix those issues first. Its output annotation contract is otherwise compatible with `cli.py run`.

## Hosted detector: `src/yolo_detect_server.py`

This script sends each RGB image to a Roboflow hosted workflow, extracts segmentation polygons or bounding boxes from the response, writes compatible JSON and annotated images, then waits two seconds before the next request.

Current remote identifiers are hardcoded:

- API URL: Roboflow serverless endpoint
- Workspace: `sherry-lei`
- Workflow: `general-segmentation-api-4`
- Class parameter: `seal`

Critical security action: the source contains a plaintext Roboflow API key. Assume it is compromised, revoke/rotate it, remove it from the file and Git history as appropriate, and load its replacement from an environment variable or secret manager. Do not copy the current value into documentation or new code.

Other operational caveats:

- Input/output paths are hardcoded.
- Hosted calls may incur usage charges or hit rate/credit limits.
- `use_cache=True` may return cached workflow results.
- The script processes only direct `.jpg`, `.jpeg`, and `.png` children and excludes `_R` stems.
- There is no application-level retry or per-image failure isolation; one exception stops the loop.

## Alignment exporter: `src/export_aligned_rgb_thermal.py`

Use this when downstream work needs the RGB and thermal images already expressed on the same thermal-coordinate grid:

```bash
python src/export_aligned_rgb_thermal.py \
  --input IMAGES_PATH \
  --calibration calibration \
  --output aligned_rgb_thermal
```

For a calibrated pair, the RGB image is warped into the raw thermal frame. By default both outputs are cropped to the bounding rectangle of valid warped RGB pixels. `--no-crop` retains the complete thermal frame. Missing-calibration pairs are skipped unless `--allow-uncalibrated-resize` is supplied, in which case RGB is merely resized to thermal dimensions.

Outputs:

```text
aligned_rgb_thermal/
  rgb/<RGB original filename>
  thermal/<RGB original filename>
  manifest.csv
  manifest.json
  skipped.csv             # only when at least one pair is skipped
```

The manifests record original paths, pairing/calibration information, output dimensions, and crop coordinates.

Known issue: despite the parser description saying “aligned PNGs,” the implementation reuses the RGB filename and extension for both outputs. For an RGB JPEG, the aligned thermal image is therefore also written as JPEG, which can lose thermal precision. Change the exporter to an explicitly lossless format (and decide how to preserve thermal bit depth) before using these files for quantitative analysis.

## Training-data tiler: `src/tile_drone_dataset.py`

Input labels must be standard YOLO detection rows:

```text
class_id x_center_normalized y_center_normalized width_normalized height_normalized
```

Example:

```bash
python src/tile_drone_dataset.py \
  --images training/full_images \
  --labels training/full_labels \
  --out training/tiled_640 \
  --tile_size 640 \
  --overlap 0.2 \
  --min-visible 0.35 \
  --include-empty
```

The script uses a stride of `round(tile_size * (1 - overlap))`, forces a final tile against each image edge, clips boxes to tile boundaries, drops a clipped box when less than `min-visible` of its original area remains, and writes:

```text
<out>/images/<source stem>_x####_y####.jpg
<out>/labels/<source stem>_x####_y####.txt
```

Known correctness issues to fix before building another dataset:

- `tile_labels` contains `None` entries for boxes that do not intersect a tile. Because the list itself is nonempty, tiles can be written as empty negatives even when `--include-empty` was not supplied.
- The `written` and `empty` counters are indented outside the inner tile loop, so their printed totals are incorrect.
- A malformed label row calls undefined `printf`, causing `NameError` instead of a useful validation error.
- A missing `<stem>.txt` label file raises `FileNotFoundError` and stops the whole run.
- There is no validation that `overlap` is in `[0, 1)` or that `tile_size`/`min-visible` are sensible.
- All image tiles are saved as JPEG even if the source is lossless.

Inspect generated image/label pairs and class counts before training. Do not infer dataset integrity from the current summary counters.

## Recommended takeover sequence

1. Revoke the Roboflow key in `yolo_detect_server.py`, remove it from source, and migrate secrets to environment variables.
2. Create a clean virtual environment and commit a dependency lock/requirements file. Do not rely on `detector_env/` or `venv/` being portable.
3. Select a tiny known RGB/thermal pair and verify pairing plus calibration visually with `cli.py run --limit 1`.
4. Run `WAID_detect_sahi.py` on one RGB frame, inspect the annotated image/JSON, then feed that annotations directory to `cli.py run`.
5. Manually verify annotations and alive/dead labels. Quantify detector precision/recall and classifier performance on held-out, verified data before operational use.
6. Convert the hardcoded detector settings into CLI arguments and place all script execution behind `if __name__ == "__main__"`.
7. Fix and test the tiler before generating new training data; then add unit tests for clipping and empty-tile behavior.
8. Decide whether thermal analysis needs radiometric values. If so, preserve source bit depth/metadata and replace grayscale JPEG-style processing with a camera-aware temperature pipeline.

## Validation and maintenance notes

At handoff, all six scripts passed Python bytecode compilation. The argument help for `cli.py`, `export_aligned_rgb_thermal.py`, and `tile_drone_dataset.py` was exercised using the existing `detector_env`. Full inference was not run as part of documentation generation because it is data/model intensive and the hosted path can consume external credits.

There is no automated test suite for these workflows. High-value initial tests are:

- image-type detection and DJI offset pairing;
- annotation/calibration schema loading;
- RGB-to-thermal point transforms;
- bbox/polygon masks and background rings near image edges;
- threshold rules in `predict_dead_alive()`;
- tile start positions, clipped YOLO normalization, and empty-tile handling;
- zero-prediction and corrupt-image behavior in each detector.

The working tree at handoff also contains many generated/untracked output and environment directories. Establish a `.gitignore` and an explicit artifact/data retention policy before cleanup; do not delete them until their provenance and value are understood.
