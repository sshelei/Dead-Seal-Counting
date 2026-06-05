# Seal Detector Baseline

This package provides a minimal RGB-first seal instance segmentation baseline:

- `coco_from_plan_annotations.py`: converts your plan-style JSON polygon annotations into COCO instance segmentation.
- `train_detectron2.py`: trains a `Mask R-CNN` model in modern Detectron2.
- `infer_sahi.py`: runs sliced inference with SAHI and writes predicted polygons per image.

## Expected annotation format

Each annotation JSON should follow the structure described in your plan:

```json
{
  "image": "DJI_0249.jpg",
  "seals": [
    {"id": "seal_001", "label": "alive", "polygon": [[1000, 720], [1090, 725], [1085, 765], [995, 760]]}
  ]
}
```

The `label` field is preserved in your source data but not used for detector training. The detector learns a single class: `seal`.

## Example workflow

Convert annotations:

```bash
python -m src.seal_detector.coco_from_plan_annotations \
  --annotations-dir data/annotations/train \
  --images-dir data/images/train \
  --output-json data/coco/train.json
```

Train:

```bash
python -m src.seal_detector.train_detectron2 \
  --train-json data/coco/train.json \
  --train-images data/images/train \
  --val-json data/coco/val.json \
  --val-images data/images/val \
  --output-dir outputs/seal_mask_rcnn
```

Run sliced inference:

```bash
python -m src.seal_detector.infer_sahi \
  --input-dir data/images/infer \
  --output-dir outputs/predictions \
  --config-file outputs/seal_mask_rcnn/config.yaml \
  --weights outputs/seal_mask_rcnn/model_final.pth
```

## Notes

- This baseline assumes one foreground class: `seal`.
- `infer_sahi.py` writes JSON files with per-instance scores, boxes, and polygons.
- This baseline now targets a current Detectron2 install built against your active PyTorch environment, not `detectron2==0.5`.
- If seals are consistently small in full-resolution drone images, use SAHI at inference time and consider sliced training crops in your dataset preparation.
- The current non-Detectron2 CLI has optional RGB-to-thermal homography calibration. See [CALIBRATION.md](/home/leish/dead_seal_project/src/CALIBRATION.md:1).

## Environment setup

See [SETUP_CURRENT_ENV.md](/home/leish/dead_seal_project/src/seal_detector/SETUP_CURRENT_ENV.md:1) for instructions to use the existing project `venv`.
