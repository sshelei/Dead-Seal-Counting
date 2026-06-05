# Setup Current Environment

These steps target the existing virtual environment at:

`/home/leish/dead_seal_project/venv`

Current observed state:

- Python `3.12.3`
- `torch==2.12.0`
- `torchvision==0.27.0`
- `detectron2` not installed
- `sahi` not installed
- compiler toolchain present: `gcc` and `g++`

This is the lowest-friction path if you want to keep using the current environment rather than create a separate Python 3.9 environment.

## 1. Activate the existing venv

```bash
cd /home/leish/dead_seal_project
source venv/bin/activate
```

## 2. Install build helpers

Detectron2 should be installed from source against the PyTorch already in this environment.

```bash
python -m pip install --upgrade pip setuptools wheel ninja
```

If `ninja` fails to install, Detectron2 can still build with the default backend, but it will usually be slower.

## 3. Install SAHI

```bash
python -m pip install -U sahi
```

## 4. Install Detectron2 from source

Use the current upstream repository so the build matches your installed PyTorch rather than trying to force an old wheel.

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If you want to pin a specific commit for reproducibility later, replace the URL with:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@<commit>'
```

## 5. Sanity check imports

```bash
python - <<'PY'
import torch
import detectron2
import sahi
print("python ok")
print("torch", torch.__version__)
print("detectron2", detectron2.__version__)
print("sahi", sahi.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_cuda", torch.version.cuda)
PY
```

## 6. Prepare training annotations

Convert your plan-style polygon annotations to COCO:

```bash
python -m src.seal_detector.coco_from_plan_annotations \
  --annotations-dir data/annotations/train \
  --images-dir data/images/train \
  --output-json data/coco/train.json
```

Do the same for validation if you have a validation split:

```bash
python -m src.seal_detector.coco_from_plan_annotations \
  --annotations-dir data/annotations/val \
  --images-dir data/images/val \
  --output-json data/coco/val.json
```

## 7. Train the baseline detector

```bash
python -m src.seal_detector.train_detectron2 \
  --train-json data/coco/train.json \
  --train-images data/images/train \
  --val-json data/coco/val.json \
  --val-images data/images/val \
  --output-dir outputs/seal_mask_rcnn \
  --device cpu
```

Use `--device cuda` only if `torch.cuda.is_available()` is `True` on your machine outside this restricted session.

## 8. Run sliced inference with SAHI

```bash
python -m src.seal_detector.infer_sahi \
  --input-dir data/images/infer \
  --output-dir outputs/predictions \
  --config-file outputs/seal_mask_rcnn/config.yaml \
  --weights outputs/seal_mask_rcnn/model_final.pth \
  --model-device cpu
```

Use `--model-device cuda:0` if GPU inference is available.

## Common failure modes

- `ModuleNotFoundError: detectron2`
  - Detectron2 did not install successfully. Re-run the source install and inspect the build output.
- `undefined symbol` or C++ extension import errors
  - Detectron2 was built against a different PyTorch than the one currently installed. Reinstall Detectron2 after confirming the `torch` version.
- `CUDA` runtime or operator errors
  - The installed PyTorch/CUDA stack does not match actual GPU availability or driver support. Fall back to CPU first to verify the pipeline.
- Very slow training
  - You are likely running on CPU. Verify `torch.cuda.is_available()` on the host machine.

## Practical recommendation

If this environment becomes unstable after future PyTorch upgrades, the reliable repair sequence is:

1. keep the same Python environment,
2. verify the `torch` version,
3. reinstall Detectron2 from source,
4. rerun the import sanity check.
