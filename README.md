# Dead seal detection project

This repository detects seals in drone RGB imagery, pairs RGB frames with thermal frames, and produces first-pass alive/dead review labels from relative thermal-image intensity.

Start with [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md). It documents the current workflow, setup, inputs and outputs, the six main scripts, file formats, operational risks, and suggested next work. RGB/thermal control-point details are in [src/CALIBRATION.md](src/CALIBRATION.md).

The currently preferred detection path is `src/WAID_detect_sahi.py` using `model_weights/best_8.pt`. Its annotation output can be passed to `src/cli.py run` for RGB/thermal pairing and classification.

Quick smoke test with the existing local environment:

```bash
detector_env/bin/python src/cli.py --help
detector_env/bin/python src/cli.py run --help
```

Do not interpret the generated `alive`/`dead` value as a validated scientific result. The current classifier is a hand-written image-intensity heuristic and its output is intended for human review.
  
