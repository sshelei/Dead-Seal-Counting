# RGB/Thermal Calibration

`src/cli.py run` can optionally use per-image RGB-to-thermal control points:

```bash
python src/cli.py run \
  --input images \
  --annotations auto_annotations \
  --calibration calibration \
  --limit 4
```

Calibration files live in the folder passed to `--calibration`. A file can be named after the RGB stem, such as `calibration/DJI_0025.json`, or it can include `rgb_image`.

Each file needs at least four point pairs. Points are image pixel coordinates in `[x, y]` order:

```json
{
  "rgb_image": "DJI_0025.jpg",
  "thermal_image": "DJI_0024_R.JPG",
  "points": [
    {"rgb": [812, 436], "thermal": [143, 89]},
    {"rgb": [2194, 508], "thermal": [381, 101]},
    {"rgb": [2320, 1670], "thermal": [405, 301]},
    {"rgb": [744, 1588], "thermal": [129, 287]}
  ]
}
```

The code estimates a homography from RGB coordinates to raw thermal coordinates. With calibration present, seal boxes/polygons from RGB are transformed into thermal space before thermal features are measured. QA overlays are also generated with thermal warped back into RGB space, using filenames with `_calibrated`.

Use visible landmarks that exist in both images. Good choices are distinct rocks, beach edges, seal cluster centers, or shoreline corners. Avoid image corners unless both cameras captured the same physical corner.

If one transform works across a whole flight, create `calibration/default.json`. Per-image files take precedence over `default.json`.
