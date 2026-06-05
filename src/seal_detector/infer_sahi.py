"""Run sliced seal instance segmentation with SAHI + Detectron2.

This script assumes a current SAHI release and a Detectron2 build compatible
with the active PyTorch environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--slice-height", type=int, default=512)
    parser.add_argument("--slice-width", type=int, default=512)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.2)
    parser.add_argument("--overlap-width-ratio", type=float, default=0.2)
    parser.add_argument("--postprocess-match-threshold", type=float, default=0.5)
    return parser.parse_args()


def bbox_to_polygon(bbox: list[float]) -> list[list[float]]:
    min_x, min_y, max_x, max_y = bbox
    return [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
    ]


def segmentation_to_polygons(segmentation: Any) -> list[list[list[float]]]:
    polygons: list[list[list[float]]] = []
    if not segmentation:
        return polygons

    for flat_polygon in segmentation:
        if len(flat_polygon) < 6:
            continue
        points = []
        for index in range(0, len(flat_polygon), 2):
            points.append([float(flat_polygon[index]), float(flat_polygon[index + 1])])
        polygons.append(points)
    return polygons


def mask_to_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        polygons.append([[float(x), float(y)] for [[x, y]] in contour])
    return polygons


def extract_polygons(object_prediction: Any) -> list[list[list[float]]]:
    mask = getattr(object_prediction, "mask", None)
    if mask is not None:
        segmentation = getattr(mask, "segmentation", None)
        polygons = segmentation_to_polygons(segmentation)
        if polygons:
            return polygons

        bool_mask = getattr(mask, "bool_mask", None)
        if bool_mask is not None:
            polygons = mask_to_polygons(np.asarray(bool_mask))
            if polygons:
                return polygons

    bbox = object_prediction.bbox.to_xyxy()
    return [bbox_to_polygon([float(value) for value in bbox])]


def serialize_prediction(object_prediction: Any) -> dict[str, Any]:
    return {
        "category_id": int(object_prediction.category.id),
        "category_name": object_prediction.category.name,
        "score": float(object_prediction.score.value),
        "bbox_xyxy": [float(value) for value in object_prediction.bbox.to_xyxy()],
        "polygons": extract_polygons(object_prediction),
    }


def main() -> None:
    args = parse_args()

    try:
        from sahi.models.detectron2 import Detectron2DetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError as exc:
        raise SystemExit("sahi is required for sliced inference.") from exc

    detection_model = Detectron2DetectionModel(
        model_path=str(args.weights),
        config_path=str(args.config_file),
        confidence_threshold=args.confidence_threshold,
        device=args.model_device,
        category_mapping={"0": "seal"},
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in args.input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    for image_path in image_paths:
        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            postprocess_match_threshold=args.postprocess_match_threshold,
        )

        predictions = [serialize_prediction(obj) for obj in result.object_prediction_list]
        output_path = args.output_dir / f"{image_path.stem}.predictions.json"
        output_path.write_text(json.dumps({"image": image_path.name, "predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
