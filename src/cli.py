from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ExifTags

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
DJI_NUMBER_RE = re.compile(r"DJI_(\d+)", re.IGNORECASE)
VALID_LABELS = {"dead", "alive", "unknown"}
IGNORED_SCAN_DIRS = {".git", ".idea", ".pytest_cache", ".venv", "__pycache__", "annotations", "outputs", "classifier_outputs"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    number: int | None
    is_thermal: bool
    width: int
    height: int
    timestamp: str | None
    gps_lat: float | None
    gps_lon: float | None


@dataclass(frozen=True)
class PairRecord:
    rgb: ImageRecord
    thermal: ImageRecord | None
    method: str
    confidence: str
    timestamp_delta_seconds: float | None
    gps_delta_meters: float | None


@dataclass(frozen=True)
class CalibrationRecord:
    path: Path
    transform_rgb_to_thermal: np.ndarray
    point_count: int
    method: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pair RGB/thermal DJI images, generate QA overlays, and extract thermal features for manual seal annotations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create the local data, annotation, and output folders.")
    init_parser.add_argument("--root", type=Path, default=Path("."), help="Project root to initialize.")

    detect_parser = subparsers.add_parser("detect", help="Propose seal boxes from RGB images for later thermal classification.")
    detect_parser.add_argument(
        "--input",
        type=Path,
        default=Path("."),
        help="Input root containing mixed DJI files or RGB images.",
    )
    detect_parser.add_argument(
        "--output-annotations",
        type=Path,
        default=Path("annotations"),
        help="Folder where proposed annotation JSON files are written.",
    )
    detect_parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("classifier_outputs/detection_review"),
        help="Folder where proposed-box review images are written.",
    )
    detect_parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path("exclusion_zones"),
        help="Optional folder of per-image exclusion-zone JSON files for rocks/debris.",
    )
    detect_parser.add_argument(
        "--manual-overrides",
        type=Path,
        default=Path("manual_overrides"),
        help="Optional folder of per-image JSON files with reject_ids and manual add boxes/polygons.",
    )
    detect_parser.add_argument("--target-rgb", action="append", default=[], help="Detect only matching RGB filenames or stems.")
    detect_parser.add_argument("--limit", type=int, default=None, help="Limit number of RGB frames processed.")
    detect_parser.add_argument("--max-candidates", type=int, default=150, help="Maximum proposed seal boxes per image.")
    detect_parser.add_argument(
        "--sensitivity",
        choices=["high_recall", "balanced", "strict"],
        default="high_recall",
        help="Detection sensitivity. high_recall misses fewer seals; strict removes more rocks.",
    )

    run_parser = subparsers.add_parser("run", help="Run pairing QA and optional seal thermal feature extraction.")
    run_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data"),
        help="Input root. Supports either mixed DJI files or data/rgb and data/thermal subfolders.",
    )
    run_parser.add_argument("--annotations", type=Path, default=Path("annotations"), help="Manual annotation folder.")
    run_parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration"),
        help="Optional folder of RGB-to-thermal control-point JSON files.",
    )
    run_parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output folder.")
    run_parser.add_argument(
        "--thermal-offset",
        type=int,
        default=-1,
        help="Expected thermal number relative to RGB number. Use -1 for RGB DJI_0249 -> thermal DJI_0248_R.",
    )
    run_parser.add_argument("--limit", type=int, default=None, help="Limit number of RGB frames processed.")
    run_parser.add_argument(
        "--target-rgb",
        action="append",
        default=[],
        help="Process only matching RGB filenames or stems. May be passed multiple times.",
    )
    run_parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=80.0,
        help="Percentile threshold for thermal binary masks.",
    )
    run_parser.add_argument(
        "--min-label-count",
        type=int,
        default=5,
        help="Minimum dead and alive examples needed before classifier work is marked ready.",
    )

    args = parser.parse_args(argv)
    if args.command == "init":
        init_layout(args.root)
        return 0
    if args.command == "detect":
        run_detection(
            input_root=args.input,
            output_annotations=args.output_annotations,
            review_output=args.review_output,
            exclusions_root=args.exclusions,
            manual_overrides_root=args.manual_overrides,
            target_rgb=set(args.target_rgb),
            limit=args.limit,
            max_candidates=args.max_candidates,
            sensitivity=args.sensitivity,
        )
        return 0
    if args.command == "run":
        run_pipeline(
            input_root=args.input,
            annotations_root=args.annotations,
            calibration_root=args.calibration,
            output_root=args.output,
            thermal_offset=args.thermal_offset,
            limit=args.limit,
            target_rgb=set(args.target_rgb),
            threshold_percentile=args.threshold_percentile,
            min_label_count=args.min_label_count,
        )
        return 0
    raise ValueError(f"Unknown command: {args.command}")


def init_layout(root: Path) -> None:
    for relative in ["data/rgb", "data/thermal", "annotations", "outputs/qa"]:
        (root / relative).mkdir(parents=True, exist_ok=True)


def run_detection(
    input_root: Path,
    output_annotations: Path,
    review_output: Path,
    exclusions_root: Path,
    manual_overrides_root: Path,
    target_rgb: set[str],
    limit: int | None,
    max_candidates: int,
    sensitivity: str,
) -> None:
    output_annotations.mkdir(parents=True, exist_ok=True)
    review_output.mkdir(parents=True, exist_ok=True)
    records = scan_images(input_root)
    rgb_records = sorted((r for r in records if not r.is_thermal), key=lambda r: (r.number is None, r.number or 0, r.path.name))
    if target_rgb:
        wanted = {normalize_name(item) for item in target_rgb}
        rgb_records = [r for r in rgb_records if normalize_name(r.path.name) in wanted or normalize_name(r.path.stem) in wanted]
    if limit is not None:
        rgb_records = rgb_records[:limit]

    total_candidates = 0
    candidate_rows: list[dict[str, Any]] = []
    for record in rgb_records:
        annotations = propose_seal_annotations(record.path, max_candidates=max_candidates, sensitivity=sensitivity)
        exclusion_zones = load_exclusion_zones(exclusions_root, record.path)
        if exclusion_zones:
            annotations = filter_excluded_annotations(annotations, exclusion_zones)
        renumber_annotations(annotations)
        annotations = apply_manual_overrides(annotations, load_manual_overrides(manual_overrides_root, record.path), record.path)
        renumber_annotations(annotations)
        total_candidates += len(annotations)
        candidate_rows.extend(candidate_csv_rows(record.path.name, annotations))
        data = {"image": record.path.name, "seals": annotations}
        annotation_path = output_annotations / f"{record.path.stem}.json"
        annotation_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        write_detection_review_image(record.path, annotations, review_output / f"{record.path.stem}_detected_candidates.jpg", exclusion_zones)
        print(f"Detected {len(annotations)} candidates in {record.path.name}; wrote {annotation_path}")
    write_csv(review_output / "detection_candidates.csv", candidate_rows, DETECTION_FIELDS)
    print(f"Detection complete: {len(rgb_records)} RGB images, {total_candidates} total candidates.")


def propose_seal_annotations(rgb_path: Path, max_candidates: int, sensitivity: str = "high_recall") -> list[dict[str, Any]]:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read RGB image: {rgb_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    dark_mask = build_dark_object_mask(blur)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, dict[str, Any]]] = []

    for contour in contours:
        candidate = candidate_from_contour(contour, gray, enhanced)
        if candidate is not None:
            candidates.append(candidate)
    candidates.extend(mser_candidates(gray, gray, polarity="dark"))
    candidates.extend(mser_candidates(255 - gray, gray, polarity="light"))
    candidates.extend(local_blob_candidates(blur, gray, polarity="dark"))
    candidates.extend(local_blob_candidates(255 - blur, gray, polarity="light"))
    candidates.extend(residual_candidates(gray, polarity="dark"))
    candidates.extend(residual_candidates(gray, polarity="light"))
    candidates.extend(edge_shape_candidates(enhanced, gray))
    if sensitivity == "high_recall":
        candidates.extend(review_residual_candidates(gray))
    candidates = add_group_split_candidates(candidates, gray)

    candidates = suppress_overlapping_candidates(
        sorted(candidates, key=lambda item: item[0], reverse=True),
        iou_threshold=overlap_threshold(sensitivity),
    )
    candidates = filter_dense_rock_clusters(candidates[:max_candidates], sensitivity=sensitivity)
    annotations = [candidate for _, candidate in candidates]
    annotations = sorted(annotations, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, annotation in enumerate(annotations, start=1):
        annotation["id"] = f"auto_seal_{index:03d}"
    return annotations


def build_dark_object_mask(gray: np.ndarray) -> np.ndarray:
    local_dark = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        81,
        4,
    )
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 25))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
    _, blackhat_mask = cv2.threshold(blackhat, float(np.percentile(blackhat, 91)), 255, cv2.THRESH_BINARY)
    global_dark = np.where(gray <= np.percentile(gray, 30), 255, 0).astype(np.uint8)
    combined = cv2.bitwise_or(local_dark, blackhat_mask.astype(np.uint8))
    combined = cv2.bitwise_or(combined, global_dark)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    return combined


def candidate_from_contour(contour: np.ndarray, gray: np.ndarray, enhanced: np.ndarray) -> tuple[float, dict[str, Any]] | None:
    image_area = gray.shape[0] * gray.shape[1]
    area = float(cv2.contourArea(contour))
    min_area = max(25, image_area * 0.000003)
    max_area = image_area * 0.00120
    if area < min_area or area > max_area:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if x <= 2 or y <= 2 or x + w >= gray.shape[1] - 2 or y + h >= gray.shape[0] - 2:
        return None
    if w < 8 or h < 5 or w > 165 or h > 125:
        return None
    bbox_aspect = max(w, h) / max(1, min(w, h))
    if bbox_aspect < 1.05 or bbox_aspect > 8.2:
        return None
    extent = area / max(w * h, 1)
    if extent < 0.10 or extent > 0.86:
        return None
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area else 0
    if solidity < 0.22:
        return None

    pad = max(10, int(round(max(w, h) * 0.6)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = gray[y : y + h, x : x + w]
    local = gray[y0:y1, x0:x1]
    if roi.size == 0 or local.size == 0:
        return None
    ring_mask = np.ones(local.shape, dtype=np.uint8)
    ring_mask[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = 0
    ring_pixels = local[ring_mask > 0]
    if ring_pixels.size == 0:
        return None
    relative_dark = float(np.median(ring_pixels) - np.median(roi))
    if relative_dark < 1.5:
        return None

    edges = cv2.Canny(local, 40, 120)
    local_edge_density = float(np.count_nonzero(edges) / edges.size) if edges.size else 0
    roi_std = float(np.std(roi))
    rock_penalty = local_edge_density * 2.7 + min(roi_std / 55.0, 1.3)
    if rock_penalty > 1.85 and relative_dark < 8:
        return None

    ideal_aspect_score = max(0.1, 1.0 - abs(math.log(max(bbox_aspect, 0.01) / 2.6)) / 2.0)
    score = relative_dark * 28 + area * ideal_aspect_score * 0.55 + solidity * 85 - rock_penalty * 95
    review_flag = "rock_like_review" if rock_penalty > 1.25 and relative_dark < 16 else "likely_seal"
    expanded = expand_bbox([x, y, w, h], gray.shape[1], gray.shape[0], pad=candidate_padding(w, h))
    annotation = {
        "id": "",
        "label": "unknown",
        "bbox": expanded,
        "detection_score": round(float(score), 4),
        "relative_dark": round(relative_dark, 4),
        "rock_penalty": round(float(rock_penalty), 4),
        "bbox_aspect": round(float(bbox_aspect), 4),
        "review_flag": review_flag,
    }
    return score, annotation


def mser_candidates(detection_gray: np.ndarray, scoring_gray: np.ndarray, polarity: str) -> list[tuple[float, dict[str, Any]]]:
    mser = cv2.MSER_create(5, 35, 6500, 0.25, 0.20)
    _, boxes = mser.detectRegions(detection_gray)
    candidates: list[tuple[float, dict[str, Any]]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for x, y, w, h in boxes:
        key = (int(x // 3), int(y // 3), int(w // 3), int(h // 3))
        if key in seen:
            continue
        seen.add(key)
        candidate = candidate_from_bbox([int(x), int(y), int(w), int(h)], scoring_gray, source=f"mser_{polarity}", polarity=polarity)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def local_blob_candidates(detection_gray: np.ndarray, scoring_gray: np.ndarray, polarity: str) -> list[tuple[float, dict[str, Any]]]:
    """Catch low-contrast elongated seals that MSER often misses on sandy areas."""
    threshold = cv2.adaptiveThreshold(
        detection_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        61,
        2,
    )
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        candidate = candidate_from_bbox(
            [int(x), int(y), int(w), int(h)],
            scoring_gray,
            source=f"local_blob_{polarity}",
            polarity=polarity,
        )
        if candidate is not None:
            score, annotation = candidate
            annotation["review_flag"] = "low_contrast_review" if annotation.get("review_flag") == "likely_seal" else annotation["review_flag"]
            candidates.append((score * 0.68, annotation))
    return candidates


def residual_candidates(gray: np.ndarray, polarity: str) -> list[tuple[float, dict[str, Any]]]:
    """Find seal-shaped objects by comparing each pixel to a smoothed local beach background."""
    candidates: list[tuple[float, dict[str, Any]]] = []
    for sigma in [7.0, 12.0, 20.0]:
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if polarity == "dark":
            residual = cv2.subtract(background, gray)
        else:
            residual = cv2.subtract(gray, background)
        threshold_percentile = 93.0 if polarity == "light" else 95.5
        threshold_value = max(3.0, float(np.percentile(residual, threshold_percentile)))
        _, mask = cv2.threshold(residual, threshold_value, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 7))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            candidate = candidate_from_bbox(
                [int(x), int(y), int(w), int(h)],
                gray,
                source=f"residual_{polarity}_{int(sigma)}",
                polarity=polarity,
            )
            if candidate is None:
                continue
            score, annotation = candidate
            residual_roi = residual[y : y + h, x : x + w]
            residual_strength = float(np.mean(residual_roi)) if residual_roi.size else 0.0
            annotation["residual_strength"] = round(residual_strength, 4)
            if annotation.get("review_flag") == "likely_seal":
                annotation["review_flag"] = "residual_review"
            candidates.append((score * 0.82 + residual_strength * 22, annotation))
    return candidates


def review_residual_candidates(gray: np.ndarray) -> list[tuple[float, dict[str, Any]]]:
    """Add broader low-contrast body proposals learned from manual review marks.

    These are intentionally capped and flagged for review. They catch seals whose
    body and shadow are visible to a human but too low-contrast or too square for
    the stricter automatic sources.
    """
    candidates: list[tuple[float, dict[str, Any]]] = []
    configs = [
        (16.0, 94.0),
        (24.0, 94.0),
        (32.0, 90.0),
        (24.0, 85.0),
    ]
    for sigma, percentile in configs:
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        residual = cv2.subtract(background, gray)
        threshold_value = max(2.0, float(np.percentile(residual, percentile)))
        _, mask = cv2.threshold(residual, threshold_value, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 9))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            candidate = candidate_from_review_residual_bbox(
                [int(x), int(y), int(w), int(h)],
                gray,
                source=f"review_residual_dark_{int(sigma)}_{int(percentile)}",
            )
            if candidate is not None:
                candidates.append(candidate)

    candidates = suppress_overlapping_candidates(
        sorted(candidates, key=lambda item: item[0], reverse=True),
        iou_threshold=0.38,
    )
    return candidates[:80]


def candidate_from_review_residual_bbox(bbox: list[int], gray: np.ndarray, source: str) -> tuple[float, dict[str, Any]] | None:
    x, y, w, h = bbox
    if x <= 2 or y <= 2 or x + w >= gray.shape[1] - 2 or y + h >= gray.shape[0] - 2:
        return None
    if w < 12 or h < 8 or w > 175 or h > 145:
        return None
    bbox_area = w * h
    if bbox_area < 220 or bbox_area > 14000:
        return None
    bbox_aspect = max(w, h) / max(1, min(w, h))
    if bbox_aspect < 1.0 or bbox_aspect > 4.5:
        return None

    pad = max(12, int(round(max(w, h) * 0.75)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = gray[y : y + h, x : x + w]
    local = gray[y0:y1, x0:x1]
    if roi.size == 0 or local.size == 0:
        return None

    ring_mask = np.ones(local.shape, dtype=np.uint8)
    ring_mask[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = 0
    ring_pixels = local[ring_mask > 0]
    if ring_pixels.size == 0:
        return None

    signed_contrast = float(np.median(ring_pixels) - np.median(roi))
    contrast_abs = abs(signed_contrast)
    roi_range = float(np.percentile(roi, 92) - np.percentile(roi, 8))
    if contrast_abs < 1.0 and roi_range < 42:
        return None

    edges = cv2.Canny(local, 40, 120)
    local_edge_density = float(np.count_nonzero(edges) / edges.size) if edges.size else 0
    roi_std = float(np.std(roi))
    rock_penalty = local_edge_density * 2.35 + min(roi_std / 68.0, 1.18)
    if rock_penalty > 2.05 and contrast_abs < 6 and roi_range < 85:
        return None

    ideal_aspect_score = max(0.1, 1.0 - abs(math.log(max(bbox_aspect, 0.01) / 1.45)) / 1.7)
    score = roi_range * 18 + contrast_abs * 24 + bbox_area * ideal_aspect_score * 0.09 - rock_penalty * 80
    expanded = expand_bbox([x, y, w, h], gray.shape[1], gray.shape[0], pad=max(5, int(round(max(w, h) * 0.16))))
    annotation = {
        "id": "",
        "label": "unknown",
        "bbox": expanded,
        "detection_score": round(float(score), 4),
        "relative_dark": round(signed_contrast, 4),
        "detection_contrast": round(float(contrast_abs), 4),
        "rock_penalty": round(float(rock_penalty), 4),
        "bbox_aspect": round(float(bbox_aspect), 4),
        "residual_strength": "",
        "review_flag": "review_residual",
        "detection_source": source,
        "detection_polarity": "dark",
    }
    return score, annotation


def edge_shape_candidates(enhanced: np.ndarray, gray: np.ndarray) -> list[tuple[float, dict[str, Any]]]:
    """Find elongated seal bodies whose shadow and highlight cancel out in simple intensity tests."""
    candidates: list[tuple[float, dict[str, Any]]] = []
    for low, high, close_size in [(24, 72, (15, 7)), (36, 108, (23, 9))]:
        edges = cv2.Canny(enhanced, low, high)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_size)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        connected = cv2.morphologyEx(connected, cv2.MORPH_OPEN, open_kernel, iterations=1)
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            candidate = candidate_from_shape_bbox(
                [int(x), int(y), int(w), int(h)],
                gray,
                source=f"edge_shape_{low}_{high}",
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def candidate_from_shape_bbox(bbox: list[int], gray: np.ndarray, source: str) -> tuple[float, dict[str, Any]] | None:
    x, y, w, h = bbox
    if x <= 2 or y <= 2 or x + w >= gray.shape[1] - 2 or y + h >= gray.shape[0] - 2:
        return None
    if w < 12 or h < 7 or w > 175 or h > 130:
        return None
    bbox_area = w * h
    if bbox_area < 120 or bbox_area > 11500:
        return None
    bbox_aspect = max(w, h) / max(1, min(w, h))
    if bbox_aspect < 1.18 or bbox_aspect > 8.5:
        return None

    pad = max(12, int(round(max(w, h) * 0.75)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = gray[y : y + h, x : x + w]
    local = gray[y0:y1, x0:x1]
    if roi.size == 0 or local.size == 0:
        return None

    ring_mask = np.ones(local.shape, dtype=np.uint8)
    ring_mask[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = 0
    ring_pixels = local[ring_mask > 0]
    if ring_pixels.size == 0:
        return None

    signed_contrast = float(np.median(ring_pixels) - np.median(roi))
    contrast_abs = abs(signed_contrast)
    roi_range = float(np.percentile(roi, 90) - np.percentile(roi, 10))
    if contrast_abs < 1.0 and roi_range < 9.0:
        return None

    roi_edges = cv2.Canny(roi, 35, 115)
    local_edges = cv2.Canny(local, 35, 115)
    roi_edge_density = float(np.count_nonzero(roi_edges) / roi_edges.size) if roi_edges.size else 0.0
    local_edge_density = float(np.count_nonzero(local_edges) / local_edges.size) if local_edges.size else 0.0
    edge_lift = roi_edge_density - local_edge_density
    if edge_lift < -0.015 and contrast_abs < 3:
        return None

    roi_std = float(np.std(roi))
    rock_penalty = local_edge_density * 2.9 + min(roi_std / 70.0, 1.15)
    if rock_penalty > 2.15 and contrast_abs < 4 and roi_range < 16:
        return None

    ideal_aspect_score = max(0.1, 1.0 - abs(math.log(max(bbox_aspect, 0.01) / 2.9)) / 2.2)
    score = contrast_abs * 20 + roi_range * 16 + bbox_area * ideal_aspect_score * 0.06 + max(edge_lift, 0) * 900 - rock_penalty * 70
    expanded = expand_bbox([x, y, w, h], gray.shape[1], gray.shape[0], pad=candidate_padding(w, h))
    annotation = {
        "id": "",
        "label": "unknown",
        "bbox": expanded,
        "detection_score": round(float(score), 4),
        "relative_dark": round(signed_contrast, 4),
        "detection_contrast": round(float(contrast_abs), 4),
        "rock_penalty": round(float(rock_penalty), 4),
        "bbox_aspect": round(float(bbox_aspect), 4),
        "residual_strength": "",
        "review_flag": "edge_review",
        "detection_source": source,
        "detection_polarity": "mixed",
    }
    return score, annotation


def add_group_split_candidates(candidates: list[tuple[float, dict[str, Any]]], gray: np.ndarray) -> list[tuple[float, dict[str, Any]]]:
    expanded: list[tuple[float, dict[str, Any]]] = []
    for score, candidate in candidates:
        splits = split_group_candidate(score, candidate, gray)
        if splits:
            expanded.extend(splits)
        expanded.append((score, candidate))
    return expanded


def split_group_candidate(score: float, candidate: dict[str, Any], gray: np.ndarray) -> list[tuple[float, dict[str, Any]]]:
    x, y, w, h = [int(v) for v in candidate["bbox"]]
    area = w * h
    long_len = max(w, h)
    short_len = min(w, h)
    aspect = long_len / max(1, short_len)
    if long_len < 58 or area < 1050:
        return []
    if aspect < 1.55 and area < 1800:
        return []

    expected_single = max(28, min(48, int(round(short_len * 1.25))))
    split_count = int(round(long_len / expected_single))
    if long_len >= 70 and aspect >= 1.75:
        split_count = max(split_count, 2)
    split_count = max(2, min(4, split_count))
    if long_len / split_count < 18:
        return []

    roi = gray[y : y + h, x : x + w]
    if roi.size == 0:
        return []
    axis = 1 if w >= h else 0
    mask = group_foreground_mask(roi)
    cuts = group_split_cuts(mask, split_count, axis=axis)
    if len(cuts) < 3:
        return []

    splits: list[tuple[float, dict[str, Any]]] = []
    for index, (start, end) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        if end - start < 8:
            continue
        sub_bbox = split_segment_bbox(mask, x, y, w, h, start, end, axis=axis)
        if sub_bbox is None:
            continue
        sx, sy, sw, sh = sub_bbox
        if sw < 10 or sh < 6 or sw * sh < 80:
            continue
        sub_aspect = max(sw, sh) / max(1, min(sw, sh))
        if sub_aspect > 8.5:
            continue
        split_score = float(score + 450 - index)
        annotation = {
            "id": "",
            "label": "unknown",
            "bbox": sub_bbox,
            "detection_score": round(split_score, 4),
            "relative_dark": candidate.get("relative_dark", ""),
            "detection_contrast": candidate.get("detection_contrast", candidate.get("relative_dark", "")),
            "rock_penalty": candidate.get("rock_penalty", ""),
            "bbox_aspect": round(float(sub_aspect), 4),
            "review_flag": "group_split_review",
            "detection_source": f"group_split_{candidate.get('detection_source', 'candidate')}",
            "detection_polarity": candidate.get("detection_polarity", ""),
            "parent_bbox": candidate["bbox"],
            "group_split_index": index,
            "group_split_count": len(cuts) - 1,
        }
        splits.append((split_score, annotation))
    return splits if len(splits) >= 2 else []


def group_foreground_mask(roi: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    background = cv2.GaussianBlur(blur, (0, 0), sigmaX=7.0, sigmaY=7.0)
    dark = cv2.subtract(background, blur)
    light = cv2.subtract(blur, background)
    diff = cv2.max(dark, light)
    threshold = max(3.0, float(np.percentile(diff, 65)))
    mask = np.where(diff >= threshold, 255, 0).astype(np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask, dilate_kernel, iterations=1)


def group_split_cuts(mask: np.ndarray, split_count: int, axis: int) -> list[int]:
    length = mask.shape[1] if axis == 1 else mask.shape[0]
    if split_count < 2 or length < split_count * 8:
        return []
    projection = np.count_nonzero(mask, axis=0 if axis == 1 else 1).astype(np.float32)
    if projection.size == 0 or float(np.max(projection)) <= 0:
        return []
    smooth_width = max(3, int(round(length / 24)))
    if smooth_width % 2 == 0:
        smooth_width += 1
    if projection.size >= smooth_width:
        kernel = np.ones(smooth_width, dtype=np.float32) / smooth_width
        projection = np.convolve(projection, kernel, mode="same")
    cuts = [0]
    accepted_valleys: list[float] = []
    for split_index in range(1, split_count):
        target = int(round(length * split_index / split_count))
        radius = max(4, int(round(length / split_count * 0.28)))
        left = max(cuts[-1] + 6, target - radius)
        right = min(length - 6, target + radius)
        if right <= left:
            return []
        valley = int(left + np.argmin(projection[left:right]))
        left_peak = float(np.max(projection[max(0, left - radius) : max(left, valley)])) if valley > left else 0.0
        right_peak = float(np.max(projection[min(right, valley + 1) : min(length, right + radius)])) if valley + 1 < right else 0.0
        neighbor_peak = max(left_peak, right_peak, 1.0)
        valley_ratio = float(projection[valley]) / neighbor_peak
        if valley_ratio > 0.55:
            return []
        accepted_valleys.append(valley_ratio)
        cuts.append(valley)
    cuts.append(length)
    if not accepted_valleys or min(accepted_valleys) > 0.52:
        return []
    unique_cuts = sorted(set(cuts))
    return unique_cuts if len(unique_cuts) == split_count + 1 else []


def split_segment_bbox(mask: np.ndarray, x: int, y: int, w: int, h: int, start: int, end: int, axis: int) -> list[int] | None:
    image_width = x + mask.shape[1]
    image_height = y + mask.shape[0]
    if axis == 1:
        return expand_bbox([x + start, y, end - start, h], image_width, image_height, pad=3)
    return expand_bbox([x, y + start, w, end - start], image_width, image_height, pad=3)


def candidate_from_bbox(bbox: list[int], gray: np.ndarray, source: str, polarity: str) -> tuple[float, dict[str, Any]] | None:
    x, y, w, h = bbox
    if x <= 2 or y <= 2 or x + w >= gray.shape[1] - 2 or y + h >= gray.shape[0] - 2:
        return None
    if w < 10 or h < 6 or w > 150 or h > 115:
        return None
    bbox_area = w * h
    if bbox_area < 80 or bbox_area > 9500:
        return None
    bbox_aspect = max(w, h) / max(1, min(w, h))
    if bbox_aspect < 1.12 or bbox_aspect > 8.0:
        return None

    pad = max(10, int(round(max(w, h) * 0.65)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    roi = gray[y : y + h, x : x + w]
    local = gray[y0:y1, x0:x1]
    if roi.size == 0 or local.size == 0:
        return None
    ring_mask = np.ones(local.shape, dtype=np.uint8)
    ring_mask[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = 0
    ring_pixels = local[ring_mask > 0]
    if ring_pixels.size == 0:
        return None

    contrast = float(np.median(ring_pixels) - np.median(roi))
    if polarity == "light":
        contrast *= -1
    if contrast < 1.2:
        return None
    edges = cv2.Canny(local, 40, 120)
    local_edge_density = float(np.count_nonzero(edges) / edges.size) if edges.size else 0
    roi_std = float(np.std(roi))
    rock_penalty = local_edge_density * 2.4 + min(roi_std / 60.0, 1.2)
    if rock_penalty > 2.05 and contrast < 9:
        return None

    ideal_aspect_score = max(0.1, 1.0 - abs(math.log(max(bbox_aspect, 0.01) / 2.8)) / 2.2)
    score = contrast * 30 + bbox_area * ideal_aspect_score * 0.11 - rock_penalty * 85
    if polarity == "light":
        score *= 0.95
    review_flag = "rock_like_review" if rock_penalty > 1.25 and contrast < 18 else "likely_seal"
    expanded = expand_bbox([x, y, w, h], gray.shape[1], gray.shape[0], pad=candidate_padding(w, h))
    annotation = {
        "id": "",
        "label": "unknown",
        "bbox": expanded,
        "detection_score": round(float(score), 4),
        "relative_dark": round(contrast if polarity == "dark" else -contrast, 4),
        "detection_contrast": round(float(contrast), 4),
        "rock_penalty": round(float(rock_penalty), 4),
        "bbox_aspect": round(float(bbox_aspect), 4),
        "review_flag": review_flag,
        "detection_source": source,
        "detection_polarity": polarity,
    }
    return score, annotation


def expand_bbox(bbox: list[int], width: int, height: int, pad: int) -> list[int]:
    x, y, w, h = bbox
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


def candidate_padding(width: int, height: int) -> int:
    return max(5, int(round(max(width, height) * 0.18)))


def suppress_overlapping_candidates(candidates: list[tuple[float, dict[str, Any]]], iou_threshold: float = 0.25) -> list[tuple[float, dict[str, Any]]]:
    kept: list[tuple[float, dict[str, Any]]] = []
    for score, candidate in candidates:
        if all(bbox_iou(candidate["bbox"], kept_candidate["bbox"]) < iou_threshold for _, kept_candidate in kept):
            kept.append((score, candidate))
    return kept


def overlap_threshold(sensitivity: str) -> float:
    if sensitivity == "strict":
        return 0.22
    if sensitivity == "balanced":
        return 0.30
    return 0.38


def filter_dense_rock_clusters(candidates: list[tuple[float, dict[str, Any]]], sensitivity: str = "high_recall") -> list[tuple[float, dict[str, Any]]]:
    settings = rock_filter_settings(sensitivity)
    pebble_indices = [index for index, (_, candidate) in enumerate(candidates) if is_pebble_like_candidate(candidate, settings)]
    if not pebble_indices:
        return candidates

    centers = [bbox_center(candidate["bbox"]) for _, candidate in candidates]
    remove: set[int] = set()
    for index in pebble_indices:
        cx, cy = centers[index]
        if settings["remove_isolated_strong"] and is_strong_pebble_candidate(candidates[index][1]):
            remove.add(index)
            continue
        neighbor_count = 0
        for other_index in pebble_indices:
            if other_index == index:
                continue
            ox, oy = centers[other_index]
            if abs(cx - ox) <= settings["neighbor_x"] and abs(cy - oy) <= settings["neighbor_y"]:
                neighbor_count += 1
        if neighbor_count >= settings["neighbor_threshold"]:
            remove.add(index)

    return [candidate for index, candidate in enumerate(candidates) if index not in remove]


def rock_filter_settings(sensitivity: str) -> dict[str, Any]:
    if sensitivity == "strict":
        return {
            "area_max": 850,
            "aspect_max": 2.35,
            "rock_min": 1.55,
            "contrast_min": 42,
            "neighbor_x": 220,
            "neighbor_y": 170,
            "neighbor_threshold": 3,
            "remove_isolated_strong": True,
        }
    if sensitivity == "balanced":
        return {
            "area_max": 760,
            "aspect_max": 2.25,
            "rock_min": 1.62,
            "contrast_min": 45,
            "neighbor_x": 200,
            "neighbor_y": 155,
            "neighbor_threshold": 5,
            "remove_isolated_strong": False,
        }
    return {
        "area_max": 700,
        "aspect_max": 2.35,
        "rock_min": 1.65,
        "contrast_min": 45,
        "neighbor_x": 170,
        "neighbor_y": 140,
        "neighbor_threshold": 6,
        "remove_isolated_strong": False,
    }


def is_pebble_like_candidate(candidate: dict[str, Any], settings: dict[str, Any]) -> bool:
    x, y, w, h = [int(v) for v in candidate["bbox"]]
    area = w * h
    aspect = float(candidate.get("bbox_aspect") or max(w, h) / max(1, min(w, h)))
    rock_penalty = float(candidate.get("rock_penalty") or 0)
    contrast = abs(float(candidate.get("detection_contrast") or candidate.get("relative_dark") or 0))
    return (
        area <= settings["area_max"]
        and aspect <= settings["aspect_max"]
        and rock_penalty >= settings["rock_min"]
        and contrast >= settings["contrast_min"]
    )


def is_strong_pebble_candidate(candidate: dict[str, Any]) -> bool:
    x, y, w, h = [int(v) for v in candidate["bbox"]]
    area = w * h
    aspect = float(candidate.get("bbox_aspect") or max(w, h) / max(1, min(w, h)))
    rock_penalty = float(candidate.get("rock_penalty") or 0)
    contrast = abs(float(candidate.get("detection_contrast") or candidate.get("relative_dark") or 0))
    return area <= 520 and aspect <= 1.9 and rock_penalty >= 1.72 and contrast >= 55


def bbox_center(bbox: list[int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return float(x + w / 2), float(y + h / 2)


def bbox_iou(first: list[int], second: list[int]) -> float:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union else 0.0


def load_exclusion_zones(exclusions_root: Path, rgb_path: Path) -> list[list[list[int]]]:
    candidates = [
        exclusions_root / f"{rgb_path.stem}.json",
        exclusions_root / f"{rgb_path.name}.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            zones = data.get("exclude_polygons", [])
            return zones if isinstance(zones, list) else []
    return []


def load_manual_overrides(overrides_root: Path, rgb_path: Path) -> dict[str, Any]:
    candidates = [
        overrides_root / f"{rgb_path.stem}.json",
        overrides_root / f"{rgb_path.name}.json",
    ]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError(f"Manual override file must be a JSON object: {path}")
            return data
    return {}


def apply_manual_overrides(
    annotations: list[dict[str, Any]],
    overrides: dict[str, Any],
    rgb_path: Path,
) -> list[dict[str, Any]]:
    if not overrides:
        return annotations

    reject_ids = {normalize_reject_id(value) for value in overrides.get("reject_ids", [])}
    kept = [annotation for annotation in annotations if normalize_reject_id(annotation.get("id", "")) not in reject_ids]

    for index, addition in enumerate(overrides.get("add", []), start=1):
        if not isinstance(addition, dict):
            raise ValueError(f"Manual add entry must be an object for {rgb_path.name}: {addition}")
        manual = dict(addition)
        manual.setdefault("id", f"manual_seal_{index:03d}")
        manual.setdefault("label", "unknown")
        manual.setdefault("detection_source", "manual_override")
        manual.setdefault("review_flag", "manual_added")
        validate_annotation(manual, rgb_path)
        kept.append(manual)

    return kept


def normalize_reject_id(value: Any) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)$", text)
    return match.group(1).zfill(3) if match else text


def filter_excluded_annotations(annotations: list[dict[str, Any]], exclusion_zones: list[list[list[int]]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    polygons = [np.array(zone, dtype=np.int32) for zone in exclusion_zones if len(zone) >= 3]
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        center = (float(x + w / 2), float(y + h / 2))
        if any(cv2.pointPolygonTest(polygon, center, False) >= 0 for polygon in polygons):
            continue
        kept.append(annotation)
    return kept


def renumber_annotations(annotations: list[dict[str, Any]]) -> None:
    annotations.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    for index, annotation in enumerate(annotations, start=1):
        annotation["id"] = f"auto_seal_{index:03d}"


def write_detection_review_image(
    rgb_path: Path,
    annotations: list[dict[str, Any]],
    output_path: Path,
    exclusion_zones: list[list[list[int]]] | None = None,
) -> None:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read RGB image: {rgb_path}")
    review = draw_detection_candidates(image, annotations, exclusion_zones or [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), review)
    write_review_tiles(review, output_path.parent / "tiles", output_path.stem)


DETECTION_FIELDS = [
    "rgb_filename",
    "seal_id",
    "label",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "detection_score",
    "relative_dark",
    "detection_contrast",
    "rock_penalty",
    "bbox_aspect",
    "residual_strength",
    "group_split_index",
    "group_split_count",
    "parent_bbox",
    "review_flag",
    "detection_source",
    "detection_polarity",
]


def candidate_csv_rows(rgb_filename: str, annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        rows.append(
            {
                "rgb_filename": rgb_filename,
                "seal_id": annotation["id"],
                "label": annotation.get("label", "unknown"),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "bbox_area": w * h,
                "detection_score": annotation.get("detection_score", ""),
                "relative_dark": annotation.get("relative_dark", ""),
                "detection_contrast": annotation.get("detection_contrast", ""),
                "rock_penalty": annotation.get("rock_penalty", ""),
                "bbox_aspect": annotation.get("bbox_aspect", ""),
                "residual_strength": annotation.get("residual_strength", ""),
                "group_split_index": annotation.get("group_split_index", ""),
                "group_split_count": annotation.get("group_split_count", ""),
                "parent_bbox": json.dumps(annotation.get("parent_bbox", "")) if annotation.get("parent_bbox") else "",
                "review_flag": annotation.get("review_flag", ""),
                "detection_source": annotation.get("detection_source", ""),
                "detection_polarity": annotation.get("detection_polarity", ""),
            }
        )
    return rows


def draw_detection_candidates(
    image: np.ndarray,
    annotations: list[dict[str, Any]],
    exclusion_zones: list[list[list[int]]],
) -> np.ndarray:
    result = image.copy()
    for zone in exclusion_zones:
        if len(zone) >= 3:
            points = np.array(zone, dtype=np.int32)
            cv2.polylines(result, [points], isClosed=True, color=(0, 0, 255), thickness=5)
    for annotation in annotations:
        x, y, w, h = [int(v) for v in annotation["bbox"]]
        color = (0, 180, 255) if annotation.get("review_flag") == "rock_like_review" else (255, 220, 0)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 3)
        short_id = compact_seal_id(str(annotation["id"]))
        draw_readable_label(result, short_id, x, max(24, y - 8), color, scale=0.65)
    return result


def write_review_tiles(image: np.ndarray, output_dir: Path, base_name: str, tile_size: int = 900, overlap: int = 120) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    step = max(1, tile_size - overlap)
    y_positions = tile_starts(height, tile_size, step)
    x_positions = tile_starts(width, tile_size, step)
    for row, y in enumerate(y_positions):
        for col, x in enumerate(x_positions):
            tile = image[y : min(y + tile_size, height), x : min(x + tile_size, width)]
            tile_path = output_dir / f"{base_name}_r{row:02d}_c{col:02d}_x{x}_y{y}.jpg"
            cv2.imwrite(str(tile_path), tile)


def tile_starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def run_pipeline(
    input_root: Path,
    annotations_root: Path,
    calibration_root: Path,
    output_root: Path,
    thermal_offset: int,
    limit: int | None,
    target_rgb: set[str],
    threshold_percentile: float,
    min_label_count: int,
) -> None:
    qa_root = output_root / "qa"
    mask_root = output_root / "thermal_masks"
    qa_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    records = scan_images(input_root)
    rgb_records = sorted((r for r in records if not r.is_thermal), key=lambda r: (r.number is None, r.number or 0, r.path.name))
    thermal_records = [r for r in records if r.is_thermal]
    if target_rgb:
        wanted = {normalize_name(item) for item in target_rgb}
        rgb_records = [r for r in rgb_records if normalize_name(r.path.name) in wanted or normalize_name(r.path.stem) in wanted]
    if limit is not None:
        rgb_records = rgb_records[:limit]

    pairs = pair_images(rgb_records, thermal_records, thermal_offset=thermal_offset)
    annotation_index = load_annotations(annotations_root)
    calibration_index = load_calibrations(calibration_root)

    pairing_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for pair in pairs:
        overlay_path = ""
        side_by_side_path = ""
        mask_path = ""
        calibration = calibration_for_pair(calibration_index, pair)
        calibration_file = str(calibration.path) if calibration else ""
        calibration_method = calibration.method if calibration else "resize_only"
        calibration_point_count = calibration.point_count if calibration else ""
        if pair.thermal:
            overlay_path, side_by_side_path, mask_path = write_qa_artifacts(
                pair.rgb,
                pair.thermal,
                qa_root,
                mask_root,
                threshold_percentile,
                calibration,
            )
            annotations = annotation_index.get(normalize_name(pair.rgb.path.name)) or annotation_index.get(normalize_name(pair.rgb.path.stem), [])
            extracted_rows = extract_feature_rows(
                pair=pair,
                annotations=annotations,
                threshold_percentile=threshold_percentile,
                overlay_path=overlay_path,
                calibration=calibration,
            )
            feature_rows.extend(extracted_rows)
            if annotations:
                write_annotation_review_artifacts(
                    rgb_path=pair.rgb.path,
                    overlay_path=Path(overlay_path),
                    annotations=annotations,
                    feature_rows=extracted_rows,
                    output_root=output_root / "annotation_review",
                )

        pairing_rows.append(
            {
                "rgb_filename": pair.rgb.path.name,
                "thermal_filename": pair.thermal.path.name if pair.thermal else "",
                "pairing_method": pair.method,
                "confidence_flag": pair.confidence,
                "rgb_timestamp": pair.rgb.timestamp or "",
                "thermal_timestamp": pair.thermal.timestamp if pair.thermal and pair.thermal.timestamp else "",
                "timestamp_delta_seconds": format_optional_float(pair.timestamp_delta_seconds),
                "gps_delta_meters": format_optional_float(pair.gps_delta_meters),
                "rgb_width": pair.rgb.width,
                "rgb_height": pair.rgb.height,
                "thermal_width": pair.thermal.width if pair.thermal else "",
                "thermal_height": pair.thermal.height if pair.thermal else "",
                "qa_overlay_path": overlay_path,
                "qa_side_by_side_path": side_by_side_path,
                "thermal_mask_path": mask_path,
                "calibration_file": calibration_file,
                "calibration_method": calibration_method,
                "calibration_point_count": calibration_point_count,
            }
        )

    write_csv(output_root / "pairing_report.csv", pairing_rows, PAIRING_FIELDS)
    write_csv(output_root / "seal_thermal_features.csv", feature_rows, FEATURE_FIELDS)
    summary_rows = summarize_label_features(feature_rows)
    write_csv(output_root / "dead_alive_summary.csv", summary_rows, SUMMARY_FIELDS)
    classification_rows = build_classification_report(feature_rows)
    write_csv(output_root / "classification_report.csv", classification_rows, CLASSIFICATION_FIELDS)
    write_readiness_report(output_root / "classification_readiness.txt", summary_rows, min_label_count)
    print(f"Scanned {len(records)} images: {len(rgb_records)} RGB and {len(thermal_records)} thermal.")
    print(f"Wrote {output_root / 'pairing_report.csv'}")
    print(f"Wrote {output_root / 'seal_thermal_features.csv'}")
    print(f"Wrote {output_root / 'dead_alive_summary.csv'}")
    print(f"Wrote {output_root / 'classification_report.csv'}")
    print(f"Wrote {output_root / 'classification_readiness.txt'}")
    print(f"Wrote QA images under {qa_root}")


def scan_images(input_root: Path) -> list[ImageRecord]:
    roots: list[Path]
    rgb_dir = input_root / "rgb"
    thermal_dir = input_root / "thermal"
    if rgb_dir.exists() or thermal_dir.exists():
        roots = [p for p in [rgb_dir, thermal_dir] if p.exists()]
    else:
        roots = [input_root]
    paths = [
        p
        for root in roots
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not is_ignored_scan_path(p, root)
    ]
    return [read_image_record(path, input_root) for path in paths]


def is_ignored_scan_path(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts[:-1]
    except ValueError:
        return False
    return any(part.lower() in IGNORED_SCAN_DIRS for part in relative_parts)


def read_image_record(path: Path, input_root: Path) -> ImageRecord:
    with Image.open(path) as image:
        width, height = image.size
        exif = image.getexif()
        timestamp = read_timestamp(exif)
        gps_lat, gps_lon = read_gps(exif)
    return ImageRecord(
        path=path,
        number=parse_dji_number(path.name),
        is_thermal=is_thermal_path(path, input_root),
        width=width,
        height=height,
        timestamp=timestamp,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
    )


def is_thermal_path(path: Path, input_root: Path) -> bool:
    parts = {part.lower() for part in path.relative_to(input_root).parts[:-1]} if path.is_relative_to(input_root) else set()
    stem = path.stem.lower()
    return "thermal" in parts or stem.endswith("_r")


def parse_dji_number(name: str) -> int | None:
    match = DJI_NUMBER_RE.search(name)
    return int(match.group(1)) if match else None


def pair_images(rgb_records: list[ImageRecord], thermal_records: list[ImageRecord], thermal_offset: int) -> list[PairRecord]:
    thermal_by_number = {record.number: record for record in thermal_records if record.number is not None}
    pairs: list[PairRecord] = []
    for rgb in rgb_records:
        thermal = None
        method = "unmatched"
        confidence = "missing_thermal"
        if rgb.number is not None:
            expected = rgb.number + thermal_offset
            thermal = thermal_by_number.get(expected)
            if thermal:
                method = f"filename_offset_{thermal_offset:+d}"
                confidence = metadata_confidence(rgb, thermal)
            elif rgb.number in thermal_by_number:
                thermal = thermal_by_number[rgb.number]
                method = "filename_same_number"
                confidence = metadata_confidence(rgb, thermal)
        pairs.append(
            PairRecord(
                rgb=rgb,
                thermal=thermal,
                method=method,
                confidence=confidence,
                timestamp_delta_seconds=timestamp_delta_seconds(rgb.timestamp, thermal.timestamp) if thermal else None,
                gps_delta_meters=gps_delta_meters(rgb, thermal) if thermal else None,
            )
        )
    return pairs


def metadata_confidence(rgb: ImageRecord, thermal: ImageRecord) -> str:
    seconds = timestamp_delta_seconds(rgb.timestamp, thermal.timestamp)
    meters = gps_delta_meters(rgb, thermal)
    if seconds is not None and seconds > 10:
        return "review_timestamp"
    if meters is not None and meters > 30:
        return "review_gps"
    if seconds is None and meters is None:
        return "filename_only"
    return "metadata_ok"


def write_qa_artifacts(
    rgb_record: ImageRecord,
    thermal_record: ImageRecord,
    qa_root: Path,
    mask_root: Path,
    threshold_percentile: float,
    calibration: CalibrationRecord | None = None,
) -> tuple[str, str, str]:
    rgb = cv2.imread(str(rgb_record.path), cv2.IMREAD_COLOR)
    thermal = cv2.imread(str(thermal_record.path), cv2.IMREAD_GRAYSCALE)
    if rgb is None:
        raise ValueError(f"Could not read RGB image: {rgb_record.path}")
    if thermal is None:
        raise ValueError(f"Could not read thermal image: {thermal_record.path}")

    if calibration:
        thermal_for_overlay = warp_thermal_to_rgb(thermal, rgb.shape[1], rgb.shape[0], calibration)
    else:
        thermal_for_overlay = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    thermal_norm = normalize_uint8(thermal_for_overlay)
    heatmap = cv2.applyColorMap(thermal_norm, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(rgb, 0.58, heatmap, 0.42, 0)

    rgb_preview = resize_to_height(rgb, 720)
    heatmap_preview = resize_to_height(heatmap, 720)
    side_by_side = np.hstack([rgb_preview, heatmap_preview])

    mask = make_binary_mask(thermal, threshold_percentile)

    suffix = "_calibrated" if calibration else ""
    base = f"{rgb_record.path.stem}__{thermal_record.path.stem}{suffix}"
    overlay_path = qa_root / f"{base}_overlay.jpg"
    side_path = qa_root / f"{base}_side_by_side.jpg"
    mask_path = mask_root / f"{thermal_record.path.stem}_mask.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(side_path), side_by_side)
    cv2.imwrite(str(mask_path), mask)
    return str(overlay_path), str(side_path), str(mask_path)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    low, high = np.percentile(image_float, [1, 99])
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    clipped = np.clip((image_float - low) / (high - low), 0, 1)
    return (clipped * 255).astype(np.uint8)


def make_binary_mask(thermal_gray: np.ndarray, threshold_percentile: float) -> np.ndarray:
    threshold = np.percentile(thermal_gray, threshold_percentile)
    return np.where(thermal_gray >= threshold, 255, 0).astype(np.uint8)


def load_annotations(annotations_root: Path) -> dict[str, list[dict[str, Any]]]:
    if not annotations_root.exists():
        return {}
    index: dict[str, list[dict[str, Any]]] = {}
    for path in annotations_root.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        image_name = data.get("image") or data.get("rgb_filename") or path.stem
        seals = data.get("seals", [])
        if not isinstance(seals, list):
            raise ValueError(f"Annotation file has non-list seals field: {path}")
        for seal in seals:
            validate_annotation(seal, path)
        normalized_name = normalize_name(str(image_name))
        index.setdefault(normalized_name, []).extend(seals)
        index.setdefault(normalize_name(Path(str(image_name)).stem), []).extend(seals)
    return index


def load_calibrations(calibration_root: Path) -> dict[str, CalibrationRecord]:
    if not calibration_root.exists():
        return {}
    index: dict[str, CalibrationRecord] = {}
    for path in calibration_root.glob("*.json"):
        calibration = load_calibration(path)
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        keys = calibration_keys(data, path)
        for key in keys:
            index[normalize_name(key)] = calibration
    return index


def load_calibration(path: Path) -> CalibrationRecord:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rgb_points, thermal_points = calibration_points(data, path)
    if len(rgb_points) < 4:
        raise ValueError(f"Calibration needs at least 4 RGB/thermal point pairs: {path}")
    transform, inliers = cv2.findHomography(
        np.array(rgb_points, dtype=np.float32),
        np.array(thermal_points, dtype=np.float32),
        method=0,
    )
    if transform is None:
        raise ValueError(f"Could not estimate homography from calibration points: {path}")
    inlier_count = int(np.count_nonzero(inliers)) if inliers is not None else len(rgb_points)
    return CalibrationRecord(
        path=path,
        transform_rgb_to_thermal=transform.astype(np.float64),
        point_count=len(rgb_points),
        method=f"homography_rgb_to_thermal_{inlier_count}_inliers",
    )


def calibration_points(data: dict[str, Any], path: Path) -> tuple[list[list[float]], list[list[float]]]:
    if "points" in data:
        rgb_points = []
        thermal_points = []
        for index, item in enumerate(data["points"], start=1):
            if not isinstance(item, dict) or "rgb" not in item or "thermal" not in item:
                raise ValueError(f"Calibration point {index} needs rgb and thermal entries in {path}")
            rgb_points.append(point_pair(item["rgb"], path))
            thermal_points.append(point_pair(item["thermal"], path))
        return rgb_points, thermal_points

    if "rgb_points" in data and "thermal_points" in data:
        rgb_points = [point_pair(item, path) for item in data["rgb_points"]]
        thermal_points = [point_pair(item, path) for item in data["thermal_points"]]
        if len(rgb_points) != len(thermal_points):
            raise ValueError(f"rgb_points and thermal_points lengths differ in {path}")
        return rgb_points, thermal_points

    raise ValueError(f"Calibration file needs points or rgb_points/thermal_points: {path}")


def point_pair(value: Any, path: Path) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Calibration point must be [x, y] in {path}: {value}")
    return [float(value[0]), float(value[1])]


def calibration_keys(data: dict[str, Any], path: Path) -> list[str]:
    keys = [path.stem]
    for field in ["rgb_image", "rgb_filename", "image"]:
        value = data.get(field)
        if value:
            keys.extend([str(value), Path(str(value)).stem])
    if path.stem.lower() == "default":
        keys.append("default")
    return keys


def calibration_for_pair(calibration_index: dict[str, CalibrationRecord], pair: PairRecord) -> CalibrationRecord | None:
    candidates = [
        pair.rgb.path.name,
        pair.rgb.path.stem,
        "default",
    ]
    for candidate in candidates:
        calibration = calibration_index.get(normalize_name(candidate))
        if calibration:
            return calibration
    return None


def validate_annotation(annotation: dict[str, Any], path: Path) -> None:
    if not isinstance(annotation, dict):
        raise ValueError(f"Annotation entry must be an object in {path}")
    if "bbox" not in annotation and "polygon" not in annotation:
        raise ValueError(f"Annotation needs bbox or polygon in {path}: {annotation}")
    label = normalize_label(annotation.get("label"))
    if label and label not in VALID_LABELS:
        raise ValueError(f"Invalid label {label!r} in {path}. Use dead, alive, or unknown.")


def extract_feature_rows(
    pair: PairRecord,
    annotations: list[dict[str, Any]],
    threshold_percentile: float,
    overlay_path: str,
    calibration: CalibrationRecord | None,
) -> list[dict[str, Any]]:
    if not pair.thermal:
        return []
    thermal = cv2.imread(str(pair.thermal.path), cv2.IMREAD_GRAYSCALE)
    if thermal is None:
        raise ValueError(f"Could not read thermal image: {pair.thermal.path}")
    if calibration:
        thermal_measurement = thermal
        measure_width = pair.thermal.width
        measure_height = pair.thermal.height
    else:
        thermal_measurement = cv2.resize(thermal, (pair.rgb.width, pair.rgb.height), interpolation=cv2.INTER_LINEAR)
        measure_width = pair.rgb.width
        measure_height = pair.rgb.height
    binary = make_binary_mask(thermal_measurement, threshold_percentile)
    rows: list[dict[str, Any]] = []
    for index, annotation in enumerate(annotations, start=1):
        measurement_annotation = transform_annotation(annotation, calibration) if calibration else annotation
        mask = annotation_mask(measurement_annotation, measure_width, measure_height)
        if mask.sum() == 0:
            continue
        bbox_x, bbox_y, bbox_w, bbox_h = annotation_polygon_points(annotation)
        thermal_bbox_x, thermal_bbox_y, thermal_bbox_w, thermal_bbox_h = annotation_polygon_points(measurement_annotation)
        seal_pixels = thermal_measurement[mask > 0]
        binary_pixels = binary[mask > 0]
        background_mask = background_ring(mask, measure_width, measure_height)
        background_pixels = thermal_measurement[background_mask > 0]
        background_mean = float(np.mean(background_pixels)) if background_pixels.size else math.nan
        mean_intensity = float(np.mean(seal_pixels))
        median_intensity = float(np.median(seal_pixels))
        hot_pixel_fraction = float(np.count_nonzero(binary_pixels) / seal_pixels.size)
        contrast = math.nan if math.isnan(background_mean) else mean_intensity - background_mean
        prediction = predict_dead_alive(
            thermal_mean=mean_intensity,
            thermal_median=median_intensity,
            hot_pixel_fraction=hot_pixel_fraction,
            contrast_vs_background=contrast,
        )
        rows.append(
            {
                "rgb_filename": pair.rgb.path.name,
                "thermal_filename": pair.thermal.path.name,
                "seal_id": annotation.get("id", f"seal_{index}"),
                "known_label": normalize_label(annotation.get("label")),
                "review_flag": annotation.get("review_flag", ""),
                "detection_source": annotation.get("detection_source", ""),
                "annotation_type": "polygon" if "polygon" in annotation else "bbox",
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "thermal_bbox_x": thermal_bbox_x,
                "thermal_bbox_y": thermal_bbox_y,
                "thermal_bbox_w": thermal_bbox_w,
                "thermal_bbox_h": thermal_bbox_h,
                "area_pixels": int(seal_pixels.size),
                "thermal_measurement_space": "raw_thermal" if calibration else "thermal_resized_to_rgb",
                "calibration_file": str(calibration.path) if calibration else "",
                "calibration_method": calibration.method if calibration else "resize_only",
                "thermal_mean": round(mean_intensity, 4),
                "thermal_min": int(np.min(seal_pixels)),
                "thermal_max": int(np.max(seal_pixels)),
                "thermal_median": round(median_intensity, 4),
                "thermal_binary_hot_pixels": int(np.count_nonzero(binary_pixels)),
                "thermal_binary_hot_fraction": round(hot_pixel_fraction, 6),
                "background_mean": "" if math.isnan(background_mean) else round(background_mean, 4),
                "contrast_vs_background": "" if math.isnan(contrast) else round(contrast, 4),
                "predicted_label": prediction["predicted_label"],
                "prediction_confidence": prediction["confidence"],
                "prediction_reason": prediction["reason"],
                "qa_overlay_path": overlay_path,
            }
        )
    return rows


def predict_dead_alive(
    thermal_mean: float,
    thermal_median: float,
    hot_pixel_fraction: float,
    contrast_vs_background: float,
) -> dict[str, str]:
    if math.isnan(contrast_vs_background):
        return {
            "predicted_label": "uncertain",
            "confidence": "low",
            "reason": "no nearby background pixels were available for contrast comparison",
        }

    if contrast_vs_background >= 8 and hot_pixel_fraction >= 0.45:
        confidence = "high" if contrast_vs_background >= 16 and hot_pixel_fraction >= 0.65 else "medium"
        return {
            "predicted_label": "alive",
            "confidence": confidence,
            "reason": (
                f"seal polygon is warmer than nearby background "
                f"(mean={thermal_mean:.2f}, median={thermal_median:.2f}, contrast={contrast_vs_background:.2f}, hot_fraction={hot_pixel_fraction:.2f})"
            ),
        }

    if contrast_vs_background <= -2 and hot_pixel_fraction <= 0.35:
        confidence = "high" if contrast_vs_background <= -8 and hot_pixel_fraction <= 0.20 else "medium"
        return {
            "predicted_label": "dead",
            "confidence": confidence,
            "reason": (
                f"seal polygon is cooler than nearby background "
                f"(mean={thermal_mean:.2f}, median={thermal_median:.2f}, contrast={contrast_vs_background:.2f}, hot_fraction={hot_pixel_fraction:.2f})"
            ),
        }

    return {
        "predicted_label": "uncertain",
        "confidence": "low",
        "reason": (
            f"thermal signal is mixed or near background "
            f"(mean={thermal_mean:.2f}, median={thermal_median:.2f}, contrast={contrast_vs_background:.2f}, hot_fraction={hot_pixel_fraction:.2f})"
        ),
    }


def warp_thermal_to_rgb(thermal: np.ndarray, rgb_width: int, rgb_height: int, calibration: CalibrationRecord) -> np.ndarray:
    thermal_to_rgb = np.linalg.inv(calibration.transform_rgb_to_thermal)
    return cv2.warpPerspective(
        thermal,
        thermal_to_rgb,
        (rgb_width, rgb_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def transform_annotation(annotation: dict[str, Any], calibration: CalibrationRecord) -> dict[str, Any]:
    points = annotation_polygon_points(annotation)
    transformed = cv2.perspectiveTransform(
        np.array([points], dtype=np.float32),
        calibration.transform_rgb_to_thermal.astype(np.float32),
    )[0]
    result = dict(annotation)
    result.pop("bbox", None)
    result["polygon"] = [[float(x), float(y)] for x, y in transformed]
    return result


def annotation_polygon_points(annotation: dict[str, Any]) -> np.ndarray:
    if "polygon" in annotation:
        return np.array(annotation["polygon"], dtype=np.float32)
    x, y, w, h = [float(v) for v in annotation["bbox"]]
    return np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )


def annotation_bbox(annotation: dict[str, Any]) -> tuple[int, int, int, int]:
    if "bbox" in annotation:
        x, y, w, h = [int(round(float(v))) for v in annotation["bbox"]]
        return x, y, w, h
    points = np.array(annotation["polygon"], dtype=np.float32)
    x0 = int(np.floor(np.min(points[:, 0])))
    y0 = int(np.floor(np.min(points[:, 1])))
    x1 = int(np.ceil(np.max(points[:, 0])))
    y1 = int(np.ceil(np.max(points[:, 1])))
    return x0, y0, x1 - x0, y1 - y0


def write_annotation_review_artifacts(
    rgb_path: Path,
    overlay_path: Path,
    annotations: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    features_by_id = {str(row.get("seal_id", "")): row for row in feature_rows}
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(f"Could not read RGB image: {rgb_path}")
    overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR) if overlay_path.exists() else None

    rgb_review = draw_annotations(rgb, annotations, features_by_id)
    rgb_review_path = output_root / f"{rgb_path.stem}_rgb_annotations.jpg"
    cv2.imwrite(str(rgb_review_path), rgb_review)
    write_review_tiles(rgb_review, output_root / "tiles", f"{rgb_path.stem}_rgb_annotations")
    if overlay is not None:
        overlay_review = draw_annotations(overlay, annotations, features_by_id)
        overlay_review_path = output_root / f"{rgb_path.stem}_overlay_annotations.jpg"
        cv2.imwrite(str(overlay_review_path), overlay_review)
        write_review_tiles(overlay_review, output_root / "tiles", f"{rgb_path.stem}_overlay_annotations")


def draw_annotations(
    image: np.ndarray,
    annotations: list[dict[str, Any]],
    features_by_id: dict[str, dict[str, Any]],
) -> np.ndarray:
    result = image.copy()
    for index, annotation in enumerate(annotations, start=1):
        seal_id = str(annotation.get("id", f"seal_{index}"))
        feature = features_by_id.get(seal_id, {})
        predicted = str(feature.get("predicted_label", "not_run"))
        confidence = str(feature.get("prediction_confidence", ""))
        known = normalize_label(annotation.get("label")) or "unlabeled"
        color = prediction_color(predicted)
        draw_annotation_shape(result, annotation, color)
        x, y = annotation_label_anchor(annotation)
        label = f"{compact_seal_id(seal_id)} {prediction_code(predicted)}"
        draw_readable_label(result, label, x, y, color, scale=0.65)
    return result


def draw_annotation_shape(image: np.ndarray, annotation: dict[str, Any], color: tuple[int, int, int]) -> None:
    if "polygon" in annotation:
        points = np.array(annotation["polygon"], dtype=np.int32)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=4)
    elif "bbox" in annotation:
        x, y, w, h = [int(round(float(v))) for v in annotation["bbox"]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)


def annotation_label_anchor(annotation: dict[str, Any]) -> tuple[int, int]:
    if "bbox" in annotation:
        x, y = annotation["bbox"][:2]
        return int(round(float(x))), max(24, int(round(float(y))) - 10)
    points = np.array(annotation["polygon"], dtype=np.float32)
    return int(np.min(points[:, 0])), max(24, int(np.min(points[:, 1])) - 10)


def draw_readable_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.8) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(x, image.shape[1] - width - 4))
    y = max(height + 4, min(y, image.shape[0] - baseline - 4))
    cv2.rectangle(image, (x, y - height - 6), (x + width + 6, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 3, y), font, scale, color, thickness, cv2.LINE_AA)


def compact_seal_id(seal_id: str) -> str:
    match = re.search(r"(\d+)$", seal_id)
    return match.group(1) if match else seal_id


def prediction_code(predicted_label: str) -> str:
    if predicted_label == "alive":
        return "A"
    if predicted_label == "dead":
        return "D"
    if predicted_label == "uncertain":
        return "U"
    return "?"


def prediction_color(predicted_label: str) -> tuple[int, int, int]:
    if predicted_label == "alive":
        return (60, 220, 60)
    if predicted_label == "dead":
        return (60, 60, 255)
    if predicted_label == "uncertain":
        return (0, 220, 255)
    return (255, 180, 60)


def annotation_mask(annotation: dict[str, Any], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if "polygon" in annotation:
        points = np.array(annotation["polygon"], dtype=np.float32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    elif "bbox" in annotation:
        x, y, w, h = [int(round(float(v))) for v in annotation["bbox"]]
        x0 = max(0, min(width, x))
        y0 = max(0, min(height, y))
        x1 = max(0, min(width, x + w))
        y1 = max(0, min(height, y + h))
        mask[y0:y1, x0:x1] = 255
    else:
        raise ValueError(f"Annotation needs bbox or polygon: {annotation}")
    return mask


def background_ring(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    kernel_size = max(9, int(round(min(width, height) * 0.015)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return cv2.subtract(dilated, mask)


PAIRING_FIELDS = [
    "rgb_filename",
    "thermal_filename",
    "pairing_method",
    "confidence_flag",
    "rgb_timestamp",
    "thermal_timestamp",
    "timestamp_delta_seconds",
    "gps_delta_meters",
    "rgb_width",
    "rgb_height",
    "thermal_width",
    "thermal_height",
    "qa_overlay_path",
    "qa_side_by_side_path",
    "thermal_mask_path",
    "calibration_file",
    "calibration_method",
    "calibration_point_count",
]

FEATURE_FIELDS = [
    "rgb_filename",
    "thermal_filename",
    "seal_id",
    "known_label",
    "review_flag",
    "detection_source",
    "annotation_type",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "thermal_bbox_x",
    "thermal_bbox_y",
    "thermal_bbox_w",
    "thermal_bbox_h",
    "area_pixels",
    "thermal_measurement_space",
    "calibration_file",
    "calibration_method",
    "thermal_mean",
    "thermal_min",
    "thermal_max",
    "thermal_median",
    "thermal_binary_hot_pixels",
    "thermal_binary_hot_fraction",
    "background_mean",
    "contrast_vs_background",
    "predicted_label",
    "prediction_confidence",
    "prediction_reason",
    "qa_overlay_path",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_FIELDS = [
    "known_label",
    "count",
    "thermal_mean_avg",
    "thermal_min_avg",
    "thermal_max_avg",
    "thermal_median_avg",
    "thermal_binary_hot_fraction_avg",
    "contrast_vs_background_avg",
]


def summarize_label_features(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in feature_rows:
        label = normalize_label(row.get("known_label"))
        if label in {"dead", "alive"}:
            grouped.setdefault(label, []).append(row)

    summary: list[dict[str, Any]] = []
    for label in ["dead", "alive"]:
        rows = grouped.get(label, [])
        summary.append(
            {
                "known_label": label,
                "count": len(rows),
                "thermal_mean_avg": average_field(rows, "thermal_mean"),
                "thermal_min_avg": average_field(rows, "thermal_min"),
                "thermal_max_avg": average_field(rows, "thermal_max"),
                "thermal_median_avg": average_field(rows, "thermal_median"),
                "thermal_binary_hot_fraction_avg": average_field(rows, "thermal_binary_hot_fraction"),
                "contrast_vs_background_avg": average_field(rows, "contrast_vs_background"),
            }
        )
    return summary


CLASSIFICATION_FIELDS = [
    "rgb_filename",
    "thermal_filename",
    "seal_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "known_label",
    "predicted_label",
    "prediction_confidence",
    "prediction_matches_known",
    "prediction_reason",
]


def build_classification_report(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in feature_rows:
        known_label = normalize_label(row.get("known_label"))
        predicted_label = str(row.get("predicted_label", ""))
        if known_label in {"dead", "alive"}:
            matches_known = str(known_label == predicted_label).lower()
        else:
            matches_known = ""
        rows.append(
            {
                "rgb_filename": row.get("rgb_filename", ""),
                "thermal_filename": row.get("thermal_filename", ""),
                "seal_id": row.get("seal_id", ""),
                "bbox_x": row.get("bbox_x", ""),
                "bbox_y": row.get("bbox_y", ""),
                "bbox_w": row.get("bbox_w", ""),
                "bbox_h": row.get("bbox_h", ""),
                "known_label": known_label,
                "predicted_label": predicted_label,
                "prediction_confidence": row.get("prediction_confidence", ""),
                "prediction_matches_known": matches_known,
                "prediction_reason": row.get("prediction_reason", ""),
            }
        )
    return rows


def average_field(rows: list[dict[str, Any]], field: str) -> str:
    values: list[float] = []
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


def write_readiness_report(path: Path, summary_rows: list[dict[str, Any]], min_label_count: int) -> None:
    counts = {row["known_label"]: int(row["count"]) for row in summary_rows}
    dead_count = counts.get("dead", 0)
    alive_count = counts.get("alive", 0)
    ready = dead_count >= min_label_count and alive_count >= min_label_count
    status = "ready_for_classifier_review" if ready else "needs_more_verified_labels"
    lines = [
        f"status: {status}",
        f"dead_labels: {dead_count}",
        f"alive_labels: {alive_count}",
        f"minimum_required_per_class: {min_label_count}",
        "",
        "V1 produces first-pass dead/alive predictions for annotated seals.",
        "Treat predictions as review targets until enough verified dead/alive labels exist.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_timestamp(exif: Image.Exif) -> str | None:
    for tag_name in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
        tag_id = next((key for key, value in ExifTags.TAGS.items() if value == tag_name), None)
        if tag_id is not None and tag_id in exif:
            value = str(exif.get(tag_id))
            try:
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").isoformat()
            except ValueError:
                return value
    return None


def read_gps(exif: Image.Exif) -> tuple[float | None, float | None]:
    gps_tag = next((key for key, value in ExifTags.TAGS.items() if value == "GPSInfo"), None)
    if gps_tag is None or gps_tag not in exif:
        return None, None
    gps_raw = exif.get_ifd(gps_tag)
    gps = {ExifTags.GPSTAGS.get(key, key): value for key, value in gps_raw.items()}
    lat = gps_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
    lon = gps_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
    return lat, lon


def gps_decimal(value: Any, ref: Any) -> float | None:
    if not value or not ref:
        return None
    degrees, minutes, seconds = [float(part) for part in value]
    result = degrees + minutes / 60 + seconds / 3600
    if str(ref).upper() in {"S", "W"}:
        result *= -1
    return result


def timestamp_delta_seconds(first: str | None, second: str | None) -> float | None:
    if not first or not second:
        return None
    try:
        first_dt = datetime.fromisoformat(first)
        second_dt = datetime.fromisoformat(second)
    except ValueError:
        return None
    return abs((first_dt - second_dt).total_seconds())


def gps_delta_meters(first: ImageRecord, second: ImageRecord) -> float | None:
    if None in {first.gps_lat, first.gps_lon, second.gps_lat, second.gps_lon}:
        return None
    return haversine_meters(first.gps_lat, first.gps_lon, second.gps_lat, second.gps_lon)


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def normalize_name(value: str) -> str:
    return value.lower().replace("\\", "/")


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    label = str(value).strip().lower()
    if label in VALID_LABELS:
        return label
    return label


def format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
