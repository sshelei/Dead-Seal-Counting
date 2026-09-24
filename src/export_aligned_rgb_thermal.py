# this program exports aligned rgb and thermal images to a output folders
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cli import (
    CalibrationRecord,
    calibration_for_pair,
    load_calibrations,
    normalize_name,
    pair_images,
    scan_images,
)


MANIFEST_FIELDS = [
    "index",
    "rgb_output",
    "thermal_output",
    "rgb_original",
    "thermal_original",
    "pairing_method",
    "confidence_flag",
    "calibration_file",
    "calibration_method",
    "calibration_point_count",
    "output_width",
    "output_height",
    "crop_x",
    "crop_y",
    "crop_w",
    "crop_h",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export paired RGB and thermal images into a shared thermal-coordinate grid. "
            "The original images are only read; aligned PNGs are written to output/rgb and output/thermal."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data"),
        help="Input root. Supports either mixed DJI files or data/rgb and data/thermal subfolders.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("calibration"),
        help="Folder of RGB-to-thermal calibration JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aligned_rgb_thermal"),
        help="Output folder. The script creates rgb/ and thermal/ under this folder.",
    )
    parser.add_argument(
        "--thermal-offset",
        type=int,
        default=-1,
        help="Expected thermal number relative to RGB number. Use -1 for RGB DJI_0249 -> thermal DJI_0248_R.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of RGB frames processed.")
    parser.add_argument(
        "--target-rgb",
        action="append",
        default=[],
        help="Process only matching RGB filenames or stems. May be passed multiple times.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Keep the full thermal frame instead of cropping both outputs to the valid warped-RGB region.",
    )
    parser.add_argument(
        "--allow-uncalibrated-resize",
        action="store_true",
        help="For pairs without calibration, resize RGB to thermal size. This does not guarantee physical alignment.",
    )
    args = parser.parse_args()

    export_aligned_pairs(
        input_root=args.input,
        calibration_root=args.calibration,
        output_root=args.output,
        thermal_offset=args.thermal_offset,
        limit=args.limit,
        target_rgb=set(args.target_rgb),
        crop_to_valid_overlap=not args.no_crop,
        allow_uncalibrated_resize=args.allow_uncalibrated_resize,
    )
    return 0


def export_aligned_pairs(
    input_root: Path,
    calibration_root: Path,
    output_root: Path,
    thermal_offset: int,
    limit: int | None,
    target_rgb: set[str],
    crop_to_valid_overlap: bool,
    allow_uncalibrated_resize: bool,
) -> None:
    rgb_output_root = output_root / "rgb"
    thermal_output_root = output_root / "thermal"
    rgb_output_root.mkdir(parents=True, exist_ok=True)
    thermal_output_root.mkdir(parents=True, exist_ok=True)

    records = scan_images(input_root)
    rgb_records = sorted(
        (record for record in records if not record.is_thermal),
        key=lambda record: (record.number is None, record.number or 0, record.path.name),
    )
    thermal_records = [record for record in records if record.is_thermal]

    if target_rgb:
        wanted = {normalize_name(item) for item in target_rgb}
        rgb_records = [
            record
            for record in rgb_records
            if normalize_name(record.path.name) in wanted or normalize_name(record.path.stem) in wanted
        ]
    if limit is not None:
        rgb_records = rgb_records[:limit]

    calibration_index = load_calibrations(calibration_root)
    pairs = pair_images(rgb_records, thermal_records, thermal_offset=thermal_offset)

    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    exported_count = 0

    for pair in pairs:
        if pair.thermal is None:
            skipped_rows.append(skip_row(pair.rgb.path.name, "", "missing thermal pair"))
            continue

        calibration = calibration_for_pair(calibration_index, pair)
        if calibration is None and not allow_uncalibrated_resize:
            skipped_rows.append(skip_row(pair.rgb.path.name, pair.thermal.path.name, "missing calibration"))
            continue

        rgb = cv2.imread(str(pair.rgb.path), cv2.IMREAD_COLOR)
        thermal = cv2.imread(str(pair.thermal.path), cv2.IMREAD_UNCHANGED)
        if rgb is None:
            skipped_rows.append(skip_row(pair.rgb.path.name, pair.thermal.path.name, "could not read RGB"))
            continue
        if thermal is None:
            skipped_rows.append(skip_row(pair.rgb.path.name, pair.thermal.path.name, "could not read thermal"))
            continue

        aligned_rgb, aligned_thermal, crop = align_pair_to_thermal_grid(
            rgb=rgb,
            thermal=thermal,
            calibration=calibration,
            crop_to_valid_overlap=crop_to_valid_overlap,
        )
        rgb = str(pair.rgb.path)
        output_name = (rgb.split("/"))[-1]
        exported_count += 1

        rgb_output_path = rgb_output_root / output_name
        thermal_output_path = thermal_output_root / output_name

        cv2.imwrite(str(rgb_output_path), aligned_rgb)
        cv2.imwrite(str(thermal_output_path), aligned_thermal)

        height, width = aligned_thermal.shape[:2]
        manifest_rows.append(
            {
                "index": f"{exported_count:06d}",
                "rgb_output": str(rgb_output_path.relative_to(output_root)),
                "thermal_output": str(thermal_output_path.relative_to(output_root)),
                "rgb_original": str(pair.rgb.path),
                "thermal_original": str(pair.thermal.path),
                "pairing_method": pair.method,
                "confidence_flag": pair.confidence,
                "calibration_file": str(calibration.path) if calibration else "",
                "calibration_method": calibration.method if calibration else "resize_only_uncalibrated",
                "calibration_point_count": calibration.point_count if calibration else "",
                "output_width": width,
                "output_height": height,
                "crop_x": crop[0],
                "crop_y": crop[1],
                "crop_w": crop[2],
                "crop_h": crop[3],
            }
        )

    write_csv(output_root / "manifest.csv", manifest_rows, MANIFEST_FIELDS)
    write_json(output_root / "manifest.json", manifest_rows)
    if skipped_rows:
        write_csv(output_root / "skipped.csv", skipped_rows, ["rgb_original", "thermal_original", "reason"])

    print(f"Scanned {len(records)} images: {len(rgb_records)} RGB and {len(thermal_records)} thermal.")
    print(f"Exported {exported_count} aligned pairs under {output_root}.")
    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} pairs; see {output_root / 'skipped.csv'}.")


def align_pair_to_thermal_grid(
    rgb: np.ndarray,
    thermal: np.ndarray,
    calibration: CalibrationRecord | None,
    crop_to_valid_overlap: bool,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    thermal_height, thermal_width = thermal.shape[:2]
    if calibration is not None:
        aligned_rgb = cv2.warpPerspective(
            rgb,
            calibration.transform_rgb_to_thermal,
            (thermal_width, thermal_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_mask = warped_valid_rgb_mask(rgb.shape[1], rgb.shape[0], calibration, thermal_width, thermal_height)
    else:
        aligned_rgb = cv2.resize(rgb, (thermal_width, thermal_height), interpolation=cv2.INTER_AREA)
        valid_mask = np.full((thermal_height, thermal_width), 255, dtype=np.uint8)

    crop = valid_crop(valid_mask) if crop_to_valid_overlap else (0, 0, thermal_width, thermal_height)
    x, y, width, height = crop
    return (
        aligned_rgb[y : y + height, x : x + width],
        thermal[y : y + height, x : x + width],
        crop,
    )


def warped_valid_rgb_mask(
    rgb_width: int,
    rgb_height: int,
    calibration: CalibrationRecord,
    thermal_width: int,
    thermal_height: int,
) -> np.ndarray:
    source_mask = np.full((rgb_height, rgb_width), 255, dtype=np.uint8)
    return cv2.warpPerspective(
        source_mask,
        calibration.transform_rgb_to_thermal,
        (thermal_width, thermal_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def valid_crop(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(mask)
    if points is None:
        return (0, 0, mask.shape[1], mask.shape[0])
    x, y, width, height = cv2.boundingRect(points)
    return (int(x), int(y), int(width), int(height))


def skip_row(rgb_original: str, thermal_original: str, reason: str) -> dict[str, str]:
    return {
        "rgb_original": rgb_original,
        "thermal_original": thermal_original,
        "reason": reason,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
