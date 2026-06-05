"""Convert plan-style polygon annotations into COCO instance segmentation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


SEAL_CATEGORY_ID = 1


@dataclass
class CocoBuilder:
    images: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    categories: list[dict[str, Any]]
    next_image_id: int = 1
    next_annotation_id: int = 1

    def add_image(self, file_name: str, width: int, height: int) -> int:
        image_id = self.next_image_id
        self.next_image_id += 1
        self.images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )
        return image_id

    def add_annotation(
        self,
        image_id: int,
        segmentation: list[float],
        bbox: list[float],
        area: float,
    ) -> None:
        annotation_id = self.next_annotation_id
        self.next_annotation_id += 1
        self.annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": SEAL_CATEGORY_ID,
                "segmentation": [segmentation],
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directory containing plan-style JSON annotation files.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing RGB images referenced by the JSON files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the COCO instances JSON.",
    )
    return parser.parse_args()


def polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_bbox(points: list[list[float]]) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def flatten_polygon(points: list[list[float]]) -> list[float]:
    return [coordinate for point in points for coordinate in point]


def load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def build_coco(annotations_dir: Path, images_dir: Path) -> dict[str, Any]:
    builder = CocoBuilder(
        images=[],
        annotations=[],
        categories=[{"id": SEAL_CATEGORY_ID, "name": "seal"}],
    )

    for annotation_path in sorted(annotations_dir.glob("*.json")):
        payload = json.loads(annotation_path.read_text())
        image_name = payload["image"]
        image_path = images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for annotation: {image_path}")

        width, height = load_image_size(image_path)
        image_id = builder.add_image(image_name, width, height)

        for seal in payload.get("seals", []):
            polygon = seal.get("polygon")
            if not polygon or len(polygon) < 3:
                continue
            segmentation = flatten_polygon(polygon)
            bbox = polygon_bbox(polygon)
            area = polygon_area(polygon)
            builder.add_annotation(image_id, segmentation, bbox, area)

    return builder.to_dict()


def main() -> None:
    args = parse_args()
    coco = build_coco(args.annotations_dir, args.images_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(coco, indent=2))


if __name__ == "__main__":
    main()

