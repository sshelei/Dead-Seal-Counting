# this program takes in the drone RGB images (4000 x 3000) and tiles each into tiles

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
     
from PIL import Image
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

@dataclass
class Box:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile original drone images into slices")
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Input image directory",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Input YOLO label directory",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output dataset directory")
    parser.add_argument("--tile_size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2, help="Fractional overlap, e.g. 0.2")
    parser.add_argument("--min-visible", type=float, default=0.35, help="Minimum fraction of original box area visible inside tile")
    parser.add_argument("--include-empty", action="store_true", help="Write tiles with empty label files for hard negatives")
    return parser.parse_args()

def read_yolo_boxes (label_path, width, height) -> list[Box]:
    boxes: list[Box] = []
    with open(label_path, "r") as label_file:
        for line in label_file:
            parts = line.split()
            if len(parts) != 5:
                printf("Formatting of YOLO Label file is invalid")
                return boxes
            x_cen = float(parts[1]) * width
            y_cen = float(parts[2]) * height
            box_w = float(parts[3]) * width
            box_h = float(parts[4]) * height
            x1 = x_cen - (box_w / 2)
            x2 = x_cen + (box_w / 2)
            y1 = y_cen - (box_h / 2)
            y2 = y_cen + (box_h / 2)
            box = Box(parts[0], x1, y1, x2, y2)
            boxes.append(box)
    return boxes
            
def split_tiles (tile_size, stride, dimension) -> list[int]:
    if dimension <= tile_size:
        return [0]
    tiles = list(range(0, dimension - tile_size + 1, stride))
    final_tile = dimension - tile_size
    if tiles[-1] != final_tile:
        tiles.append(final_tile)
    return tiles

def clip_box_to_tile(box, tile_x, tile_y, tile_size, min_visible) -> str | None:
    ix1 = max(box.x1, tile_x)
    iy1 = max(box.y1, tile_y)
    ix2 = min(box.x2, tile_x + tile_size)
    iy2 = min(box.y2, tile_y + tile_size)

    if ix2 <= ix1 or iy2 <= iy1:
        return None
    original_area = max((box.x2 - box.x1) * (box.y2 - box.y1), 1.0)
    visible_area = (ix2 - ix1) * (iy2 - iy1)
    if visible_area / original_area < min_visible:
        return None
    tile_x1 = ix1 - tile_x
    tile_y1 = iy1 - tile_y
    tile_x2 = ix2 - tile_x
    tile_y2 = iy2 - tile_y

    xc = ((tile_x1 + tile_x2) / 2) / tile_size
    yc = ((tile_y1 + tile_y2) / 2) / tile_size
    bw = (tile_x2 - tile_x1) / tile_size
    bh = (tile_y2 - tile_y1) / tile_size
    return f"{box.cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"



def main() -> None:
    args = parse_args()
    tile_size = args.tile_size
    stride = max(1, round(tile_size * (1 - args.overlap)))
    
    out_images = args.out / "images"
    out_labels = args.out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    
    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    written = 0
    empty = 0
    
    for image_path in image_paths:
        label_path = args.labels / f"{image_path.stem}.txt"
        with Image.open(image_path) as img:
            width, height = img.size
            boxes = read_yolo_boxes(label_path, width, height)
            x_tiles = split_tiles(tile_size, stride, width)
            y_tiles = split_tiles(tile_size, stride, height)
            for x in x_tiles:
                for y in y_tiles:
                    tile_labels = []
                    for box in boxes:
                        label = clip_box_to_tile(box, x, y, tile_size, args.min_visible)
                        tile_labels.append(label)
                    if not tile_labels and not args.include_empty:
                            continue
                    stem = f"{image_path.stem}_x{x:04d}_y{y:04d}"
                    tile = img.crop((x, y, x + tile_size, y + tile_size))
                    tile.save(out_images / f"{stem}.jpg", quality=95)
                    label_file = out_labels / f"{stem}.txt"
                    with open(label_file, "w") as file:
                        for line in tile_labels:
                            if line is None:
                                file.write("")
                            else:
                                file.write(line + "\n")
                written += 1
                empty += int(not tile_labels)
    
    print(f"wrote {written} tiles to {args.out}")
    print(f"empty hard-negative tiles: {empty}")


if __name__ == "__main__":
    main()


