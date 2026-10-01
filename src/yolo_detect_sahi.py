# This program uses sahi to slice up drone images and sends to downloaded model defined by best_1.pt files

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from pathlib import Path
import cv2
import json
import numpy as np

# best_1.pt file is downloaded bounding box model for "seal Computer Vision Model" by KSpicY
model_path = "model_weights/best_1.pt" 

images_dir = Path("/home/leish/dead_seal_project/images")
output_dir = Path("/home/leish/dead_seal_project/yolo_outputs_v3")
annotated_dir = output_dir / "annotated"
annotations_dir = output_dir / "annotations"
annotated_dir.mkdir(parents=True, exist_ok=True)
annotations_dir.mkdir(parents=True, exist_ok=True)

single_image_path = "/home/leish/dead_seal_project/images/DJI_0249.jpg"


# Run on your massive drone image!
#result = get_sliced_prediction(
    #single_image_path,
    #detection_model,
    #slice_height=640,
    #slice_width=640,
    #overlap_height_ratio=0.2,
    #overlap_width_ratio=0.2
#)

#result.export_visuals(export_dir="yolo_outputs_v2/")


def points_from_prediction(pred):
    points = pred.get("points") or pred.get("polygon")
    if points:
        parsed_points = []
        for point in points:
            if isinstance(point, dict):
                parsed_points.append([int(round(point["x"])), int(round(point["y"]))])
            else:
                parsed_points.append([int(round(point[0])), int(round(point[1]))])
        return parsed_points

    x_center = float(pred["x"])
    y_center = float(pred["y"])
    width = float(pred["width"])
    height = float(pred["height"])
    x1 = int(round(x_center - width / 2))
    y1 = int(round(y_center - height / 2))
    x2 = int(round(x_center + width / 2))
    y2 = int(round(y_center + height / 2))
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def bbox_from_points(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)

def draw_prediction(image, polygon, points, label, confidence):
    #x1, y1, x2, y2 = bbox_from_points(polygon)
    polygon_array = np.array(polygon, dtype=np.int32)
    cv2.polylines(image, [polygon_array], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.rectangle(image, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 3)
    text = f"{label} {confidence:.2f}"
    cv2.putText(image, text, (points[0], points[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    


def run_workflow(image_path, height, width):
    return get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=int(height/2),
        slice_width=int(width/2),
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3
    )


def annotate_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width, _ = image.shape
    result = run_workflow(image_path, height, width)
    json_output_seals = []
    #predictions = extract_predictions(result)
   
    print(f"{image_path.name}: found {len(result.object_prediction_list)} predictions")
    #print(f"{image_path}: found {len(predictions)} predictions")
    pred = result.object_prediction_list[0]
    print(pred)
    print(hasattr(pred, "mask"), pred.mask)
    for i, pred in enumerate(result.object_prediction_list):
        x1, y1, x2, y2 = [int(round(v)) for v in pred.bbox.to_xyxy()]
        confidence = float(pred.score.value)
        label = pred.category.name

        bbox = [x1, y1, x2 - x1, y2 - y1]
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        points = [x1, y1, x2, y2]
        draw_prediction(image, polygon, points, label, confidence)

        json_output_seals.append(
            {
                "id": f"seal_{len(json_output_seals) + 1:03d}",
                "label": "unknown",
                "confidence": confidence,
                "bbox": bbox,
                "polygon": polygon,
            }
        )

    output_image_path = annotated_dir / f"{image_path.stem}_annotated.jpg"
    #output_image_path = annotated_dir / f"{image_path}_annotated.jpg"
    cv2.imwrite(str(output_image_path), image)
    #output_json_path = annotations_dir / f"{image_path}.json"
    output_json_path = annotations_dir / f"{image_path.stem}.json"
    final_json = {
        "image": image_path.name,
        "seals": json_output_seals,
    }
    output_json_path.write_text(json.dumps(final_json, indent=4), encoding="utf-8")
    print(f"Wrote {output_image_path} and {output_json_path}")

def is_rgb_image(path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"} and not path.stem.lower().endswith("_r")


detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.2,
    device="cpu" # Or "cuda" if you have a local NVIDIA GPU
)

for image_path in sorted(images_dir.iterdir()):
    if is_rgb_image(image_path):
        annotate_image(image_path)