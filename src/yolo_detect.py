import cv2
import json
import numpy as np
from pathlib import Path
import time

# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workspace
client = InferenceHTTPClient(
  api_url="https://serverless.roboflow.com",
  api_key="VTevmVsVleH9iHTYjg4t"
)
images_dir = Path("/home/leish/dead_seal_project/images")
output_dir = Path("/home/leish/dead_seal_project/yolo_outputs")
annotated_dir = output_dir / "annotated"
annotations_dir = output_dir / "annotations"
annotated_dir.mkdir(parents=True, exist_ok=True)
annotations_dir.mkdir(parents=True, exist_ok=True)

#single_image_path = "/home/leish/dead_seal_project/images/DJI_0249.jpg"

def extract_predictions(workflow_result):
    """Handle common Roboflow workflow response shapes."""
    payload = workflow_result[0] if isinstance(workflow_result, list) and workflow_result else workflow_result

    if isinstance(payload, dict):
        predictions = payload.get("predictions", [])
        if isinstance(predictions, dict):
            return predictions.get("predictions", [])
        if isinstance(predictions, list):
            return predictions

    return []


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


def is_rgb_image(path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"} and not path.stem.lower().endswith("_r")


def run_workflow(image_path):
    return client.run_workflow(
        workspace_name="sherry-lei",
        workflow_id="general-segmentation-api-4",
        images={"image": str(image_path)},
        parameters={"classes": "seal"},
        use_cache=True,
    )


def draw_prediction(image, polygon, label, confidence):
    x1, y1, x2, y2 = bbox_from_points(polygon)
    polygon_array = np.array(polygon, dtype=np.int32)
    cv2.polylines(image, [polygon_array], isClosed=True, color=(0, 255, 0), thickness=3)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    text = f"{label} {confidence:.2f}"
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return x1, y1, x2, y2


def annotate_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    result = run_workflow(image_path)
    predictions = extract_predictions(result)
    print(f"{image_path.name}: found {len(predictions)} predictions")
    #print(f"{image_path}: found {len(predictions)} predictions")
    json_output_seals = []
    for i, pred in enumerate(predictions):
        if not isinstance(pred, dict):
            print(f"Skipping non-dict prediction for {image_path.name}: {pred!r}")
            continue

        polygon = points_from_prediction(pred)
        confidence = float(pred.get("confidence", pred.get("score", 0.0)))
        label = pred.get("class", pred.get("class_name", "seal"))
        x1, y1, x2, y2 = draw_prediction(image, polygon, label, confidence)

        json_output_seals.append(
            {
                "id": f"seal_{len(json_output_seals) + 1:03d}",
                "label": "unknown",
                "confidence": confidence,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
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

#annotate_image(single_image_path)
for image_path in sorted(images_dir.iterdir()):
    if is_rgb_image(image_path):
        annotate_image(image_path)
        time.sleep(2)
