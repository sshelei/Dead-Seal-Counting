from pathlib import Path
from ultralytics import YOLO
import cv2
import json
import numpy as np

model_path = "src/best.pt" 

image_path = "/home/leish/dead_seal_project/images/DJI_0031.jpg"

output_dir = Path("/home/leish/dead_seal_project/yolo_compare_outputs")
annotated_dir = output_dir / "annotated"
annotations_dir = output_dir / "annotations"
annotated_dir.mkdir(parents=True, exist_ok=True)
annotations_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(model_path)
results = model.predict(image_path, conf=0.1, rect=False, imgsz=1920, save=True)

for result in results:
    print(result.boxes.to_list())
    result.show() 
