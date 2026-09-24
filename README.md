cli.py has the workflow of 
1) Filtering and masking RGB images to detect seals
2) Using the detected seals to determine their average temperature in the thermal and classifying each seal as alive, dead, or unknown
---
In other words,
1) Seal Detection
2) Classifying as Alive, Dead, or Uncertain
---
Make sure your environment has python installed. If on Linux distribution (WSL, etc), set up a python virtual environment and install needed packages. Dataset used is shared Box folder Island thermal-20260502T194355Z-3-00. Download dataset and move images to data folder.  
To run cli.py:  
&emsp; cli.py has three subcommands: init (initializes the folders needed, can be skipped), detect (corresponds to Step 1), run (corresponds to Step 2)    
&emsp; `python src/cli.py init`  
&emsp; `python src/cli.py detect`[optional args]  
&emsp; &nbsp; Optional args are explained in more detail in the file  
&emsp;  `python src/cli.py run` [optional args]  
&emsp; &nbsp;     Note: To align the RGB and Thermal images, include `--calibration calibration`. See src/CALIBRATION.md for more info.  

To replace Step 1) Seal Detection (the detect step), different YOLO models were tested.
The best is WAID with manually labeled drone images in 640 x 640 tiles (defined by weights model_weights/best_8.pt)
WAID_detect_sahi.py is intended to use downloaded WAID model and sahi to slice up images and creates output folder with annotated/annotations subfolder. Feed the annotations subfolder path to `python src/cli.py run`

yolo_detect_server.py uses pre-trained YOLO model “seal Computer Vision Model” by KSpicY (https://universe.roboflow.com/kspicy/seal-ekfsj).  

Due to the model not having weights available to download, either would have to send requests to local server (under API credits limit) or download dataset and train locally as done in yolo_detect_sahi.py.  

To download dataset (see train_seal_model.ipynb for reference), 
```python
  pip install roboflow ultralytics  
  from roboflow import Roboflow  
  
  rf = Roboflow(api_key="YOUR_API_KEY")  
  project = rf.workspace("kspicy").project("seal-ekfsj")  
  model_dir = project.version(1).download("yolov8")
```
To train model,  
```python
  yolo train model=yolov8n.pt data=path/to/data.yaml epochs=25 imgsz=640
``` 
To run, use `python src/yolo_detect.py`  
&emsp; &nbsp;  Note: yolo_detect.py works better with folder of images, not really single image path  
&emsp; &nbsp;  Note: Need to change input and output paths
  
