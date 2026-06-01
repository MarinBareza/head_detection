# Head Detection (CCTV)

A YOLO-based pipeline for detecting and counting heads (people) in CCTV footage.
The model is trained on a custom Roboflow dataset and applied to video to count
customers in the scene, including counts within defined zones such as the
entrance and the counter.

## Features

- Train a YOLO head-detection model on a custom dataset.
- Run inference on single images.
- Process video files to count people per frame, overlaying:
  - total customers in the scene,
  - customers within the entrance zone,
  - customers within the counter zone,
  - mean detection confidence.
- Utilities for preparing data from raw video (cutting clips, extracting frames).

## Requirements

- Python 3.8+
- A CUDA-capable GPU (the scripts use `cuda:0`)
- Dependencies:

```bash
pip install ultralytics opencv-python numpy roboflow
```

## Project structure

| File | Description |
| --- | --- |
| `dataset.py` | Downloads the training dataset from Roboflow. |
| `train.py` | Trains the YOLO model and exports it to ONNX. |
| `inference.py` | Runs detection on a single image and saves an annotated result. |
| `video_capture.py` | Runs detection on a video, counts people per zone, and writes an annotated output video. |
| `utils.py` | Helpers for cutting video and extracting frames, plus drawing utilities. |

## Usage

### 1. Download the dataset

```bash
python dataset.py
```

### 2. Train the model

```bash
python train.py
```

Trained weights are written under `runs/detect/<run_name>/weights/best.pt`.

### 3. Run inference on an image

```bash
python inference.py
```

Produces `annotated.jpg`.

### 4. Process a video

Set `video_path` in `video_capture.py` to your input video, then run:

```bash
python video_capture.py
```

The annotated result is saved to `output.mp4`.

## Notes

- The detection zones (`entrance_polygon`, `counter_polygon`) in `video_capture.py`
  are tuned to a specific camera view. Adjust the polygon coordinates to match
  your own footage.
- Update the model weight paths in the scripts to point at the run you want to use.
