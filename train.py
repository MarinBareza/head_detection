from ultralytics import YOLO

model = YOLO("runs/detect/head_detection12/weights/best.pt")
model.to(device='cuda:0')

results = model.train(data=f"datasets/Head-detector-6/data.yaml", epochs=100, imgsz=640, batch=4, name="head_detection")

model.export(format="onnx")
