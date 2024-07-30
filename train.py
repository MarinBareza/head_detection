from ultralytics import YOLO

model = YOLO("runs/detect/head_detection15/weights/best.pt")
model.to(device='cuda:0')
results = model.train(data=f"datasets/Head-detector-8/data.yaml", epochs=50, imgsz=640, batch=4, name="head_detection")
model.export(format="onnx")
