from ultralytics import YOLO

model = YOLO("yolov9m.pt")

results = model.train(data=f"Head-Detection-(CCTV)-5/data.yaml", epochs=50, imgsz=640, batch=4, name="head_detection")

model.export(format="onnx")
