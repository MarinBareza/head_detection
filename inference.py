import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/head_detection5/weights/best.pt")

results = model("test_sample.jpg")
for result in results:
    img = results[0].plot()
    cv2.imwrite('annotated.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
