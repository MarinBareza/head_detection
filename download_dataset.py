from roboflow import Roboflow

rf = Roboflow(api_key="wjZeZJ7LL07LmZ6aMCBG")
project = rf.workspace("trisha-then").project("head-detection-cctv")
version = project.version(5)
dataset = version.download("yolov9")
