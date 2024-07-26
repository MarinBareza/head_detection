from roboflow import Roboflow

rf = Roboflow(api_key="wjZeZJ7LL07LmZ6aMCBG")
project = rf.workspace("debilicar").project("head-detector-xmlxl")
version = project.version(6)
dataset = version.download("yolov9")
