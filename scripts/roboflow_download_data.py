import os
from roboflow import Roboflow

rf = Roboflow(api_key="HWo2Ck1iIr6i527KxZv4")
project = rf.workspace("farmpipeline-qbiuc").project("dockloading")
version = project.version(1)
dataset = version.download("yolov11") # yolov5

os.rename('Dockloading-1', 'datasets')

                