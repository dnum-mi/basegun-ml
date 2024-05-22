from ultralytics import YOLO
import os
from basegunML.utils import load_models


this_dir, this_filename = os.path.split(__file__)

modelClassif = YOLO(os.path.join(this_dir,"YOLOClassifN.pt"))
warmupClassif = modelClassif(os.path.join(this_dir, "warmup.jpg"))

modelCard,modelKeypoints= load_models(os.path.join(this_dir,"./modelCard.onnx"),os.path.join(this_dir,"./keypoints.pt"),os.path.join(this_dir, "warmup.jpg"))