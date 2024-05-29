from ultralytics import YOLO
import os
from basegunML.utils import load_models


this_dir, this_filename = os.path.split(__file__)

model_classif = YOLO(os.path.join(this_dir,"YOLOClassifN.pt"))
warmup_classif = model_classif(os.path.join(this_dir, "warmup.jpg"))

model_card,model_keypoints= load_models(os.path.join(this_dir,"./modelCard.onnx"),os.path.join(this_dir,"./keypoints.pt"),os.path.join(this_dir, "warmup.jpg"))