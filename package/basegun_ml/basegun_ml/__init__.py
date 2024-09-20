import os

import pyiqa
import torch
from paddleocr import PaddleOCR
from ultralytics import YOLO

from basegun_ml.utils import load_models

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

this_dir, this_filename = os.path.split(__file__)

model_classif = YOLO(os.path.join(this_dir, "classification.pt"))
warmup_classif = model_classif(os.path.join(this_dir, "warmup.jpg"))

model_card, model_keypoints = load_models(
    os.path.join(this_dir, "./card_detection.onnx"),
    os.path.join(this_dir, "./keypoints.pt"),
    os.path.join(this_dir, "warmup.jpg"),
)
model_ocr = PaddleOCR(
    det_model_dir=os.path.join(this_dir, "PaddleModels/detection"),
    rec_model_dir=os.path.join(this_dir, "PaddleModels/recognition"),
    cls_model_dir=os.path.join(this_dir, "PaddleModels/classification"),
    use_angle_cls=True,
    show_log=False,
)
device = torch.device("cpu")
metric_iqa = pyiqa.create_metric(
    "cnniqa", device=device, pretrained_model_path=os.path.join(this_dir, "CNNIQA.pth")
)
