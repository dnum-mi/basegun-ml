# Measure Module

This module measure the length of the weapon and its barrel using a photo of a weapon and a card for reference. Those length are required in some cases to determine the legal class of the weapon.

# How to use it
The pretrained models and the datasets are available in the basegun resana.

## Librairies to install

- Ultralytics
- opencv-python
- onnxruntime

```
pip install ultralytics opencv-python onnxruntime
```


### Python code
```
from MeasureModule import load_models, get_lengths
import cv2

#Import and load models
model_card,model_weapon=load_models("./best_card.onnx","./best_keypoints.pt")


# Calculate lengths
GlobalLength, BarrelLength, ConfCard =get_lengths(cv2.imread(".imagePath.jpg"),model_card,model_weapon)

```

