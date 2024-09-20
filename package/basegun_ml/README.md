# Project description
Basegun_ml is a deeplearning package for the basegun weapon recongition app.

# Installation
```
pip install basegun-ml
```

# Usage
## Classification
**Gun Mechanism Classification**: This feature categorizes an image into a list of families representing different firearm mechanisms. The classification is based on descriptive, objective criteria that are independent of legal classification.
```Python
from basegun_ml.classification import get_typology, list_typologies
#After the import the model is already warmed-up for faster inference

#Convert image to bytes
with open("test.jpg", "rb") as file:
    image_bytes = file.read()

#Prediction of the weapon typology
typology,confidence_score,confidence_level=get_typology(image_bytes)

#Obtain the list of the different typologies
list_typologies()

```
### Variables description
<li> <b>typology</b>: it corresponds to the weapon class predicted from the image. The list of typologies can be obtained from the function

<li> <b>confidence_score</b>: it corresponds to the confidence of the class prediction of the algorithm, the closer to 1 to more confident is the prediction

<li> <b>confidence_level</b>: there are 3 level of confidence defined. According to this performance level the basegun user will have more information.

## Measure length
**Measure Length Module**: Measuring the overall length of a firearm or its barrel length is crucial for its legal classification. In France, the classification of long guns depends on these measurements. This module measures these lengths using an image.

```Python
from basegun_ml.measure import get_lengths

#Convert image to bytes
with open("test.jpg", "rb") as file:
    image_bytes = file.read()

#Get lengths
weapon_length,barrel_length,confidence_card=get_lengths(image_bytes)
```
### Variables description
<li> <b>weapon_length</b>: it corresponds to the weapon overall length predicted from the image.

<li> <b>barrel_length</b>: it corresponds to the barrel length of the weapon predicted from the image.

<li> <b>confidence_card</b>: it corresponds to the confidence score for the card prediction. A card is used as a reference for the measure module

<li> If the gun is not detected, the exception <b>MissingGun</b> is raised

<li> If the card is not detected, the exception <b>MissingCard</b> is raised

## Alarm Model detection
**Alarm Gun Recognition**: An alarm gun is a type of blank gun recognized as an alarm by French legislation. These guns are considered impossible to modify to make them lethal. The associated algorithm detects alarm guns using markings on the weapon.
```Python
from basegun_ml.ocr import is_alarm_weapon
#After the import the model is already warmed-up for faster inference

#Convert image to bytes
with open("test.jpg", "rb") as file:
    image_bytes = file.read()

#Prediction of the weapon typology
alarm_model = is_alarm_weapon(image_bytes, quality_check=True )


```
### Variables description
<li> <b>alarm_model</b> if the gun is one of the alarm model it returns "Alarm_model". If the gun has the PAK marking then alarm_model returns "PAK" else it return "Not_alarm"

<li> <b>quality_check</b> specify if the quality analysis is run before the text detection

<li> If the image quality is too low, the exception <b>LowQuality</b> is raised

<li> If no text is detected, the exception <b>MissingText</b> is raised



# Tests
Tests are available for the classification task and the measure length task
```
pytest tests/test_classification.py 
pytest tests/test_measure.py
pytest tests/test_OCR.py
```
# Credits

- This project uses the [Ultralytics Library](https://github.com/ultralytics/ultralytics) 
- The oriented bounding box detection is inspired from [this YOLOV5 implementation](https://github.com/hukaixuan19970627/yolov5_obb) 
- The image quality analysis uses [Pyiqa](https://github.com/chaofengc/IQA-PyTorch)
- The OCR tasks are computed using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR?tab=readme-ov-file)
