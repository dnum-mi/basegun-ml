# Installation
```
pip install .\Basegun-ml\MLpackages\basegunML\dist\basegunml-0.1.tar.gz
```
# Usage
## Classification
```Python
from basegunML.classification import get_typology, list_typologies
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
```Python
from basegunML.measure import get_lengths

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

# Tests
Tests are available for the classification task and the measure length task
```
pytest tests/test_classification.py 
pytest tests/test_measure.py
```
# Credits

<li>This project uses the [Ultralytics Library](https://github.com/ultralytics/ultralytics)
<li>The oriented bounding box detection is inspired from [this](https://github.com/hukaixuan19970627/yolov5_obb) YOLOV5 implementation
