# Installation
```
pip install .\Basegun-ml\MLpackages\basegunML\dist\basegunml-0.1.tar.gz
```
# Usage
## Classification

```Python
from basegunML.classification import *

#Convert image to bytes
with open("test.jpg", "rb") as file:
    image_bytes = file.read()

#Get typology
typology,confidenceScore,confidenceLevel=get_typology(image_bytes)
```

## Measure length
```Python
from basegunML.measure import *

#Convert image to bytes
with open("test.jpg", "rb") as file:
    image_bytes = file.read()

#Get lengths
weaponLength,barrelLength,confidenceCard=get_lengths(image_bytes)
```


# Tests
Tests are available for the classification task and the measure length task
```
pytest tests/test_classification.py 
pytest tests/test_measure.py
```
