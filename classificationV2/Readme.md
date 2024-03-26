# Training a Keypoint Detection Model

# Dataset
The dataset is described [here](https://github.com/datalab-mi/Basegun-ml/wiki/Classification-dataset)

## How to Train a YOLOV8 model on a Custom Dataset
The YOLOV8 documentation provides comprehensive and well-explained instructions for training the model on a custom dataset. You can find it [here](https://docs.ultralytics.com/modes/train/).

The Ultralytics library makes it easy to initiate the training of a model. It can be used in Python or directly in a terminal, as demonstrated [here](https://docs.ultralytics.com/tasks/pose/#train).

## How to use the trained model
To use the trained model, one can use the inference.py file that contains functions for loading the model and run it on an image. Using the Ultralytics library, there is no preprocessing require to convert the image as the library allow multiple sources of input.
