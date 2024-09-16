# Training a Keypoint Detection Model

# Dataset
Check out our wiki section about the [measure dataset](https://github.com/datalab-mi/Basegun-ml/wiki/Measure-dataset).

## How to Train a YOLOV8 Pose on a Custom Dataset
The YOLOV8 documentation provides comprehensive and well-explained instructions for training the model on a custom dataset. You can find it [here](https://docs.ultralytics.com/modes/train/).

The Ultralytics library makes it easy to initiate the training of a model. It can be used in Python or directly in a terminal, as demonstrated [here](https://docs.ultralytics.com/tasks/pose/#train).

### Config File 
The .yaml config file is used to specify the dataset's location and format, the number of classes, and the number of keypoints to the model.

**Warning**: The config file must be in .yaml format, not .yml. Otherwise, the model will return an error.

```markdown
path: C:/path/dataset
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Keypoints
kpt_shape: [4, 2]  # [number of keypoints, number of dimensions]
flip_idx: [0, 1, 2, 3]

# Classes
names:
  0: weapon
```

## Modified Metric
Check out our wiki for more details [here](https://github.com/datalab-mi/Basegun-ml/wiki/Keypoint-Detection-Training#keypoint-detection-loss)

As previously explained, the keypoint metric has been modified. To do this, you need to clone the YOLOV8 repository and modify the keypoint loss function as explained [here](https://github.com/ultralytics/ultralytics/issues/2543).

The file to modify is [this one](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py) at line 106. You can replace it with the L1 Loss as the example explains, or use a different metric such as the Euclidean distance.


# Oriented Bounding Box training
The following documentation provides comprehensive and well-explained instructions for training the OBB model on a custom dataset. You can find it [here](https://github.com/hukaixuan19970627/yolov5_obb).
