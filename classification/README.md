# Training Basegun models

This folder contains the code and documentation used for creating the Machine Learning models used in Basegun project

# 1. Data preparation

## 1.1 Train / Validation

We choose from the beginning to separate the train/validation subsets for the following reasons:
- we can manually check the correct distribution of weapon images in validation, for example that in the “other gun” category there are not only Derringers
- Pytorch basically expects the Dataset to be organized like this
- reference datasets for scientific papers usually have their separate train and validation images.
- if train/val datasets are fixed (and not created randomly) we are sure 2 training with the same parameters will be done in the exact same conditions

## 1.2 Formatting

The images, to enter the network, must respect certain constraints:

- have only 3 channels
- be a precise size (depends on the DL model chosen, see section “3. Model choice”)
- be normalized in the same way as ImageNet

For these reasons we apply these transformations to the images of train and val.

## 1.3 Data augmentation

To increase the quality of training, we artificially increase the number of images that the neural network will see during training. At each epoch, the entire train dataset will undergo the following transformations:

- random rotation of +- 5°
- random perspective
- colorimetric distortions on brightness, saturation and contrast.

> NB: We do not apply vertical axial symmetry, often common in DL training, because **our images are not symmetry invariant**: the left side of arms is often not the same as the right side.

# 2. Training method : transfer learning VS fine-tuning

Since it is faster and works for small datasets, we decide for a start to train the model by doing transfer learning (also called [feature extraction in Pytorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor)), which means only training the last layers of the network.

Since our dataset is larger than most datasets used in the industry, if it seems relevant we can later fine-tune the result of this transfer learning (meaning retraining the whole network with a small learning rate), as suggested in [this Keras tutorial](https://keras.io/guides/transfer_learning/).

> *Which layers to train ?*  
> We choose for the moment to retrain only the last Dense layer of our network, which replaced the original network layer since it did not have the right number of classes. Later we can test adding BatchNormalization and Dropout before this Dense layer as suggested [in this Keras tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning).

# 3. DL model choice

By making a state of the art of classification models at the end of 2021, we notice that there are very few light models (<100M parameters) being better than EfficientNet. These are EfficientNetV2, FixEfficientNet and NoisyStudent. But [by studying their papers more closely](https://www.notion.so/Recherches-Machine-Learning-395831933cae492b84e46282196da432), we see that these adaptations are not relevant for the Basegun project.

<aside>
➡️ Therefore we will use a classif EfficientNet for a start.
</aside>
> EfficientNet model is only available in Pytorch starting torchvision>=0.11.3, torch>=1.10.2

Images entering EfficientNet must have a fixed size, listed in this table:

| EfficientNet Version | Input size | Recommended before crop size (optional) |
| --- | --- | --- |
| B0 | 224 | 256 |
| B1 | 240 | 256 |
| B2 | 288 | 288 |
| B3 | 300 | 320 |
| B4 | 380 | 384 |
| B5 | 456 | 489 |
| B6 | 528 | 561 |
| B7 | 600 | 633 |

# 4. Choice of parameters

- number of Epochs: at random, to be adjusted with training
- batch_size: the maximum 2^x value fitting in the GPU memory of the machines that we will use for training. This is recommended by [research](https://arxiv.org/pdf/1506.01186.pdf) in ML.
- type of optimizer: Adam is generally recommended because it adapts quite well on its own. A friend who does Pytorch in Computer Vision recommended me the [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) variant which apparently performs better in Pytorch than the Basic Adam.
- type of learning rate: this same friend recommended an LR type One Cycle. [Reading articles](https://towardsdatascience.com/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6) about this, it does seem like a solution recommended by the [scientific literature ](https://arxiv.org/pdf/1506.01186.pdf).

# 5. Model evaluation

## During training
We log the training and validation phases accuracy and loss at each Epoch to Tensorboard and visualize the curves.
```bash
# run from this directory
tensorboard --logdir runs
```
Open http://localhost:6006/ to visualize training/validation curves.

## Post training
* We write to a file details.txt the accuracy, precision, recall on val dataset and parameters used for the training of this model
* We write the confusion matrix (format .csv) on val dataset
* We write a .csv file containing for each image of val dataset, the probability output by the model to belong to each class. This file can be used later to visualize easily images of class X being confused with Y.
* We can highlight the parts of images being responsible of the predicted result by using a GradCam code we implemented.