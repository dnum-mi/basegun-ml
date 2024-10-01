[![PyPI](https://img.shields.io/pypi/v/basegun-ml)](https://pypi.org/project/basegun-ml/)
![GitHub License](https://img.shields.io/github/license/dnum-mi/basegun-ml)

# Basegun-ml

The Basegun-ml repository contains all the research code based on Machine Learning (ML) used in the [Basegun app](https://github.com/dnum-mi/Basegun), a tool designed to assist in the identification and legal categorization of firearms in France.

The repository includes two main folders:

- **Package Folder**: This contains the Python package used by the Basegun backend to run various machine learning algorithms. The package is available on pip:
```
pip install basegun-ml
```

- **Research Folder**: This contains all the experimental work that led to the creation of features in Basegun.

Currently, Basegun has three main ML features:

1. **Gun Mechanism Classification**: This feature categorizes an image into a list of families representing different firearm mechanisms. The classification is based on descriptive, objective criteria that are independent of legal classification.

2. **Measure Length Module**: Measuring the overall length of a firearm or its barrel length is crucial for its legal classification. In France, the classification of long guns depends on these measurements. This module measures these lengths using an image.

3. **Alarm Gun Recognition**: An alarm gun is a type of blank gun recognized as an alarm by French legislation. These guns are considered impossible to modify to make them lethal. The associated algorithm detects alarm guns using markings on the weapon.

For more information, you can check our [Wiki](https://github.com/dnum-mi/basegun-ml/wiki)

**Note**: Some models of Basegun-ml are based on YOLOv8 from [Ultralytics](https://github.com/ultralytics/ultralytics). To comply with the AGPL-3.0 license, Basegun-ml and all code using Basegun-ml must be under the same AGPL-3.0 license.
