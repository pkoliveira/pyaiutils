# PYAIUtils


![Version](https://img.shields.io/pypi/v/pyaiutils)
![Repo Size](https://img.shields.io/github/repo-size/CRIA-CIMATEC/pyaitutils)
![Code Size](https://img.shields.io/github/languages/code-size/CRIA-CIMATEC/pyaitutils)
![Total lines](https://img.shields.io/tokei/lines/github/CRIA-CIMATEC/pyaitutils)
![Downloads](https://img.shields.io/github/downloads/CRIA-CIMATEC/pyaitutils/total) 
![Contributors](https://img.shields.io/github/contributors/CRIA-CIMATEC/pyaitutils?color=dark-green) 
![Issues](https://img.shields.io/github/issues/CRIA-CIMATEC/pyaitutils) 
![License](https://img.shields.io/github/license/CRIA-CIMATEC/pyaitutils) 


This library was developed to assist in the validation of Artificial Intelligence models for classification problems, allowing the Data Scientist to obtain, with a few lines of code, a complete and robust list of information.

Other metrics will be added to make validation more robust, as well as options to enable or disable metrics, as well as validation for other types of issues.

## Installation

Use can be done by cloning the repository.

* The project can also be downloaded through the package manager [pip](https://pypi.org/project/pyaiutils/), but it is still in the correction phase.

```bash
pip install pyaiutils
```

## Requirements
```
sklearn >= 0.23.1
numpy >= 1.18.1
matplotlib >= 3.1.3
pandas >= 1.2.4
imblearn >= 0.7.0
```

## Usage

```python
import pyaiutils
# OR
from pyaiutils import get_metrics, plot_graphics

'''
 It will present a table with a set of metrics for model validation.
 The metrics are:
- F1-score
- ROC Auc
- PRC AUC
- Precision
- Recall
- Specificity
- Accuracy
'''
pyaiutils.get_metrics(y_test, y_pred [, classes])

'''
It will present three graphics, as follows:
- Confusion matrix
- AUC ROC curve
- AUC PRC curve
'''
pyaiutils.plot_graphics(y_test, y_pred [, classes])
```

## Example using Cifar 10
```
pyaiutils.get_metrics(y_test, y_pred, classes)
```

| # | Classes | F1-score | ROC AUC | PRC AUC | Precision | Recall | Specificity | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | airplane | 0.59 | 0.91 | 0.61 | 0.54 | 0.64 | 0.94 | 0.91 |
| 1 | automobile | 0.65 | 0.94 | 0.71 | 0.58 | 0.75 | 0.94 | 0.92 |
| 2 | bird | 0.26 | 0.82 | 0.36 | 0.44 | 0.18 | 0.97 | 0.90 |
| 3 | cat | 0.27 | 0.82 | 0.30 | 0.32 | 0.23 | 0.95 | 0.88 |
| 4 | deer | 0.40 | 0.85 | 0.41 | 0.38 | 0.42 | 0.92 | 0.87 |
| 5 | dog | 0.44 | 0.86 | 0.44 | 0.42 | 0.45 | 0.93 | 0.88 |
| 6 | frog | 0.57 | 0.91 | 0.61 | 0.61 | 0.53 | 0.96 | 0.92 |
| 7 | horse | 0.53 | 0.91 | 0.66 | 0.40 | 0.78 | 0.87 | 0.86 |
| 8 | ship | 0.48 | 0.93 | 0.67 | 0.74 | 0.35 | 0.99 | 0.92 |
| 9 | truck | 0.52 | 0.91 | 0.59 | 0.55 | 0.49 | 0.96 | 0.91 |
| 10 | Mean | 0.47 | 0.89 | 0.54 | 0.50 | 0.48 | 0.94 | 0.90 |


```
pyaiutils.plot_graphics(y_test, y_pred, classes)
```
![Confusion Matrix](https://i.imgur.com/lMOeECX.png)
![AUC - ROC Curve](https://i.imgur.com/b70dA0C.png)
![AUC - PRC Curve](https://i.imgur.com/jLHgbVS.png)

## How metrics are calculated

Take the next table as an example:
![Class Table](https://i.imgur.com/Hd7PV97.png)
[Source](https://www.researchgate.net/post/Can_someone_help_me_to_calculate_accuracy_sensitivity_of_a_66_confusion_matrix)

### 1. First step is to know how many classes we have:
```
classes = [N, L, R, A, P, V]
```

### 2. For each class, we will have to obtain the TP, TN, FP, FN, and perform the calculations as in the following image:
![Confusion Matrix computation](https://i.imgur.com/ILjo2GB.png)
Note: Above example only for class N, the other classes follow the same idea.

```
So for class N
TP 1971
TN 9501
FP 37
NF 29

Acc 99.43%
precision= TP / (TP + FP) 98.16%
sensitivity = TP / (TP + FN) 98.55%
specificity = TN / (FP + TN) 99.61%
F-score = 2TP / (2TP + FP + FN) 98.35%


For class L
TP 1940
TN 9532
FP 83
FN 60

Acc 98.77%
precision= TP / (TP + FP) 95.90%
sensitivity = TP / (TP + FN) 97.00%
specificity = TN / (FP + TN) 99.14%
F-score = 2TP /(2TP + FP + FN) 96.45%


For class R
TP 1891
TN 9581
FP 195
FN 109

Acc 97.42%
precision= TP / (TP + FP) 90.65%
sensitivity = TP / (TP + FN) 94.55%
specificity = TN / (FP + TN) 98.01%
F-score = 2TP /(2TP + FP + FN) 92.56%

For class A
TP 1786
TN 9686
FP 137
FN 214

Acc 97.03%
precision= TP / (TP + FP) 92.88%
sensitivity = TP / (TP + FN) 89.30%
specificity = TN / (FP + TN) 98.61%
F-score = 2TP /(2TP + FP + FN) 91.05%


Class P
TP 1958
TN 9514
FP 36
FN 42

Acc 99.32%
precision= TP / (TP + FP) 98.19%
sensitivity = TP / (TP + FN) 97.90%
specificity = TN / (FP + TN) 99.62%
F-score = 2TP /(2TP + FP + FN) 98.05%


Class V
TP 1926
TN 9546
FP 40
FN 74
Acc 99.02%

precision= TP / (TP + FP) 97.97%
sensitivity = TP / (TP + FN) 96.30%
specificity = TN / (FP + TN) 99.58%
F-score = 2TP /(2TP + FP + FN) 97.13%


Accuracy average for all classes 98.50%
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)
