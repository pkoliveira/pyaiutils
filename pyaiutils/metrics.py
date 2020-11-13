import pandas as pd
import numpy as np
import sklearn
import os
from . import utils

def recall(tp, p):
    return tp / p

def specificity(tn, n):
    return tn / n

def accuracy(tn, tp, p, n):
    return (tn + tp) / (p + n)

def precision(tp, fp):
    return tp / (fp + tp)

def f1_score(y_true, y_pred):
    if(len(np.unique(y_pred)) != len(np.unique(y_true))):
        y_pred = utils.to_1d(y_pred)
        y_true = utils.to_1d(y_true)
    return sklearn.metrics.f1_score(y_true, y_pred, average=None)

def prc_auc(y_true, y_pred, class_names):

    if(len(y_pred.shape) == 1):
        y_pred = utils.to_categorical(y_pred, np.unique(y_pred))
        y_true = utils.to_categorical(y_true, np.unique(y_true))
    n_classes = len(class_names)
    precision = dict()
    recall = dict()
    average_precision = []
    for i in range(n_classes):
        precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(y_true[:, i],
                                                                    y_pred[:, i])
        average_precision.append(
            sklearn.metrics.average_precision_score(y_true[:, i], y_pred[:, i]))
    return average_precision


def roc_auc(y_true, y_pred, class_names):
    if(len(y_pred.shape) == 1):
        y_pred = utils.to_categorical(y_pred, np.unique(y_pred))
        y_true = utils.to_categorical(y_true, np.unique(y_true))

    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc.append(sklearn.metrics.auc(fpr[i], tpr[i]))
    return roc_auc


def get_metrics(y_test, y_pred, class_names, save_path=None):
    """Returns and saves a dataframe containing the `['F1', 'ROC AUC', 'PRC AUC', 'Precision', 'Recall', 'Specificity', 'Accuracy']` metrics.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    class_names : array-like of shape (n_classes)
        List of labels containing each class of the dataset

    save_path : string, default=None
        Path to the folder where the metrics are to be saved
    """

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    if(len(y_test.shape) != 1):
        y_test = utils.to_1d(y_test)
        
    if(len(y_pred.shape) != 1):
        y_pred = utils.to_1d(y_pred)

    class_indexes = np.arange(len(class_names))

    matrix = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=class_indexes)


    TP = np.diag(matrix)
    FP = matrix.sum(axis=0) - TP
    FN = matrix.sum(axis=1) - TP
    TN = matrix.sum() - (FP + FN + TP)

    P = TP+FN
    N = TN+FP

    metrics_ = pd.DataFrame()
    rows = class_names.copy()
    rows.append('Média')
    metrics_['Classes'] = rows

    _f1 = np.around(f1_score(y_test, y_pred), decimals=2)
    _f1 = np.append(_f1, np.around(np.mean(_f1), decimals=2))

    _roc_auc = np.around(roc_auc(y_test, y_pred, class_names), decimals=2)
    _roc_auc = np.append(_roc_auc, np.around(np.mean(_roc_auc), decimals=2))

    _prc_auc = np.around(prc_auc(y_test, y_pred, class_names), decimals=2)
    _prc_auc = np.append(_prc_auc, np.around(np.mean(_prc_auc), decimals=2))

    _precision = np.around(precision(TP, FP), decimals=2)
    _precision = np.append(_precision, np.around(
        np.mean(_precision), decimals=2))

    _recall = np.around(recall(TP, P), decimals=2)
    _recall = np.append(_recall, np.around(np.mean(_recall), decimals=2))
    _specificity = np.around(specificity(TN, N), decimals=2)
    _specificity = np.append(_specificity, np.around(
        np.mean(_specificity), decimals=2))

    _accuracy = np.around(accuracy(TN, TP, P, N), decimals=2)
    _accuracy = np.append(_accuracy, np.around(np.mean(_accuracy), decimals=2))

    metrics_["F1"] = _f1
    metrics_["ROC AUC"] = _roc_auc
    metrics_["PRC AUC"] = _prc_auc
    metrics_["Precision"] = _precision
    metrics_["Recall"] = _recall
    metrics_["Specificity"] = _specificity
    metrics_["Accuracy"] = _accuracy

    if(save_path is not None):
        if(not os.path.isdir(save_path)):
            os.makedirs(save_path, exist_ok=True)
        metrics_.to_csv(os.path.join(save_path, 'metrics.csv'),
                        index=False, header=True)
    return metrics_