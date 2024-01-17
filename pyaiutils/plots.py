import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import sklearn
import os

from . import utils
from . import metrics
import logging
log = logging.getLogger()

def plot_confusion_matrix(
    y_test, y_pred, class_names=None, save_path=None, 
    visualize=False, cmap=None, normalize=True, labels=True, 
    title='Matriz de confusão'
    ):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    uniques = np.unique(y_pred)

    if(len(y_pred.shape) == 1):
        cm = sklearn.metrics.confusion_matrix(y_test, y_pred, labels=uniques)
    else:
        y_test = utils.to_1d(y_test)
        y_pred = utils.to_1d(y_pred)

        cm = sklearn.metrics.confusion_matrix(y_test, y_pred)

    rotulos = []
    
    for index, value in enumerate(uniques):
        for i, v in enumerate(uniques):
            rotulos.append('')

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        cmap = utils.truncate_colormap(cmap, 0.35, 0.85)

    perc_cm = None
    if normalize:
        perc_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # modificação wenisten para poder elevar para percetual o resultado.
        perc_cm = perc_cm*100

    fig = plt.figure(figsize=(6, 6), edgecolor='k')  # (8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.clim(-5, 2.0)
    plt.xlim(-0.5, len(np.unique(y_test))-0.5)
    plt.ylim(len(np.unique(y_test))-0.5, -0.5)
    plt.title(title, fontsize=16)

    plt.colorbar()
    #plt.ylim(-0.5, len(class_names) - 0.5)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=16,
                   rotation=45, ha='right', rotation_mode="anchor")
        plt.yticks(tick_marks, class_names, fontsize=16)

    contador = 0
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, f"{'{:0.2f}%'.format(perc_cm[i, j])}\n({cm[i, j]})", fontsize=16,
                         horizontalalignment='center', verticalalignment='center',
                         color='white' if cm[i, j] > thresh else 'white')
                contador = contador+1
            else:
                plt.text(j, i, '{:,}'.format(cm[i, j]), fontsize=16,
                         horizontalalignment='center', verticalalignment='center',
                         color='white' if cm[i, j] > thresh else 'white')

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)

    if(save_path is not None):
        if(not os.path.isdir(save_path)):
            os.makedirs(save_path, exist_ok=True)
        
        dest = os.path.join(save_path, 'confusion_matriz.png')
        log.debug(f"Saving confusion matriz in {dest}")
        fig.savefig(dest, dpi=180, bbox_inches='tight')
    if(visualize):
        plt.show()
    plt.close()


def plot_auc_roc_multi_class(y_test, y_pred, class_names, visualize=False, save_path=None):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    if(len(y_pred.shape) == 1):
        y_pred = utils.to_categorical(y_pred, np.unique(y_test))
        y_test = utils.to_categorical(y_test, np.unique(y_test))
    # else:
        #y_pred = to_1d(y_pred)
        #y_test = to_1d(y_test)
        #y_pred = to_categorical(y_pred, class_names)
        #y_test = to_categorical(y_test, class_names)
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(
        y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

    lw = 2
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(15, 10))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
    roc_auc_of_classes = []
    for i, color in zip(range(n_classes), colors):
        roc_auc_of_classes.append(roc_auc[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC - ROC Curve')
    plt.legend(loc="lower right")

    if(save_path is not None):
        if(not os.path.isdir(save_path)):
            os.makedirs(save_path, exist_ok=True)
        dest = os.path.join(save_path, 'AUC_ROC.png')
        log.debug(f"Saving AUC_ROC in {dest}")
        
        plt.savefig(dest)
    
    if(visualize):
        plt.show()
    plt.close()


def plot_prc_auc_multiclass(y_test, y_pred, class_names, visualize=False, save_path=None):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    if(len(y_pred.shape) == 1):
        y_pred = utils.to_categorical(y_pred, np.unique(y_test))
        y_test = utils.to_categorical(y_test, np.unique(y_test))
    n_classes = len(class_names)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = sklearn.metrics.precision_recall_curve(y_test[:, i],
                                                                    y_pred[:, i])
        average_precision[i] = sklearn.metrics.average_precision_score(
            y_test[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = sklearn.metrics.precision_recall_curve(y_test.ravel(),
                                                                            y_pred.ravel())
    average_precision["micro"] = sklearn.metrics.average_precision_score(y_test, y_pred,
                                                                 average="micro")
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #      .format(average_precision["micro"]))

    colors = itertools.cycle(
        ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    plt.figure(figsize=(15, 10))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(class_names[i], average_precision[i]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    if(save_path is not None):
        if(not os.path.isdir(save_path)):
            os.makedirs(save_path, exist_ok=True)
        dest = os.path.join(save_path, 'AUC_PRC.png')
        log.debug(f"Saving AUC_PRC in {dest}")
        plt.savefig(dest)
    
    if(visualize):
        plt.show()
    plt.close()


def plot_graphics(y_true, y_pred, class_names=None, visualize=False, normalize=True, save_path=None):
    if(class_names is None):
        class_names = np.unique(np.array(y_pred))
        
    plot_confusion_matrix(
        y_true, y_pred, class_names=class_names, save_path=save_path, 
        visualize=visualize, normalize=normalize
        )
    plot_auc_roc_multi_class(
        y_true, y_pred,class_names=class_names, save_path=save_path,
        visualize=visualize
        )
    plot_prc_auc_multiclass(
        y_true, y_pred, class_names=class_names, save_path=save_path, 
        visualize=visualize
        )