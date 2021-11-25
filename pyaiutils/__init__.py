from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import pandas as pd
import math
import os

def cycle(iterable):
    """Form a complex number.

    Keyword arguments:
    iterable -- the real part (default 0.0)
    """
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        for element in saved:
              yield element
    
    return saved

def iteraux(frstIter, scndIter):
    """Form a complex number.

    Keyword arguments:
    frstIter -- the real part (default 0.0)
    scndIter -- the imaginary part (default 0.0)
    """
    itermtx = []
    for i in range(frstIter):
        for j in range(scndIter):
            itermtx.append((i, j))
    
    return itermtx

def prc_auc(y_true, y_pred, class_names):
    """Form a complex number.

    Keyword arguments:
    y_true -- the real part (default 0.0)
    y_pred -- the imaginary part (default 0.0)
    class_names -- the imaginary part (default 0.0)
    """
    n_classes = len(class_names)
    precision = dict()
    recall = dict()
    average_precision = []
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true[:, i],
                                                                    y_pred[:, i])
        average_precision.append(
            metrics.average_precision_score(y_true[:, i], y_pred[:, i]))
    return average_precision


def roc_auc(y_true, y_pred, class_names):
    """Form a complex number.

    Keyword arguments:
    y_true -- the real part (default 0.0)
    y_pred -- the imaginary part (default 0.0)
    class_names -- the imaginary part (default 0.0)
    """
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = []
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc.append(metrics.auc(fpr[i], tpr[i]))
    return roc_auc


def weighted_average(metric_p, metric_n, P, N):
    """Form a complex number.

    Keyword arguments:
    metric_p -- the real part (default 0.0)
    metric_n -- the imaginary part (default 0.0)
    P -- the real part (default 0.0)
    N -- the imaginary part (default 0.0)
    """
    media_p = (metric_p*P + metric_n*N)/(P+N)
    return media_p


def recall(tp, p):
    """Form a complex number.

    Keyword arguments:
    tp -- the real part (default 0.0)
    p -- the imaginary part (default 0.0)
    """
    return tp/p


def specificity(tn, n):
    """Form a complex number.

    Keyword arguments:
    tn -- the real part (default 0.0)
    n -- the imaginary part (default 0.0)
    """
    return tn/n


def accuracy(tn, tp, p, n):
    """Form a complex number.

    Keyword arguments:
    tn -- the real part (default 0.0)
    tp -- the imaginary part (default 0.0)
    p -- the real part (default 0.0)
    n -- the imaginary part (default 0.0)
    """
    return (tn + tp) / (p + n)


def precision(tp, fp):
    """Form a complex number.

    Keyword arguments:
    tp -- the real part (default 0.0)
    fp -- the imaginary part (default 0.0)
    """
    return tp/(fp + tp)


def f1_score(y_true, y_pred):
    """Form a complex number.

    Keyword arguments:
    y_true -- the real part (default 0.0)
    y_pred -- the imaginary part (default 0.0)
    """
    return metrics.f1_score(y_true, y_pred, average=None)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Form a complex number.

    Keyword arguments:
    cmap -- the real part (default 0.0)
    minval -- the imaginary part (default 0.0)
    maxval -- the real part (default 0.0)
    n -- the imaginary part (default 0.0)
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_confusion_matrix(y_test, y_pred, class_names=None, save_path=None, visualize=False, cmap=None, normalize=True, labels=True, title='Matriz de confusão'):
    """Form a complex number.

    Arguments:
    y_true -- the real part (default 0.0)
    y_pred -- the imaginary part (default 0.0)
    class_names -- the imaginary part (default 0.0)
    save_path -- the imaginary part (default 0.0)
    visualize -- the imaginary part (default 0.0)
    cmap -- the imaginary part (default 0.0)
    normalize -- the imaginary part (default 0.0)
    labels -- the imaginary part (default 0.0)
    title -- the imaginary part (default 0.0)
    """
    y_test = np.array(y_test)
    
    try:
        y_pred = np.argmax(y_pred, axis=1) 
    except Exception as e:
        print(e)
         
        
    uniques = np.unique(y_test)

    if(len(class_names) == 0):
        if(len(y_test.shape) == 1):
            class_names = np.unique(y_test)
        else:
            class_names = np.unique(np.argmax(y_test, axis=1))
    
    if(len(y_test.shape) > 1):
        y_test = np.argmax(y_test, axis=1)
        
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    rotulos = []
    for index, value in enumerate(uniques):
        for i, v in enumerate(uniques):
            rotulos.append('')

    if cmap is None:
        cmap = plt.get_cmap('Blues')
        cmap = truncate_colormap(cmap, 0.35, 0.85)

    perc_cm = None
    if normalize:
        perc_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # modificação wenisten para poder elevar para percetual o resultado.
        perc_cm = perc_cm*100

    factor = (len(uniques) // 3)
    expansion = 1 if factor < 1 else factor
    
    fig = plt.figure(figsize=(6 * expansion, 6 * expansion), edgecolor='k')  # (8, 6))
    plt.title(title, fontsize=16)
    plt.imshow(cm, interpolation='nearest', cmap='Blues',aspect='auto', alpha=1)
    
    plt.colorbar()
    #plt.ylim(-0.5, len(class_names) - 0.5)
    
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, fontsize=16,
                   rotation=45, ha='right', rotation_mode="anchor")
        plt.yticks(tick_marks, class_names, fontsize=16)

    if labels:
        for i, j in iteraux(cm.shape[0], cm.shape[1]):
            if normalize:
                perc_cm[i, j] = 0 if math.isnan(perc_cm[i, j]) else perc_cm[i, j]
                cm[i, j] = 0 if math.isnan(cm[i, j]) else cm[i, j]
                plt.text(j, i, f"{'{:0.2f}%'.format(perc_cm[i, j])}\n({cm[i, j]})", fontsize=16,
                         horizontalalignment='center', verticalalignment='center',
                         color='white' if cm[i, j] > thresh else 'black')
            else:
                plt.text(j, i, '{:,}'.format(cm[i, j]), fontsize=16,
                         horizontalalignment='center', verticalalignment='center',
                         color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)

    if(save_path is not None):
        if(not os.path.isdir(save_path)):
            os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, 'confusion_matriz.png'),
                    dpi=180, bbox_inches='tight')
    if(visualize):
        plt.show()
    plt.close()


def plot_auc_roc_multi_class(y_test, y_pred, class_names, save_path=None):
    """Form a complex number.

    Arguments:
    y_true -- the real part (default 0.0)
    y_pred -- the imaginary part (default 0.0)
    class_names -- the imaginary part (default 0.0)
    save_path -- the imaginary part (default 0.0)
    """
    y_test = np.array(y_test)
    
    try:
        y_pred = np.argmax(y_pred, axis=1) 
    except Exception as e:
        print(e)
    
    if(len(class_names) == 0):
        if(len(y_test.shape) == 1):
            class_names = np.unique(y_test)
        else:
            class_names = np.unique(np.argmax(y_test, axis=1))
    
    if(len(y_test.shape) == 1):
        matrix = metrics.confusion_matrix(y_test, y_pred)
        y_test = pd.get_dummies(y_test).values
    else:
        matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        
    if(len(y_pred.shape) == 1):
        class_check = np.unique(y_pred)
        y_pred = pd.get_dummies(y_pred).values
        
    if(y_pred.shape[1] != y_test.shape[1]):
        y_pred_df = pd.DataFrame(columns=class_names)
        i_control = 0
        for i in class_names:
            if i in class_check:
                y_pred_df[i] = y_pred[:, i_control]
                i_control += 1
            else:
                y_pred_df[i] = np.array([0]*y_pred.shape[0])
                
        y_pred = y_pred_df.values
        
        
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
        y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

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
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

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

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue']) #itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
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
        plt.savefig(os.path.join(save_path, 'AUC_ROC.png'))
    plt.show()


def plot_prc_auc_multiclass(y_test, y_pred, class_names, save_path=None):
    """Form a complex number.

    Arguments:
    y_true -- the real part (default 0.0)
    y_pred -- the imaginary part (default 0.0)
    class_names -- the imaginary part (default 0.0)
    save_path -- the imaginary part (default 0.0)
    """
    y_test = np.array(y_test)
    
    try:
        y_pred = np.argmax(y_pred, axis=1) 
    except Exception as e:
        print(e)
          
    
    if(len(class_names) == 0):
        if(len(y_test.shape) == 1):
            class_names = np.unique(y_test)
        else:
            class_names = np.unique(np.argmax(y_test, axis=1))
    
    if(len(y_test.shape) == 1):
        matrix = metrics.confusion_matrix(y_test, y_pred)
        y_test = pd.get_dummies(y_test).values
    else:
        matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        
    if(len(y_pred.shape) == 1):
        class_check = np.unique(y_pred)
        y_pred = pd.get_dummies(y_pred).values
        
    if(y_pred.shape[1] != y_test.shape[1]):
        y_pred_df = pd.DataFrame(columns=class_names)
        i_control = 0
        for i in class_names:
            if i in class_check:
                y_pred_df[i] = y_pred[:, i_control]
                i_control += 1
            else:
                y_pred_df[i] = np.array([0]*y_pred.shape[0])
                
        y_pred = y_pred_df.values
        
    n_classes = len(class_names)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_test[:, i],
                                                                    y_pred[:, i])
        average_precision[i] = metrics.average_precision_score(
            y_test[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_test.ravel(),
                                                                            y_pred.ravel())
    average_precision["micro"] = metrics.average_precision_score(y_test, y_pred,
                                                                 average="micro")
    # print('Average precision score, micro-averaged over all classes: {0:0.2f}'
    #      .format(average_precision["micro"]))

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue']) #itertools.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
#     print(list(colors))
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
        plt.savefig(os.path.join(save_path, 'AUC_PRC.png'))
    plt.show()


############################################################## Função Principal

def get_metrics(y_test, y_pred, class_names=[], save_path=None):
    """Form a complex number.

    Arguments:
    y_true -- the real part (default 0.0)
    y_pred -- the imaginary part (default 0.0)
    class_names -- the imaginary part (default 0.0)
    save_path -- the imaginary part (default 0.0)
    """
    y_test = np.array(y_test)
    
    try:
        y_pred = np.argmax(y_pred, axis=1) 
    except Exception as e:
        print(e)
    
    if(len(y_test.shape) == 1):
        matrix = metrics.confusion_matrix(y_test, y_pred)
        y_test = pd.get_dummies(y_test).values
    else:
        matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        
    if(len(y_pred.shape) == 1):
        y_pred = pd.get_dummies(y_pred).values
        
    if(len(class_names) == 0):
        if(len(y_test.shape) == 1):
            class_names = np.unique(y_pred)
        else:
            class_test = np.unique(np.argmax(y_test, axis=1))
            class_pred = np.unique(np.argmax(y_pred, axis=1))
            class_names = class_test if len(class_test) >= len(class_pred) else class_pred
        
    if(y_pred.shape[1] != y_test.shape[1]):
        if(len(class_test) > len(class_pred)):
            y_pred_df = pd.DataFrame(columns=class_names)
            i_control = 0
            for i in class_names:
                if i in class_pred:
                    y_pred_df[i] = y_pred[:, i_control]
                    i_control += 1
                else:
                    y_pred_df[i] = np.array([0]*y_pred.shape[0])

            y_pred = y_pred_df.values
        else:
            y_test_df = pd.DataFrame(columns=class_names)
            i_control = 0
            for i in class_names:
                if i in class_test:
                    y_test_df[i] = y_test[:, i_control]
                    i_control += 1
                else:
                    y_test_df[i] = np.array([0]*y_test.shape[0])

            y_test = y_test_df.values
        
    TP = np.diag(matrix)
    FP = matrix.sum(axis=0) - TP
    FN = matrix.sum(axis=1) - TP
    TN = matrix.sum() - (FP + FN + TP)

    P = TP+FN
    N = TN+FP

    metrics_ = pd.DataFrame()
    rows = list(class_names.copy())
    rows.append('Average')
    rows.append('Weighted Avg')
    metrics_['Classes'] = rows
    
    _f1 = np.around(f1_score(y_test, y_pred), decimals=2)
    _f1 = np.append(_f1, np.around(np.mean(_f1), decimals=2))
    _f1 = np.append(_f1, np.round(weighted_average(_f1[0], _f1[1], P[0], P[1]), decimals=2))

    _roc_auc = np.around(roc_auc(y_test, y_pred, class_names), decimals=2)
    _roc_auc = np.append(_roc_auc, np.around(np.mean(_roc_auc), decimals=2))
    _roc_auc = np.append(_roc_auc, np.round(weighted_average(_roc_auc[0], _roc_auc[1], P[0], P[1]), decimals=2))

    _prc_auc = np.around(prc_auc(y_test, y_pred, class_names), decimals=2)
    _prc_auc = np.append(_prc_auc, np.around(np.mean(_prc_auc), decimals=2))
    _prc_auc = np.append(_prc_auc, np.round(weighted_average(_prc_auc[0], _prc_auc[1], P[0], P[1]), decimals=2))

    _precision = np.around(precision(TP, FP), decimals=2)
    _precision = np.append(_precision, np.around(np.mean(_precision), decimals=2))
    _precision = np.append(_precision, np.round(weighted_average(_precision[0], _precision[1], P[0], P[1]), decimals=2))

    _recall = np.around(recall(TP, P), decimals=2)
    _recall = np.append(_recall, np.around(np.mean(_recall), decimals=2))
    _recall = np.append(_recall, np.round(weighted_average(_recall[0], _recall[1], P[0], P[1]), decimals=2))
    
    _specificity = np.around(specificity(TN, N), decimals=2)
    _specificity = np.append(_specificity, np.around(np.mean(_specificity), decimals=2))
    _specificity = np.append(_specificity, np.round(weighted_average(_specificity[0], _specificity[1], P[0], P[1]), decimals=2))

    _accuracy = np.around(accuracy(TN, TP, P, N), decimals=2)
    _accuracy = np.append(_accuracy, np.around(np.mean(_accuracy), decimals=2))
    _accuracy = np.append(_accuracy, np.round(weighted_average(_accuracy[0], _accuracy[1], P[0], P[1]), decimals=2))
    
    metrics_["F1"] = [0 if math.isnan(i) else i for i in _f1]
    metrics_["ROC AUC"] = [0 if math.isnan(i) else i for i in _roc_auc]
    metrics_["PRC AUC"] = [0 if math.isnan(i) else i for i in _prc_auc]
    metrics_["Precision"] = [0 if math.isnan(i) else i for i in _precision]
    metrics_["Recall"] = [0 if math.isnan(i) else i for i in _recall]
    metrics_["Specificity"] = [0 if math.isnan(i) else i for i in _specificity]
    metrics_["Accuracy"] = [0 if math.isnan(i) else i for i in _accuracy]

    return metrics_

def plot_graphics(y_true, y_pred, class_names=[], save_path=None):
    """Form a complex number.

    Arguments:
    y_true -- the real part (default 0.0)
    y_pred -- the imaginary part (default 0.0)
    class_names -- the imaginary part (default 0.0)
    save_path -- the imaginary part (default 0.0)
    """
    if(len(class_names) == 0):
        if(len(y_test.shape) == 1):
            class_names = np.unique(y_test)
        else:
            class_names = np.unique(np.argmax(y_test, axis=1))
    
    
    display(plot_confusion_matrix(y_true, y_pred, visualize=True, normalize=True, class_names=class_names, save_path=save_path))
    display(plot_auc_roc_multi_class(y_true, y_pred, class_names=class_names, save_path=save_path))
    display(plot_prc_auc_multiclass(y_true, y_pred, class_names=class_names, save_path=save_path))

