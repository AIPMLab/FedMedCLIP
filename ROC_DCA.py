import numpy as np
from sklearn.metrics import roc_curve, auc, det_curve
from scipy import interp
from sklearn.preprocessing import label_binarize, OneHotEncoder
import pandas as pd


def net_benefit_all(y_true):
    encoder = OneHotEncoder(sparse=False)
    y_true_one_hot = encoder.fit_transform(y_true.reshape(-1, 1))
    n_classes = y_true_one_hot.shape[ 1 ]
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefits = []
    for threshold in thresholds:
        net_benefit_for_class = []
        for class_idx in range(n_classes):
            prevalence = np.mean(y_true_one_hot[:, class_idx ])  # prevalence is the portions of this class among all the samples.
            net_benefit_for_class.append([
                prevalence - (1 - prevalence) * (threshold / (1 - threshold))])  # [100] samples, one dimension.
        net_benefits.append(np.mean(net_benefit_for_class))
    return net_benefits

def net_benefit_none():
    thresholds = np.linspace(0.01, 0.99, 100)
    return [0 for _ in thresholds]


def calculate_net_benefit_multiclass(y_true, y_proba):
    classes = set(y_true)
    num_classes = len(classes)
    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefits = [ ]
    for threshold in thresholds:
        class_net_benefit = np.zeros(num_classes)
        for class_idx in range(num_classes):
            w = threshold / (1 - threshold)  # Weight for false positives
            predictions = y_proba[ :, class_idx ] >= threshold
            tp = np.sum((predictions == 1) & (y_true == class_idx))
            fp = np.sum((predictions == 1) & (y_true != class_idx))
            epsilon = 1e-6
            class_net_benefit[ class_idx ] = (tp - fp * w) / len(y_true)

        net_benefits.append(np.mean(class_net_benefit))

    return net_benefits




def roc(y_true, pred):
    classes = []
    real = set(y_true)
    for i in real:
        classes.append(i)

    if len(classes) == 2:
        mean_fpr, mean_tpr, _ = roc_curve(y_true, pred[:,1])

    else:
        y_true = label_binarize(y_true, classes=classes)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        mean_fpr = np.linspace(0, 1, 100)
        for i in range(len(classes)):  # Calculate fpr, tpr for each class
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        mean_tpr = np.zeros_like(mean_fpr)
        for i in range(len(classes)):
            mean_tpr += interp(mean_fpr, fpr[i], tpr[i])
        mean_tpr /= len(classes)
        roc_auc['macro'] = auc(mean_fpr, mean_tpr)

    return mean_tpr, mean_fpr


def det(y_true, pred):
    classes = []
    real = set(y_true)
    for i in real:
        classes.append(i)
    y_true = label_binarize(y_true, classes=classes)
    fpr = dict()
    fnr = dict()
    for i in range(len(classes)):
        fpr[ i ], fnr[ i ], _ = det_curve(y_true[ :, i ], pred[ :, i ])

    all_fpr = np.unique(np.concatenate([ fpr[ i ] for i in range(len(classes)) ]))
    mean_fnr= np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_fnr += interp(all_fpr, fpr[ i ], fnr[ i ])
    mean_fnr /= len(classes)
    interp_fpr = np.linspace(0.01, 1, 100)
    interp_fnr = interp(interp_fpr, all_fpr, mean_fnr)

    return interp_fnr, interp_fpr


