import os
import numpy as np
import sklearn.metrics as skm
import torch.nn
from joblib import Parallel, delayed
from functools import partial
from sklearn.metrics import accuracy_score
from collections import defaultdict
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# def plot_reliability_diagram(preds, confs, labels, n_bins=10, title=None, save_dir=None):
#     bins = np.linspace(0, 1, n_bins + 1)
#     bin_indices = np.digitize(confs, bins) - 1
#
#     bin_acc = []
#     bin_confidences = []
#     for i in range(n_bins):
#         in_bin = bin_indices == i
#         if np.sum(in_bin) > 0:
#             accuracy = np.mean(labels[in_bin] == preds[in_bin])
#             mean_confidence = np.mean(confs[in_bin])
#         else:
#             accuracy = 0
#             mean_confidence = 0
#         bin_acc.append(accuracy)
#         bin_confidences.append(mean_confidence)
#
#     bin_acc = np.array(bin_acc)
#     bin_confidences = np.array(bin_confidences)
#
#     weights = np.histogram(confs, bins)[0] / len(confs)
#     ece = np.sum(weights * np.abs(bin_confidences - bin_acc))
#
#     # Plot
#     delta = 1.0 / n_bins
#     x = np.arange(0, 1, delta)
#     mid = np.linspace(delta / 2, 1 - delta / 2, n_bins)
#     error = np.abs(np.subtract(mid, bin_acc))
#
#     plt.rcParams["font.family"] = "Times New Roman"
#     # Size and axis limits
#     plt.figure(figsize=(8, 8))
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)  # Keep y-axis from 0 to 1
#
#     # Plot grid
#     plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0)
#
#     # Plot bars and identity line
#     plt.bar(x, bin_acc, color='b', width=delta, align='edge', edgecolor='k', zorder=5)  # Remove label for Outputs
#     plt.bar(x, error, bottom=np.minimum(bin_acc, mid), color='mistyrose', alpha=0.5, width=delta, align='edge', edgecolor='r', hatch='/', zorder=10)  # Remove label for Gap
#     ident = [0.0, 1.0]
#     plt.plot(ident, ident, linestyle='--', color='tab:grey', zorder=15)
#
#     # Add text labels above the plot (outside the y-axis range)
#     plt.text(0.15, 1.02, 'Outputs', fontsize=37, color='b', ha='center', va='bottom', transform=plt.gca().transAxes,
#              bbox=dict(boxstyle='round,pad=0.25', facecolor='blue', edgecolor='k', alpha=0.5))
#     plt.text(0.43, 1.02, 'Gap', fontsize=37, color='r', ha='center', va='bottom', transform=plt.gca().transAxes,
#              bbox=dict(boxstyle='round,pad=0.25', facecolor='mistyrose', edgecolor='r', alpha=0.5))
#     plt.text(0.8, 1.02, f'ECE: {ece * 100:.2f}%', fontsize=37, color='black', ha='center', va='bottom', transform=plt.gca().transAxes,
#              bbox=dict(boxstyle='round,pad=0.25', facecolor='wheat', edgecolor='orange', alpha=0.5))
#
#     # Labels and legend
#     plt.ylabel('Accuracy', fontsize=49)
#     plt.xlabel('Confidence', fontsize=49)
#
#     # Set tick label font size
#     plt.tick_params(axis='both', which='major', labelsize=47)
#
#     # Title
#     if title is not None:
#         plt.title(title, fontsize=20)
#
#     plt.tight_layout()
#
#     # Save figure
#     if save_dir is not None:
#         plt.savefig(save_dir, bbox_inches='tight')  # Ensure the text is not cut off
#
#     return plt

def plot_reliability_diagram(preds, confs, labels, n_bins=10, title=None, save_dir=None):
    import numpy as np
    import matplotlib.pyplot as plt

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confs, bins) - 1

    bin_acc = []
    bin_confidences = []
    for i in range(n_bins):
        in_bin = bin_indices == i
        if np.sum(in_bin) > 0:
            accuracy = np.mean(labels[in_bin] == preds[in_bin])
            mean_confidence = np.mean(confs[in_bin])
        else:
            accuracy = 0
            mean_confidence = 0
        bin_acc.append(accuracy)
        bin_confidences.append(mean_confidence)

    bin_acc = np.array(bin_acc)
    bin_confidences = np.array(bin_confidences)

    weights = np.histogram(confs, bins)[0] / len(confs)
    ece = np.sum(weights * np.abs(bin_confidences - bin_acc))

    # Plot
    delta = 1.0 / n_bins
    x = np.arange(0, 1, delta)
    mid = np.linspace(delta / 2, 1 - delta / 2, n_bins)
    error = np.abs(np.subtract(mid, bin_acc))

    # --- 颜色定义：将深蓝色 ('b') 替换为浅蓝色 ---
    LIGHT_BLUE = '#6495ED'

    plt.rcParams["font.family"] = "Times New Roman"

    # Size and axis limits: 保持 (12, 12)
    plt.figure(figsize=(12, 12))
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Plot grid
    plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0)

    # Plot bars and identity line
    plt.bar(x, bin_acc, color=LIGHT_BLUE, width=delta, align='edge', edgecolor='k', zorder=5)
    plt.bar(x, error, bottom=np.minimum(bin_acc, mid), color='mistyrose', alpha=0.5, width=delta, align='edge',
            edgecolor='r', hatch='/', zorder=10)
    ident = [0.0, 1.0]
    plt.plot(ident, ident, linestyle='--', color='tab:grey', zorder=15)

    # Add text labels inside the plot frame, top-left, three lines
    # 使用图表的坐标系（Axes coordinates）而不是变换坐标系
    # x=0.05, y=0.95 (左上角), 调整行间距
    # 增大字体大小 (fontsize)

    # Outputs 标签 (第一行)
    plt.text(0.05, 0.95, 'Outputs', fontsize=80, color=LIGHT_BLUE, ha='left', va='top',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round,pad=0.2', facecolor=LIGHT_BLUE, edgecolor='k', alpha=0.5))

    # Gap 标签 (第二行)
    plt.text(0.55, 0.95, 'Gap', fontsize=80, color='r', ha='left', va='top', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='mistyrose', edgecolor='r', alpha=0.5))

    # ECE 标签 (第三行)
    plt.text(0.05, 0.78, f'ECE: {ece * 100:.2f}%', fontsize=80, color='black', ha='left', va='top',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', edgecolor='orange', alpha=0.5))


    # Labels and legend
    plt.ylabel('Accuracy', fontsize=85)
    plt.xlabel('Confidence', fontsize=85)

    # Set tick label font size
    plt.tick_params(axis='both', which='major', labelsize=85)

    # Title
    if title is not None:
        plt.title(title, fontsize=36)

    plt.tight_layout()

    # Save figure
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')

    return plt

def ECE(conf, pred, gt, conf_bin_num=10):
    """
    Expected Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)

    Returns:
        ece: expected calibration error
    """
    bins = np.linspace(0, 1, conf_bin_num + 1)
    bin_indices = np.digitize(conf, bins) - 1

    bin_acc = []
    bin_confidences = []
    for i in range(conf_bin_num):

        in_bin = bin_indices == i

        if np.sum(in_bin) > 0:
            accuracy = np.mean(gt[in_bin] == pred[in_bin])
            mean_confidence = np.mean(conf[in_bin])
        else:
            accuracy = 0
            mean_confidence = 0
        bin_acc.append(accuracy)
        bin_confidences.append(mean_confidence)

    bin_acc = np.array(bin_acc)
    bin_confidences = np.array(bin_confidences)

    weights = np.histogram(conf, bins)[0] / len(conf)
    ece = np.sum(weights * np.abs(bin_confidences - bin_acc)) # value, not percentage. so we have to multiply 100.

    return ece


def ECE_per_class(conf, pred, gt, conf_bin_num=10):
    """
    Expected Calibration Error for each class in a multi-class classification problem.

    Args:
        conf (numpy.ndarray): Array of confidence scores (shape: (n_samples, n_classes)).
        pred (numpy.ndarray): Array of predicted class labels (shape: (n_samples,)).
        gt (numpy.ndarray): Array of true class labels (shape: (n_samples,)).
        conf_bin_num (int): Number of bins to divide the confidence scores.

    Returns:
        ece_per_class (dict): Dictionary containing ECE values for each class.
    """
    n_classes = conf.shape[1]
    ece_per_class = {}

    bins = np.linspace(0, 1, conf_bin_num + 1)

    for class_idx in range(n_classes):
        class_conf = conf[:, class_idx]  # Confidence scores for the current class
        class_pred = (pred == class_idx).astype(int)  # Predictions for the current class (0 or 1)
        class_gt = (gt == class_idx).astype(int)  # True labels for the current class (0 or 1)

        bin_indices = np.digitize(class_conf, bins) - 1
        bin_acc = []
        bin_confidences = []

        for i in range(conf_bin_num):
            in_bin = bin_indices == i

            if np.sum(in_bin) > 0:
                accuracy = np.mean(class_gt[in_bin] == class_pred[in_bin])
                mean_confidence = np.mean(class_conf[in_bin])
            else:
                accuracy = 0
                mean_confidence = 0

            bin_acc.append(accuracy)
            bin_confidences.append(mean_confidence)

        bin_acc = np.array(bin_acc)
        bin_confidences = np.array(bin_confidences)

        weights = np.histogram(class_conf, bins)[0] / len(class_conf)
        ece = np.sum(weights * np.abs(bin_confidences - bin_acc))

        ece_per_class[f'class_{class_idx}'] = ece

    return ece_per_class

def accuracy_retention_curve(ground_truth, predictions, uncertainties, fracs_retained, parallel_backend=None):
    # uncertainties =

    def compute_acc(frac_, preds_, gts_, N_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate((preds_[:pos], gts_[pos:]))

        return skm.accuracy_score(gts_, curr_preds)

    if parallel_backend is None:
        parallel_backend = Parallel(n_jobs=1)

    ordering = uncertainties.argsort()
    gts = ground_truth[ordering].copy()
    preds = predictions[ordering].copy()

    N = len(gts)

    process = partial(compute_acc, preds_=preds, gts_=gts, N_=N)
    accuracy_scores = np.asarray(
        parallel_backend(delayed(process)(frac) for frac in fracs_retained)
    )

    return accuracy_scores


def save_retention_curve_values(testLabels, t1, m_name, net):
    fracs_retained = np.log(np.arange(200 + 1)[1:])
    fracs_retained /= np.amax(fracs_retained)
    n_jobs = 1

    with Parallel(n_jobs=n_jobs) as parallel_backend:
        accuracy_rc = accuracy_retention_curve(ground_truth=np.argmax(testLabels, axis=1),
                                               predictions=np.argmax(t1, axis=1), uncertainties=1 - np.max(t1, axis=1),
                                               fracs_retained=fracs_retained, parallel_backend=parallel_backend)

    # print(accuracy_rc)
    print(net)
    save_dir = './RetentionCurves/ISIC2019/MICCAI2025/'
    print(f'Saved in: {save_dir}')
    if net == 'baseline':
        np.savetxt(f'{save_dir}/{m_name}_baseline.txt', accuracy_rc, delimiter=",")
    elif net == 'LS':
        np.savetxt(f'{save_dir}/{m_name}_LS.txt', accuracy_rc, delimiter=",")
    elif net == 'FL':
        np.savetxt(f'{save_dir}/{m_name}_FL.txt', accuracy_rc, delimiter=",")
    elif net == 'dca':
        np.savetxt(f'{save_dir}/{m_name}_dca.txt', accuracy_rc, delimiter=",")
    elif net == 'mdca':
        np.savetxt(f'{save_dir}/{m_name}_mdca.txt', accuracy_rc, delimiter=",")
    elif net == 'ours_alpha05':
        np.savetxt(f'{save_dir}/{m_name}_ours.txt', accuracy_rc, delimiter=",")


def get_other_scores(probs, targets, nbin=10, epsilon=1e-7):
    """
    Calculate accuracy, ECE, negative log-likelihood, Brier score
    :param probs: (numpy.array) predictions of dimension N x C where N is number of example, C is classes
    :param targets: (numpy.array) targets of dimension N
    :param nbin: (int) number of bins for calculating ECE
    :param fn: (function) function to transform conf - acc to fn(conf - acc) for ECE, sECE
    :return: tuple containing Accuracy, ECE, NLL, Brier
    """
    preds = np.argmax(probs, axis=1)
    correct = (preds == targets)
    class_probs = np.take_along_axis(probs, targets.astype(np.uint8)[:, None], axis=1)
    class_probs = np.clip(class_probs, epsilon, 1.0)
    nll = np.mean(-np.log(class_probs))
    maxprobs = np.max(probs, axis=-1)
    one_hot = np.eye(probs.shape[1])[targets.astype(np.int32)]
    brier_score = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return nll, brier_score