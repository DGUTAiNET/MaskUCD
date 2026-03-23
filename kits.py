import numpy as np
import errno
import os
import sklearn.metrics as ms
import kornia
from sklearn.metrics import roc_curve, auc, confusion_matrix, cohen_kappa_score, f1_score
# from sklearn.mixture import GaussianMixture
# from scipy.stats import skew
# from skimage import morphology
# from scipy.stats import norm, boxcox
# from scipy.signal import find_peaks
# from sklearn.cluster import KMeans
# from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian
# import pydensecrf.densecrf as dcrf
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import csv
# import torch
# from collections import deque
# import cv2
# from skimage import measure


def bestCM(d, gt):
    H, W = gt.shape[-2:]
    d = d.flatten()
    gt = gt.flatten().astype(int)
    fpr, tpr, thre = roc_curve(gt, d)
    sample_idx = np.linspace(0, len(thre) - 1, 200, dtype=int)
    sampled_thres = thre[sample_idx]
    f1_scores = []
    for t in sampled_thres:
        pred = (d >= t).astype(int)
        f1_scores.append(f1_score(gt, pred))

    best_idx = np.argmax(f1_scores)
    best_thre = sampled_thres[best_idx]

    pred_flat = (d >= best_thre).astype(int)
    cm = pred_flat.reshape(H, W)

    tn, fp, fn, tp = confusion_matrix(gt, pred_flat, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    kappa = cohen_kappa_score(gt, pred_flat)

    metrics_text = (
        f"Optimal Threshold (Kappa): {best_thre:.4f}\n"
        f"OA: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"Kappa: {kappa:.4f}\n"
        f"F1: {f1:.4f}\n"
    )
    return cm, metrics_text



def gaussian_blur_(input_tensor, kernel_size=3, sigma=1.0):
    """
    input_tensor:  (1, 1, H, W)
    """
    assert input_tensor.dim() == 4 and input_tensor.shape[1] == 1
    return kornia.filters.gaussian_blur2d(
        input_tensor,
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma) if isinstance(sigma, float) else sigma,
        border_type='replicate'
    )


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return path


def compute_metrics(gt, cm, metrics_history):
    gt = np.array(gt, dtype=np.uint8).flatten()
    cm = np.array(cm, dtype=np.uint8).flatten()

    OA = ms.accuracy_score(gt, cm)
    Precision = ms.precision_score(gt, cm, zero_division=0)
    Recall = ms.recall_score(gt, cm, zero_division=0)
    Kappa = ms.cohen_kappa_score(gt, cm)
    F1 = ms.f1_score(gt, cm, zero_division=0)

    metrics_history["OA"].append(OA)
    metrics_history["Precision"].append(Precision)
    metrics_history["Recall"].append(Recall)
    metrics_history["Kappa"].append(Kappa)
    metrics_history["F1"].append(F1)


def calc_auc(d_np, gt):
    d_np = d_np.flatten()
    gt = gt.flatten().astype(int)
    fpr, tpr, _ = roc_curve(gt, d_np)
    roc_auc = auc(fpr, tpr)
    return roc_auc

