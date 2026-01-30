from typing import Dict, Any, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


def evaluate_model(model, dataloader, device) -> Dict[str, Any]:
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return metrics


def compute_micro_roc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray, float],
           Tuple[np.ndarray, np.ndarray, float]]:
    """Returns (fpr, tpr, roc_auc), (precision, recall, pr_auc) micro-averaged."""
    classes = np.arange(num_classes)
    y_true_bin = label_binarize(y_true, classes=classes)

    # ROC
    fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc = auc(fpr, tpr)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(
        y_true_bin.ravel(), y_prob.ravel()
    )
    pr_auc = average_precision_score(y_true_bin, y_prob, average="micro")

    return (fpr, tpr, roc_auc), (precision, recall, pr_auc)
