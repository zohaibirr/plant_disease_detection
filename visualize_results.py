import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.metrics import compute_micro_roc_pr


def plot_training_curves(history, figures_dir: str, prefix: str):
    os.makedirs(figures_dir, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix} - Training Loss")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_loss_curve.png"))
    plt.close()

    # Val accuracy
    plt.figure()
    plt.plot(np.array(history["val_accuracy"]) * 100, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{prefix} - Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_accuracy_curve.png"))
    plt.close()


def plot_confusion_matrix(cm, class_names, figures_dir: str, prefix: str):
    os.makedirs(figures_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{prefix} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_confusion_matrix.png"))
    plt.close()


def plot_roc_pr_curves(y_true, y_prob, num_classes, figures_dir: str, prefix: str):
    os.makedirs(figures_dir, exist_ok=True)

    (fpr, tpr, roc_auc), (precision, recall, pr_auc) = compute_micro_roc_pr(
        y_true, y_prob, num_classes
    )

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"Micro-avg ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{prefix} - ROC Curve (Micro-avg)")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_roc_curve_micro.png"))
    plt.close()

    # Precision-Recall
    plt.figure()
    plt.plot(recall, precision, label=f"Micro-avg PR (AP = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{prefix} - Precision-Recall Curve (Micro-avg)")
    plt.legend()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_precision_recall_curve_micro.png"))
    plt.close()
