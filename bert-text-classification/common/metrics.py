"""Core classification metrics for the 5-class problem:
accuracy; precision / recall / F1 in macro, micro, weighted averages;
ROC-AUC (macro, one-vs-rest) and PR-AUC / average precision (macro, OVR).
Per-class P/R/F1 remain in the classification_report text (logged to Comet)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def compute_all_metrics(y_true, y_pred, y_proba, class_names) -> dict:
    """y_proba: (N, num_classes) softmax probabilities."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    # renormalize: float32->list round-trip can drift enough to trip
    # sklearn's "probabilities must sum to 1" checks
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    labels = list(range(len(class_names)))

    metrics = {"accuracy": accuracy_score(y_true, y_pred)}

    for average in ("macro", "micro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0, labels=labels
        )
        metrics[f"precision_{average}"] = p
        metrics[f"recall_{average}"] = r
        metrics[f"f1_{average}"] = f

    try:
        metrics["roc_auc_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=labels
        )
        metrics["pr_auc_macro"] = average_precision_score(
            y_true, y_proba, average="macro"
        )
    except ValueError:
        pass  # can fail if a class is missing from a small validation split

    return metrics


def confusion_matrix_and_report(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0
    )
    return cm, report
