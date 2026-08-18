"""Maximal set of classification metrics for a multi-class problem (5 classes)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    roc_auc_score,
    log_loss,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
)


def compute_all_metrics(y_true, y_pred, y_proba, class_names) -> dict:
    """y_proba: (N, num_classes) softmax probabilities."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    # renormalize: softmax outputs sum to ~1 but float32->list round-trip can
    # drift enough to trip sklearn's strict "probabilities must sum to 1" check
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    labels = list(range(len(class_names)))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = (
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=labels)
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba, labels=labels),
        "top2_accuracy": top_k_accuracy_score(y_true, y_proba, k=2, labels=labels),
    }

    try:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=labels
        )
        metrics["roc_auc_ovo_macro"] = roc_auc_score(
            y_true, y_proba, multi_class="ovo", average="macro", labels=labels
        )
    except ValueError:
        pass  # can fail if a class is missing from a small validation split

    for i, name in enumerate(class_names):
        key = name.lower().replace(" ", "_")
        metrics[f"precision_{key}"] = precision_per_class[i]
        metrics[f"recall_{key}"] = recall_per_class[i]
        metrics[f"f1_{key}"] = f1_per_class[i]

    return metrics


def confusion_matrix_and_report(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0
    )
    return cm, report
