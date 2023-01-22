from typing import Dict

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def get_classification_metrics(y_true: np.array, y_pred: np.array) -> Dict[str, str]:
    """Compute classification metrics:
        Balanced accuracy, F1-Score, Precision and Recall.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary of classification performance metrics.
    """
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }


def multi_label_balanced_accuracy(y_true: np.array, y_pred: np.array) -> float:
    """Compute mean balanced accuracy for multiple labels.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Mean balanced accuracy across all labels.

    """
    return np.mean(
        [
            balanced_accuracy_score(y_true[:, i], y_pred[:, i])
            for i in range(y_true.shape[1])
        ]
    )
