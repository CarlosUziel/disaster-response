from typing import Dict

import numpy as np
import pandas as pd
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


def get_best_cv_indx(cv_results: np.array) -> int:
    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.sort_values(
        [
            "mean_test_f1_weighted",
            "mean_test_precision_weighted",
            "mean_test_recall_weighted",
        ],
        ascending=False,
        inplace=True,
    )
    return cv_results_df.index.to_list()[0]
