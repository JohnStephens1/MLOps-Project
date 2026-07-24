import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def get_classification_metrics(
    y_true: np.typing.ArrayLike,
    y_pred: np.typing.ArrayLike,
    y_proba: np.typing.ArrayLike | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(
            y_true, y_proba, multi_class="ovo", average="macro"
        )

    return metrics


def print_pred_report(
    y_true: np.typing.ArrayLike,
    y_pred: np.typing.ArrayLike,
    y_proba: np.typing.ArrayLike | None = None,
):
    metrics = get_classification_metrics(y_true, y_pred, y_proba)
    
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1: {metrics['f1']}")

    if y_proba is not None:
        print(f"ROC AUC: {metrics['roc_auc']}")

    print()

    print(f"Classification Report:\n{classification_report(y_true, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    