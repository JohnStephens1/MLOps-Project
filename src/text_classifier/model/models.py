from typing import Any, Mapping
from xgboost import XGBClassifier
from scipy.stats import randint, uniform, loguniform  # type: ignore


def get_xgboost_param_distribution() -> dict[str, Any]:
    return {
        # Tree complexity
        "max_depth": randint(3, 11),
        "min_child_weight": randint(1, 10),
        # Learning
        "learning_rate": loguniform(1e-3, 3e-1),
        "n_estimators": randint(100, 1000),
        # Row/column sampling
        "subsample": uniform(0.5, 0.5),  # 0.5 - 1.0
        "colsample_bytree": uniform(0.5, 0.5),  # 0.5 - 1.0
        # Regularization
        "gamma": uniform(0, 5),
        "reg_alpha": loguniform(1e-4, 10),
        "reg_lambda": loguniform(1e-3, 100),
        # Optional
        "scale_pos_weight": uniform(0.5, 4.5),  # useful if classes are imbalanced
    }


def get_model_XGBClassifier(
    model_params: Mapping[str, Any] | None = None,
) -> XGBClassifier:
    if model_params is None:
        model_params = {}

    model = XGBClassifier(**model_params)

    return model
