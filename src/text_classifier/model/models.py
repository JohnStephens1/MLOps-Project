from typing import Any, Mapping
from xgboost import XGBClassifier
from scipy.stats import randint, uniform, loguniform  # type: ignore


def get_xgboost_param_distribution() -> dict[str, Any]:
    return {
        # Tree complexity
        "model__max_depth": randint(3, 11),
        "model__min_child_weight": randint(1, 10),
        # Learning
        "model__learning_rate": loguniform(1e-3, 3e-1),
        "model__n_estimators": randint(100, 1000),
        # Row/column sampling
        "model__subsample": uniform(0.5, 0.5),  # 0.5 - 1.0
        "model__colsample_bytree": uniform(0.5, 0.5),  # 0.5 - 1.0
        # Regularization
        "model__gamma": uniform(0, 5),
        "model__reg_alpha": loguniform(1e-4, 10),
        "model__reg_lambda": loguniform(1e-3, 100),
    }


def get_model_XGBClassifier(
    model_params: Mapping[str, Any] | None = None,
) -> XGBClassifier:
    if model_params is None:
        model_params = {}

    model = XGBClassifier(**model_params)

    return model
