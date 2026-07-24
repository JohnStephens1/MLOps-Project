from typing import Any, Mapping
from xgboost import XGBClassifier


def get_model_XGBClassifier(model_params: Mapping[str, Any] | None = None) -> XGBClassifier:
    if model_params is None:
        model_params = {}

    model = XGBClassifier(**model_params)

    return model
