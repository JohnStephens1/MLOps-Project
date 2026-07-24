


from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from text_classifier.data.model import get_preprocessor


def get_model_pipe() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", get_preprocessor()),
            ("model", XGBClassifier()),
        ]
    )
