from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from text_classifier.data.model import get_preprocessor


def get_model_pipe(estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", get_preprocessor()),
            ("model", estimator),
        ]
    )
