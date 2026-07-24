import numpy as np
import pandas as pd
import typing
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from text_classifier.model.pipeline import get_model_pipe
from text_classifier.data.model import get_encoder_train_data
from text_classifier.model.models import (
    get_model_XGBClassifier,
    get_xgboost_param_distribution,
)


def get_cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def get_cv_score(
    pipe: Pipeline, X: pd.DataFrame, y: np.typing.ArrayLike
) -> np.typing.ArrayLike:
    return cross_val_score(
        pipe,
        X,
        y,
        cv=get_cv_splitter(),
        scoring="f1_macro",
        verbose=1,
    )


def get_random_search(
    pipe: Pipeline,
    param_distribution: dict[str, typing.Any],
):
    return RandomizedSearchCV(
        pipe,
        param_distribution,
        n_iter=10,
        cv=get_cv_splitter(),
        scoring="f1_macro",
        random_state=42,
        verbose=1,
    )


# GridSearchCV
# RandomizedSearchCV


def train_qm():
    encoder, train_data = get_encoder_train_data()

    pipe = get_model_pipe(get_model_XGBClassifier())

    param_distribution = get_xgboost_param_distribution()

    search = get_random_search(pipe, param_distribution)

    search.fit(train_data.X_train, train_data.y_train)

    # TODO classify .

    return encoder, train_data, search
