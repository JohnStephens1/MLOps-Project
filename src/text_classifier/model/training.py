import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline


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


# GridSearchCV
# RandomizedSearchCV


def train_qm():
    pass


# get pipe
# get datasets + encoder
# get model
