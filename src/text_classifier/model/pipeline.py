from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler


def get_preprocessor(
    target_cols: list[str] = ["days_since_start"],
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("scaler", MinMaxScaler(), target_cols),
        ],
        remainder="passthrough",
    )


def get_model_pipe(estimator: BaseEstimator) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", get_preprocessor()),
            ("model", estimator),
        ]
    )
