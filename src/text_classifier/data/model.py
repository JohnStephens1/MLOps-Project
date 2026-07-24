import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from text_classifier.data.data import data_pipeline, get_raw_dataset


def get_pre_pipe_model_data() -> pd.DataFrame:
    df = get_raw_dataset()
    df = data_pipeline(df)
    df = df.drop(["created_on", "title", "description", "text"], axis=1)

    return df


def get_X_y(
    df: pd.DataFrame, target_col: str = "tag"
) -> tuple[pd.DataFrame, pd.Series]:
    """Splits target_col off of df, then returns the remaining dataset and the isolated column

    Args:
        df (pd.DataFrame): df
        target_col (str, optional): the name of the column to be split off. Defaults to "tag".

    Returns:
        tuple[pd.DataFrame, pd.Series]: the remaining dataset and the isolated column
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return X, y


def get_encoded_y(y: np.typing.ArrayLike) -> tuple[LabelEncoder, np.typing.ArrayLike]:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return label_encoder, y_encoded


def get_train_test_df(
    X: pd.DataFrame, y: np.typing.ArrayLike, test_size: float = 0.2, seed: int = 1234
) -> tuple[pd.DataFrame, pd.DataFrame, np.typing.ArrayLike, np.typing.ArrayLike]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=seed
    )

    return X_train, X_test, y_train, y_test


def get_preprocessor(
    target_cols: list[str] = ["days_since_start"],
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("scaler", MinMaxScaler(), target_cols),
        ],
        remainder="passthrough",
    )


def get_encoder_train_test_df() -> tuple[
    LabelEncoder, pd.DataFrame, pd.DataFrame, np.typing.ArrayLike, np.typing.ArrayLike
]:
    df = get_pre_pipe_model_data()
    X, y = get_X_y(df)

    label_encoder, y_encoded = get_encoded_y(y)
    X_train, X_test, y_train, y_test = get_train_test_df(X, y_encoded)

    return label_encoder, X_train, X_test, y_train, y_test
