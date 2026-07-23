import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from text_classifier.data.data import data_pipeline, get_raw_dataset


def get_X_y(
    df: pd.DataFrame,
    target_col: str = "tag"
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y


def get_train_test_df(
    df: pd.DataFrame,
    target_col: str = "tag",
    test_size: float = 0.2,
    seed: int = 1234
) -> tuple[pd.DataFrame, pd.DataFrame, np.typing.ArrayLike, np.typing.ArrayLike]:
    X, y = get_X_y(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=test_size,
        random_state=seed
    )

    return X_train, X_test, y_train, y_test


def get_scaled_target(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    range: tuple[int, int] = (0, 1),
    target_col: str = "days_since_start"
) -> tuple[MinMaxScaler, pd.DataFrame, pd.DataFrame]:
    scaler = MinMaxScaler(range)

    X_train[target_col] = scaler.fit_transform(X_train[[target_col]])
    X_test[target_col] = scaler.transform(X_test[[target_col]])

    return scaler, X_train, X_test


def get_encoded_target(
    y_train: np.typing.ArrayLike,
    y_test: np.typing.ArrayLike
) -> tuple[LabelEncoder, np.typing.ArrayLike, np.typing.ArrayLike]:
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    
    return encoder, y_train, y_test


def get_model_data(
    df: pd.DataFrame = get_raw_dataset(),
    time_col: str = "created_on",
    target_col: str = "tag"
) -> tuple[MinMaxScaler, LabelEncoder, pd.DataFrame, pd.DataFrame, np.typing.ArrayLike, np.typing.ArrayLike]:
    df = data_pipeline(df, time_col)
    
    df = df.drop([time_col, "title", "description", "text"], axis=1)
    
    X_train, X_test, y_train, y_test = get_train_test_df(df, target_col)
    scaler, X_train, X_test = get_scaled_target(X_train, X_test)
    encoder, y_train, y_test = get_encoded_target(y_train, y_test)
    
    return scaler, encoder, X_train, X_test, y_train, y_test
