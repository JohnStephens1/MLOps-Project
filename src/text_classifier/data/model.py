import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from text_classifier.data.data import data_pipeline, get_raw_dataset
from text_classifier.model.model import get_model_pipe


# OLD

# def get_train_test_df(
#     df: pd.DataFrame, target_col: str = "tag", test_size: float = 0.2, seed: int = 1234
# ) -> tuple[pd.DataFrame, pd.DataFrame, np.typing.ArrayLike, np.typing.ArrayLike]:
#     """Gets train and test dataframes

#     Args:
#         df (pd.DataFrame): the dataframe
#         target_col (str, optional): the name of the target column. Defaults to "tag".
#         test_size (float, optional): the proportion of the dataset to be used as the test set. Defaults to 0.2.
#         seed (int, optional): the random seed. Defaults to 1234.

#     Returns:
#         tuple[pd.DataFrame, pd.DataFrame, np.typing.ArrayLike, np.typing.ArrayLike]: X_train, X_test, y_train, y_test. the train and test dataframes
#     """
#     X, y = get_X_y(df, target_col)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, stratify=y, test_size=test_size, random_state=seed
#     )

#     return X_train, X_test, y_train, y_test


# def get_scaled_target(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     range: tuple[int, int] = (0, 1),
#     target_col: str = "days_since_start",
# ) -> tuple[MinMaxScaler, pd.DataFrame, pd.DataFrame]:
#     """scales the target_col in input dataframes via MinMaxScaling

#     Args:
#         X_train (pd.DataFrame): X_train df
#         X_test (pd.DataFrame): X_test df
#         range (tuple[int, int], optional): the range of the scaler. Defaults to (0, 1).
#         target_col (str, optional): the name of the target column. Defaults to "days_since_start".

#     Returns:
#         tuple[MinMaxScaler, pd.DataFrame, pd.DataFrame]: scaler, X_train, X_test
#     """
#     scaler = MinMaxScaler(range)

#     X_train[target_col] = scaler.fit_transform(X_train[[target_col]])
#     X_test[target_col] = scaler.transform(X_test[[target_col]])

#     return scaler, X_train, X_test


# def get_encoded_target(
#     y_train: np.typing.ArrayLike, y_test: np.typing.ArrayLike
# ) -> tuple[LabelEncoder, np.typing.ArrayLike, np.typing.ArrayLike]:
#     """encodes y_train, y_test via LabelEncoder

#     Args:
#         y_train (np.typing.ArrayLike): y_train array
#         y_test (np.typing.ArrayLike): y_test array

#     Returns:
#         tuple[LabelEncoder, np.typing.ArrayLike, np.typing.ArrayLike]: encoder, y_train, y_test
#     """
#     encoder = LabelEncoder()

#     y_train = encoder.fit_transform(y_train)
#     y_test = encoder.transform(y_test)

#     return encoder, y_train, y_test


# class PreparedData(TypedDict):
#     scaler: MinMaxScaler
#     encoder: LabelEncoder
#     X_train: pd.DataFrame
#     X_test: pd.DataFrame
#     y_train: np.typing.ArrayLike
#     y_test: np.typing.ArrayLike


# def get_model_data(
#     df: pd.DataFrame = get_raw_dataset(),
#     time_col: str = "created_on",
#     target_col: str = "tag",
# ) -> tuple[
#     MinMaxScaler,
#     LabelEncoder,
#     pd.DataFrame,
#     pd.DataFrame,
#     np.typing.ArrayLike,
#     np.typing.ArrayLike,
# ]:
#     """gets the data for model training and eval, as well as the corresponding scaler and encoder

#     Args:
#         df (pd.DataFrame, optional): raw df. Defaults to get_raw_dataset().
#         time_col (str, optional): the name of the time column. Defaults to "created_on".
#         target_col (str, optional): the name of the target column. Defaults to "tag".

#     Returns:
#         tuple[MinMaxScaler, LabelEncoder, pd.DataFrame, pd.DataFrame, np.typing.ArrayLike, np.typing.ArrayLike]: scaler, encoder, X_train, X_test, y_train, y_test
#     """
#     df = data_pipeline(df, time_col)

#     df = df.drop([time_col, "title", "description", "text"], axis=1)

#     X_train, X_test, y_train, y_test = get_train_test_df(df, target_col)
#     scaler, X_train, X_test = get_scaled_target(X_train, X_test)
#     encoder, y_train, y_test = get_encoded_target(y_train, y_test)

#     return scaler, encoder, X_train, X_test, y_train, y_test
# return {
#     "scaler": scaler,
#     "encoder": encoder,
#     "X_train": X_train,
#     "X_test": X_test,
#     "y_train": y_train,
#     "y_test": y_test,
# }


# NEW


def get_pre_pipe_model_data():
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


def get_preprocessor(target_cols: list[str] = ["days_since_start"]):
    return ColumnTransformer(
        [
            ("scaler", MinMaxScaler(), target_cols),
        ],
        remainder="passthrough",
    )


def nother_df_fn():
    df = get_pre_pipe_model_data()

    X, y = get_X_y(df)
    label_encoder, y_encoded = get_encoded_y(y)

    X_train, X_test, y_train, y_test = get_train_test_df(X, y_encoded)

    pipe = get_model_pipe()


# post pipe

# potentially store encoder with pipe in dict
# "joblib to dump model"
