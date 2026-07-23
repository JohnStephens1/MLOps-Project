import numpy as np
import pandas as pd
import text_classifier.data.model

from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from text_classifier.data.features import add_features
from text_classifier.data.preprocessing import preprocess_data
from text_classifier.config import DATASET_PATH


def get_raw_dataset(ds_path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(ds_path)


def data_pipeline(
    df: pd.DataFrame = get_raw_dataset(),
    time_col: str = "created_on"
) -> pd.DataFrame:
    df = df.set_index("id")
    df = preprocess_data(df)
    df = add_features(df, time_col)

    return df


def get_model_data(
    df: pd.DataFrame = get_raw_dataset(),
    time_col: str = "created_on",
    target_col: str = "tag"
) -> tuple[
    MinMaxScaler, LabelEncoder, pd.DataFrame, pd.DataFrame, np.typing.ArrayLike, np.typing.ArrayLike
]:
    return text_classifier.data.model.get_model_data(
        df,
        time_col,
        target_col
    )
