import pandas as pd

from pathlib import Path
from text_classifier.data.features import add_features
from text_classifier.data.preprocessing import preprocess_data
from text_classifier.config import DATASET_PATH


def get_raw_dataset(ds_path: Path = DATASET_PATH) -> pd.DataFrame:
    """gets the raw dataset

    Args:
        ds_path (Path, optional): path to the dataset. Defaults to DATASET_PATH.

    Returns:
        pd.DataFrame: the loaded df
    """
    return pd.read_csv(ds_path)


def data_pipeline(
    df: pd.DataFrame = get_raw_dataset(),
    time_col: str = "created_on"
) -> pd.DataFrame:
    """preprocesses the data and adds features to the dataset

    Args:
        df (pd.DataFrame, optional): target df. Defaults to get_raw_dataset().
        time_col (str, optional): name of the column with time data. Defaults to "created_on".

    Returns:
        pd.DataFrame: the prepared df
    """
    df = df.set_index("id")
    df = preprocess_data(df)
    df = add_features(df, time_col)

    return df
