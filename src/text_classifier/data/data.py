import pandas as pd

from pathlib import Path
from text_classifier.data.features import add_features
from text_classifier.data.preprocessing import preprocess_data
from text_classifier import config


def get_raw_dataset(ds_path: Path | None = None) -> pd.DataFrame:
    """gets the raw dataset

    Args:
        ds_path (Path, optional): path to the dataset. Defaults to DATASET_PATH.

    Returns:
        pd.DataFrame: the loaded df
    """
    if ds_path is None:
        ds_path = config.DATASET_PATH
    
    return pd.read_csv(ds_path)


def data_pipeline() -> pd.DataFrame:
    """preprocesses the data and adds features to the dataset

    Returns:
        pd.DataFrame: the prepared df
    """
    df = get_raw_dataset()
    df = df.set_index("id")

    df = preprocess_data(df)
    df = add_features(df)

    return df
