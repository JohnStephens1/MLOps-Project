import pandas as pd

from pathlib import Path
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
