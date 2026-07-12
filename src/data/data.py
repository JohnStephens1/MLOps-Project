import pandas as pd


def get_dataset(dataset_path: str = "datasets/dataset.csv") -> pd.DataFrame :
    return pd.read_csv(dataset_path)
