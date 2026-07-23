import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataset(dataset_path: str = "datasets/dataset.csv") -> pd.DataFrame :
    return pd.read_csv(dataset_path)

def get_train_test_df(
        df: pd.DataFrame,
        test_size: float = 0.2,
        seed: int = 1234
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df: pd.DataFrame
    test_df: pd.DataFrame

    train_df, test_df = train_test_split(
        df,
        stratify=df.tag,
        test_size=test_size,
        random_state=seed
    )
    
    return (train_df, test_df)

def hello_from_data(a_string_pls: str) -> str:
    print("helloo i was in dataa")
    print(f"also your string: {a_string_pls}")

    return "we done did it"
