import pandas as pd


def preprocess_string(string: str) -> str:
    string = string.lower().strip()

    return string


def preprocess_text(
        df: pd.DataFrame,
        target_cols: list[str] = ["title", "description", "tag"]
    ) -> pd.DataFrame:

    df[target_cols] = df[target_cols].apply(lambda col: col.apply(preprocess_string))

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()

    df['created_on'] = pd.to_datetime(df['created_on'])
    df = preprocess_text(df)

    return df
