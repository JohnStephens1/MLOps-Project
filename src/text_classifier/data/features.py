import numpy as np
import pandas as pd

from text_classifier.data.embeddings import add_text_embeddings


def set_sin_cos_features(df: pd.DataFrame, result_col_name: str, input_col: pd.Series, time_span: int) -> pd.DataFrame:
    """adds cos and sin time series features to the df

    Args:
        df (pd.DataFrame): target df
        result_col_name (str): the name of the new feature column
        input_col (pd.Series): the input column to calculate for
        time_span (int): the length of the time series

    Returns:
        pd.DataFrame: the modified df with added time series features
    """
    df[result_col_name + "_sin"] = np.sin(2 * np.pi * input_col / time_span)
    df[result_col_name + "_cos"] = np.cos(2 * np.pi * input_col / time_span)

    return df


def add_time_features(df: pd.DataFrame, time_col: str = "created_on") -> pd.DataFrame:
    """adds time series features to the df

    Args:
        df (pd.DataFrame): input df
        time_col (str, optional): column containing time data. Defaults to "created_on".

    Returns:
        pd.DataFrame: the modified df with added time series features 
    """
    # hour / 24 - cyclic
    df = set_sin_cos_features(df, "hour_of_day", df[time_col].dt.hour, 24)

    # day / 7 - cyclic
    df = set_sin_cos_features(df, "day_of_week", df[time_col].dt.day_of_week, 7)

    # month / 12 - cyclic
    df = set_sin_cos_features(df, "month_of_year", df[time_col].dt.month, 12)

    # weekend?
    df["is_weekend"] = df[time_col].dt.day_of_week >= 5

    # days since start
    df["days_since_start"] = (df[time_col] - df[time_col].min()).dt.days

    return df


def add_features(df: pd.DataFrame, time_col: str, text_col: str = "text") -> pd.DataFrame:
    """adds features to the dataset, specifically a text column, time series features and text embeddings

    Args:
        df (pd.DataFrame): input df
        time_col (str): name of the column containing time data
        text_col (str, optional): name of the created text column. Defaults to "text".

    Returns:
        pd.DataFrame: modified df with added features
    """
    df[text_col] = df.title + " " + df.description
    df = add_time_features(df, time_col)
    df = add_text_embeddings(df, text_col)

    return df
