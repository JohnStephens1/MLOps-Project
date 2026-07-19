# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: mlops-project (3.12.12)
#     language: python
#     name: python3
# ---

# %% [markdown]
# #### imports

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# %% [markdown]
# #### ideas / todos

# %%
# steps to greatness
# load data into pd
# preprocess
# - check data types
# - filter trash

# feature engineering update

# potential class for data processing

# model

# graphing

# %% [markdown]
# data checking
# - right format for
#     - created_on : date_time
#

# %% [markdown]
# #### definitions

# %%
def get_raw_dataset(ds_path: Path = Path("../datasets/dataset.csv")) -> pd.DataFrame:
    return pd.read_csv(ds_path)


# %% [markdown]
# ##### data preprocessing

# %%
def preprocess_string(string: str) -> str:
    string = string.lower().strip()

    return string


# %%
def preprocess_text(
        df: pd.DataFrame,
        target_cols: list[str] = ["title", "description", "tag"]
    ) -> pd.DataFrame:
    # TODO
    # could filter stopwords, special chars, multi-space
    # bert for text

    df[target_cols] = df[target_cols].apply(lambda col: col.apply(preprocess_string))

    return df


# %%
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()

    df['created_on'] = pd.to_datetime(df['created_on'])
    df = preprocess_text(df)

    return df


# %% [markdown]
# ##### feature engineering

# %%
def set_sin_cos_features(df: pd.DataFrame, result_col_name: str, input_col: pd.Series, time_span: int) -> pd.DataFrame:
    df[result_col_name + "_sin"] = np.sin(2 * np.pi * input_col / time_span)
    df[result_col_name + "_cos"] = np.cos(2 * np.pi * input_col / time_span)

    return df


# %%
def add_time_features(df: pd.DataFrame, time_col: str = "created_on") -> pd.DataFrame:
    # hour / 24 - cyclic
    df = set_sin_cos_features(df, "hour_of_day", df[time_col].dt.hour, 24)

    # day / 7 - cyclic
    df = set_sin_cos_features(df, "day_of_week", df[time_col].dt.day_of_week, 7)

    # month / 12 - cyclic
    df = set_sin_cos_features(df, "month_of_year", df[time_col].dt.month, 12)

    # # weekend?
    df["is_weekend"] = df[time_col].dt.day_of_week >= 5

    # days since start
    df["days_since_start"] = (df[time_col] - df[time_col].min()).dt.days

    return df


# %%
def add_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df["text"] = df.title + " " + df.description
    df = add_time_features(df, time_col)

    return df


# %%
def data_pipeline(
    df: pd.DataFrame = get_raw_dataset(),
    time_col: str = "created_on"
) -> pd.DataFrame:
    df = preprocess_data(df)
    df = add_features(df, time_col)

    return df


# %% [markdown]
# ##### model data

# %%
def get_X_y(
    df: pd.DataFrame,
    target_col: str = "tag"
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y


# %%
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


# %%
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


# %%
def get_encoded_target(
    y_train: np.typing.ArrayLike,
    y_test: np.typing.ArrayLike
) -> tuple[LabelEncoder, np.typing.ArrayLike, np.typing.ArrayLike]:
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    
    return encoder, y_train, y_test


# %%
def get_model_data(
    df: pd.DataFrame = get_raw_dataset(),
    time_col: str = "created_on",
    target_col: str = "tag"
) -> tuple[MinMaxScaler, LabelEncoder, pd.DataFrame, pd.DataFrame, np.typing.ArrayLike, np.typing.ArrayLike]:
    df = data_pipeline(df, time_col)
    
    df = df.drop(['id', time_col], axis=1)
    
    X_train, X_test, y_train, y_test = get_train_test_df(df, target_col)
    scaler, X_train, X_test = get_scaled_target(X_train, X_test)
    encoder, y_train, y_test = get_encoded_target(y_train, y_test)
    
    return scaler, encoder, X_train, X_test, y_train, y_test


# %% [markdown]
# #### running 

# %%
df = data_pipeline()
df.head()

# %%
scaler, encoder, X_train, X_test, y_train, y_test = get_model_data()
X_train.head()


# %% [markdown]
# #### exploration

# %%
def show_created_on_distribution():
    df["created_on"].hist(bins=50)
    plt.title("Created On")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.show()

show_created_on_distribution()


# %%
def show_entries_by_weekday():
    weekday_counts = (
        df["created_on"]
        .dt.day_name()
        .value_counts()
        .reindex([
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ])
    )

    weekday_counts.plot(kind="bar")
    plt.title("Entries by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Number of entries")
    plt.show()

show_entries_by_weekday()

# %%
df.head()

# %%
df.columns
# 5 columns, named
# ['id', 'created_on', 'title', 'description', 'tag']

# %%
df.dtypes
# all of type str except id

# %%
df.describe(include='all')
# tag has 4 unique values

# %%
set(df.tag)
# the unique values are
# {'computer-vision', 'mlops', 'natural-language-processing', 'other'}

# %%
df.isna().sum()
# no missing values
