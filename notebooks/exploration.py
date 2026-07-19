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
from sklearn.preprocessing import LabelEncoder
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
def add_time_features(df: pd.DataFrame, target_col: str = "created_on") -> pd.DataFrame:
    # year is always 2020
    # TODO
    # add type features?
        # df["year"] = df["created_on"].dt.year
    # add age
    # df["age"] = df[target_col].
    # add cyclical features?
        # df["hour_sin"] = np.sin(2*np.pi*df["created_on"].dt.hour/24)
        # df["hour_cos"] = np.cos(2*np.pi*df["created_on"].dt.hour/24)

        # df["month_sin"] = np.sin(2*np.pi*df["created_on"].dt.month/12)
        # df["month_cos"] = np.cos(2*np.pi*df["created_on"].dt.month/12)
    # add age feature?

    return df


# %%
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df.title + " " + df.description
    df = add_time_features(df)

    return df


# %%
def data_pipeline(
    df: pd.DataFrame = get_raw_dataset()
) -> pd.DataFrame:
    df = preprocess_data(df)
    # df = add_features(df)

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
    target_col: str = "tag"
) -> tuple[LabelEncoder, pd.DataFrame, pd.DataFrame, np.typing.ArrayLike, np.typing.ArrayLike]:
    df = data_pipeline(df)
    
    X_train, X_test, y_train, y_test = get_train_test_df(df, target_col)
    encoder, y_train, y_test = get_encoded_target(y_train, y_test)
    
    return encoder, X_train, X_test, y_train, y_test


# %% [markdown]
# #### running 

# %%
df = data_pipeline()
df.head()

# %%
# max 2020
df.created_on.dt.year.min()


# %%
def set_sin_cos_features(df: pd.DataFrame, result_col_name: str, input_col: pd.Series, time_span: int) -> pd.DataFrame:
    df[result_col_name + "_sin"] = np.sin(2 * np.pi * input_col / time_span)
    df[result_col_name + "_cos"] = np.cos(2 * np.pi * input_col / time_span)

    return df


# %%
df = data_pipeline()

# hour / 24 - cyclic
df = set_sin_cos_features(df, "hour_per_day", df["created_on"].dt.day_of_week, 7)
# day / 7 - cyclic
# day of week / 7? - cyclic
# # weekend?

# day of week / 7? - cyclic
df["day_of_week_sin"] = df["created_on"].dt.day_of_week

# days since start
df["days_since_start"] = (
    df["created_on"] - df["created_on"].min()
).dt.days

# %%
df.head()

# %%
df["days_since_start"]

# %%
encoder, X_train, X_test, y_train, y_test = get_model_data()

# %%
X_test.head()

# %%
print(y_test)


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
