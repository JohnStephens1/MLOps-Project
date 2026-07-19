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
encoder, X_train, X_test, y_train, y_test = get_model_data()

# %%
X_test.head()

# %%
print(y_test)

# %% [markdown]
# #### exploration

# %%
# created on distribution
df["created_on"].hist(bins=50)
plt.title("Created On")
plt.xlabel("Date")
plt.ylabel("Count")
plt.show()

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
