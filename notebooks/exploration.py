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
# import tensorflow as tf
# import matplotlib.pyplot as plt
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
# #### definitions

# %% [markdown]
# ##### dataprep

# %%
def preprocess_string(string: str) -> str:
    string = string.lower().strip()

    return string


# %%
def get_raw_dataset(ds_path: Path = Path("../datasets/dataset.csv")) -> pd.DataFrame:
    return pd.read_csv(ds_path)


# %%
def get_train_test_df(
    df: pd.DataFrame = get_raw_dataset(),
    test_size: float = 0.2,
    seed: int = 1234
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        stratify=df.tag,
        test_size=test_size,
        random_state=seed
    )

    return train_df, test_df


# %%
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates()

    string_cols = df.select_dtypes(include=["object", "string"]).columns
    df[string_cols] = df[string_cols].apply(lambda col: col.apply(preprocess_string))

    # could filter stopwords, special chars, multi-space
    # one hot encoding for tag
    # df.tag = 
    # bert for text

    return df


# %%
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df.title + " " + df.description

    return df


# %%
def data_pipeline() -> pd.DataFrame:
    df = get_raw_dataset()
    df = preprocess_data(df)
    # df = add_features(df)

    return df


# %% [markdown]
# #### running 

# %%
def testing_smth(
        df: pd.DataFrame,
        target_cols: list[str] = ["title", "description", "tag"]
    ) -> pd.DataFrame:
    # modify title, description, tag
    # created_on to date_time
    # tag to target ; 0 1 2 3

    target_cols = ["title", "description", "tag"]

    # string_cols = df.select_dtypes(include=["object", "string"]).columns
    df[target_cols] = df[target_cols].apply(lambda col: col.apply(preprocess_string))

    return df


df = data_pipeline()
df = testing_smth(df)
df.head()

# %% [markdown]
# #### exploration

# %%
pd.get_dummies(df.tag)

# %%
train_df, test_df = get_train_test_df()

# %%
test_df.shape

# %%
df.shape

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
