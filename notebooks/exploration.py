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
# data checking
# - right format for
#     - created_on : date_time
#

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
def preprocess_text(
        df: pd.DataFrame,
        target_cols: list[str] = ["title", "description", "tag"]
    ) -> pd.DataFrame:
    # TODO
    # could filter stopwords, special chars, multi-space
    # tag to target ; 0 1 2 3
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


# %%
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # TODO
    # add type features?
        # df["year"] = df["created_on"].dt.year
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
df = data_pipeline()
df.head()

# %%
df.created_on[0]

# %%
df_raw = get_raw_dataset()
df_raw.created_on


# %%
pd.to_datetime(df_raw.created_on[0])

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
