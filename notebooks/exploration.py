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
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# %%
# automatically reloads imported packages when changes are made to them
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# #### ideas / todos

# %%
# UP NEXT
# models

# %%
# potential additions
# - check data types, everywhere
# - expand on text cleanup, e.g. filter stopwords, special chars, multi-space
# - classify
# - target embeddings, difference of text_emb - target_emb * classes etc.
# - weighted embeds, e.g. 0.7*title, 0.3*desc

# %%
# extra
# create documentation
# proper tests

# %%
# data checking
# - created_on : date_time
# ...

# %% [markdown]
# #### running 

# %%
from text_classifier.data.data import data_pipeline
from text_classifier.data.data import get_model_data

# %%
df = data_pipeline()
df.head()

# %%
scaler, encoder, X_train, X_test, y_train, y_test = get_model_data()
X_train.head()

# %%
# modeling

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# %%
model = XGBClassifier(
    objective="multi:softprob",   # multiclass classification
    num_class=4,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)


# %%
def print_pred_report(
    y: np.typing.ArrayLike,
    y_pred: np.typing.ArrayLike
):
    print(f"Accuracy: {accuracy_score(y, y_pred)}\n")

    print("Classification Report")
    print(classification_report(y, y_pred))

    print("Confusion Matrix")
    print(confusion_matrix(y, y_pred))


# %%
model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)
print_pred_report(y_test, y_pred)

# %%
y_train_pred = model.predict(X_train)
print_pred_report(y_train, y_train_pred)

# %% [markdown]
# #### embeddings experiments

# %%
from text_classifier.data.data import get_raw_dataset

def get_test_df():
    df = get_raw_dataset()
    df = df.drop(["created_on", "description", "tag"], axis=1)

    return df[:10]


# %%
def save_test_df(df: pd.DataFrame):
    df.to_pickle("test_df")


# %%
def lets_keep_u_for_now(
        model_str: str = "all-MiniLM-L6-v2"
    ):
    test_df: pd.DataFrame = pd.read_pickle("test_df")
    test_df = test_df.set_index("id")
    test_df.loc[16] = "helloo"
    # test_df.loc[15] = "helooooooo"
    test_df.loc[18] = "helooooooo"

    cache = np.load("test_embeddings.npz")
    cache_embeddings = cache["embeddings"]
    cache_ids = cache["ids"]

    embedding_df = pd.DataFrame(
        cache_embeddings,
        index=cache_ids,
        columns=[f"embedding_{i}" for i in range(cache_embeddings.shape[1])]
    )

    missing_df = test_df.loc[
        ~test_df.index.isin(embedding_df.index)
    ]

    model = SentenceTransformer(model_str)
    
    new_embeddings = model.encode(
        missing_df["title"].tolist(),
        convert_to_numpy=True
    )

    new_embedding_df = pd.DataFrame(
        new_embeddings,
        index=missing_df.index,
        columns=embedding_df.columns
    )

    result_df = test_df.combine_first(embedding_df).combine_first(new_embedding_df)

    return result_df


# %%
from text_classifier.data.embeddings import add_text_embeddings

def test_add_text_embeddings(file_path: Path = Path("test_embeddings.npz")):
    test_df: pd.DataFrame = pd.read_pickle("test_df")
    test_df = test_df.set_index("id")
    # test_df.loc[16] = "helloo"
    test_df.loc[192] = "helloooo"

    # result = add_text_embeddings2(test_df, "title")
    result = add_text_embeddings(test_df, "title", file_path)

    return result


# %% [markdown]
# #### exploration

# %%
df["created_on"]


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
