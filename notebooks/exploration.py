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
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pathlib import Path


# %% [markdown]
# #### ideas / todos

# %%
# NEXT
# use ID col as unique identifier, merge with embeddings to create guaranteed association
# implement ids in train test split, or just
# df = df.set_index("id", drop=False)

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

# %%
# potential issue: row change when loading text embeddings

# %%
# data checking
# - created_on : date_time
# ...

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
def get_embeddings(
    df: pd.DataFrame,
    text_col: str,
    model_str: str = "all-MiniLM-L6-v2",
    text_embeddings_path: Path = Path("../data/embeddings/text_embeddings.npy"),
    regenerate: bool = False
) -> np.ndarray:
    if os.path.exists(text_embeddings_path) and not regenerate:
        embeddings = np.load(text_embeddings_path)
    else:
        model = SentenceTransformer(model_str)

        embeddings = model.encode(df[text_col].tolist(), convert_to_numpy=True)
        
        np.save(text_embeddings_path, embeddings)
    
    return embeddings


# %%
def add_text_embeddings(
    df: pd.DataFrame,
    text_col: str
) -> pd.DataFrame:
    embeddings = get_embeddings(df, text_col)
    
    vector_df = pd.DataFrame(
        embeddings,
        columns=[f"{text_col}_{i}" for i in range(embeddings.shape[1])]
    )

    df = pd.concat([df, vector_df], axis=1)

    return df


# %%
def add_features(df: pd.DataFrame, time_col: str, text_col: str = "text") -> pd.DataFrame:
    df[text_col] = df.title + " " + df.description
    df = add_time_features(df, time_col)
    df = add_text_embeddings(df, text_col)

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
    
    df = df.drop(['id', time_col, "title", "description", "text"], axis=1)
    
    X_train, X_test, y_train, y_test = get_train_test_df(df, target_col)
    scaler, X_train, X_test = get_scaled_target(X_train, X_test)
    encoder, y_train, y_test = get_encoded_target(y_train, y_test)
    
    return scaler, encoder, X_train, X_test, y_train, y_test


# %% [markdown]
# #### running 

# %%
# ALL FNs FOR THE FOLLOWING BEAUTY

# %%
def load_ids_embeddings(file_path: Path = Path("test_embeddings.npz")) -> tuple[np.ndarray, np.ndarray]:
    loaded_file = np.load(file_path)

    return loaded_file['ids'], loaded_file['embeddings']


# %%
def get_intersecting_complementing_ids(
    df: pd.DataFrame,
    ids_loaded: np.ndarray
) -> tuple[pd.Index, pd.Index]:
    # all ids in input df and embeddings -> to load
    intersecting_ids = df.index.intersection(ids_loaded.tolist())
    # all ids - intersecting ids -> to generate
    ids_to_generate = df.index.difference(intersecting_ids)

    return intersecting_ids, ids_to_generate


# %%
def get_new_embeddings(
    df: pd.DataFrame,
    ids_to_generate: pd.Index,
    text_col: str,
    model_str: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    model = SentenceTransformer(model_str)
    
    text_to_generate = df.loc[ids_to_generate][text_col].to_list()

    new_embeddings = model.encode(
        text_to_generate,
        convert_to_numpy=True
    )

    return new_embeddings


# %%
def get_embeddings_dic(
        ids_loaded: np.ndarray,
        embeddings_loaded: np.ndarray,
        intersecting_ids: pd.Index,
        ids_to_generate: pd.Index,
        new_embeddings: np.ndarray
) -> dict[pd.Index, np.ndarray]:
    # format id: embedding
    intersecting_ids_dic = {
        id: emb for id, emb in zip(ids_loaded, embeddings_loaded) if id in intersecting_ids
    }
    new_embeddings_dic = dict(zip(ids_to_generate, new_embeddings))
    embeddings_dic = intersecting_ids_dic | new_embeddings_dic

    return embeddings_dic



# %%
def get_embeddings_df(
    embeddings_dic: dict[pd.Index, np.ndarray],
    text_col: str
) -> pd.DataFrame:
    embedding_dim = next(iter(embeddings_dic.values()), np.array([])).shape[0]
    embeddings_df = pd.DataFrame.from_dict(
        embeddings_dic,
        orient="index",
        columns=[f'{text_col}_{i}' for i in range(embedding_dim)]
    )

    return embeddings_df
    


# %%
# u can dance if u want to
def u_can_dance_if_u_want_to(df: pd.DataFrame, text_col: str = "title"):
    ids_loaded, embeddings_loaded = load_ids_embeddings()
    intersecting_ids, ids_to_generate = get_intersecting_complementing_ids(df, ids_loaded)

    new_embeddings = np.array([]) if ids_to_generate.empty else get_new_embeddings(df, ids_to_generate, text_col)
    
    embeddings_dic = get_embeddings_dic(
        ids_loaded,
        embeddings_loaded,
        intersecting_ids,
        ids_to_generate,
        new_embeddings
    )

    embeddings_df = get_embeddings_df(
        embeddings_dic,
        text_col
    )

    # merge by index with input df
    df_result = df.combine_first(embeddings_df)

    return df_result


test_df: pd.DataFrame = pd.read_pickle("test_df")
test_df = test_df.set_index("id")
# test_df.loc[16] = "helloo"
# test_df.loc[192] = "helloooo"

# result = add_text_embeddings2(test_df, "title")
result = u_can_dance_if_u_want_to(test_df, "title")
result

    # embedding_df_loaded = pd.DataFrame(
    #     cached_embeddings,
    #     index=cached_ids,
    #     columns=[f"embedding_{i}" for i in range(cached_embeddings.shape[1])]
    # )


# %%
# questions
# keep id as column or use as index?
# keep embeddings as one col or expand to 384?

# %%
# let's make the id thing happen

# %%
def get_test_df():
    df = get_raw_dataset()
    df = df.drop(["created_on", "description", "tag"], axis=1)

    return df[:10]


# %%
df = get_test_df()


# %%
# save test_df
def save_df(df: pd.DataFrame):
    df.to_pickle("test_df")


# %%
# load test_df
test_df = pd.read_pickle("test_df")

# %%
# smth embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(test_df["title"].tolist(), convert_to_numpy=True)

# np.save(text_embeddings_path, embeddings)


# %%
np.savez(
    "test_embeddings.npz",
    ids=test_df.index.values,
    embeddings=embeddings,
    model="all-MiniLM-L6-v2"
)

# %%
# loading
loaded_test_embeddings = np.load("test_embeddings.npz")

# %%
test_df: pd.DataFrame = pd.read_pickle("test_df")
test_df = test_df.set_index("id")


# %%
def uhh_simple_embeddings_yeah_that() -> pd.DataFrame:
    # nan after
    # ids in df ! in embeddings
    # ids in embeddings ! in df

    # get_test_df
    test_df: pd.DataFrame = pd.read_pickle("test_df")
    test_df = test_df.set_index("id")

    cache = np.load("test_embeddings.npz", allow_pickle=True)
    embeddings_df = pd.DataFrame(
        cache["embeddings"][::2],
        index=cache["ids"][::2],
        columns=[f"embedding_{i}" for i in range(cache["embeddings"][::2].shape[1])]
    )

    # merge_w_test_df_on_index
    result_df = test_df.join(
        embeddings_df
    )

    return result_df

output = uhh_simple_embeddings_yeah_that()


# %%
def lets_b_smart_embeddings() -> pd.DataFrame:
    test_df: pd.DataFrame = pd.read_pickle("test_df")
    test_df = test_df.set_index("id")

    cache = np.load("test_embeddings.npz")
    # cache_embeddings = cache["embeddings"]
    # cache_ids = cache["ids"]
    
    embeddings_df = pd.DataFrame(
        cache["embeddings"],
        index=cache["ids"],
        columns=[f"embedding_{i}" for i in range(cache["embeddings"].shape[1])]
    )

    result_df = test_df.join(
        embeddings_df
    )

    return result_df
    
output = lets_b_smart_embeddings()

# %%
# I NEED TO TEST SMTH ;-;
test_df: pd.DataFrame = pd.read_pickle("test_df")
test_df = test_df.set_index("id")

cache = np.load("test_embeddings.npz")
cache_embeddings = cache["embeddings"]
cache_ids = cache["ids"]

# %%
# hi_im_something = np.array([6, 9, 15, 25, 27, 28, 29, 45, 61])
hi_im_something = [6, 9, 15, 25, 27, 28, 29, 45, 61]

# %%
test_df.index.isin(hi_im_something)

# %%
print(f"test_df_index: {test_df.index}\nother: {hi_im_something}")

# %%
# all ids in dataframe and in emb -> to load
index_intersections = test_df.index.intersection(hi_im_something)
# all ids - result -> to generate
remainder = test_df.index.difference(index_intersections)

# %%
# I NEED TO TEST SMTH ;-;
test_df: pd.DataFrame = pd.read_pickle("test_df")
test_df = test_df.set_index("id")

cache = np.load("test_embeddings.npz")
cache_embeddings = cache["embeddings"]
cache_ids = cache["ids"]
cache_dict = dict(zip(cache_ids, cache_embeddings))

# all ids in dataframe and in emb -> to load
index_intersection = test_df.index.intersection(cache_ids)
# all ids - result -> to generate
remainder = test_df.index.difference(index_intersections)

intersection_emb_dic = {id: emb for id, emb in zip(cache_ids, cache_embeddings) if id in index_intersection}

# embedding_df = pd.DataFrame(
#     cached_embeddings,
#     index=cached_ids,
#     columns=[f"embedding_{i}" for i in range(cached_embeddings.shape[1])]
# )

# %%
def i_give_up():
    test_df: pd.DataFrame = pd.read_pickle("test_df")
    test_df = test_df.set_index("id")

    cache = np.load("test_embeddings.npz")
    cache_embeddings = cache["embeddings"]
    cache_ids = cache["ids"]

    embedding_df = pd.DataFrame(
        cache_embeddings,
        index=cache_ids,
        columns=[f"embedding_{i}" for i in range(cache_embeddings.shape[1])]
    )

    missing_ids = test_df.loc[
        ~test_df.index.isin(embedding_df.index),
        "id"
    ]

    missing_rows = test_df[test_df.index.isin(missing_ids)]

    new_embeddings = model.encode(
        missing_rows["title"].tolist(),
        convert_to_numpy=True
    )

    new_embedding_df = pd.DataFrame(
        new_embeddings,
        index=missing_rows["id"],
        columns=embedding_df.columns
    )

    embedding_df = pd.concat(
        [embedding_df, new_embedding_df]
    )

    embedding_df = embedding_df[~embedding_df.index.duplicated(keep="last")]

    return embedding_df

i_give_up()


# %%
# save and load fun
def save_embeddings(
    ids: np.typing.ArrayLike,
    embeddings: np.typing.ArrayLike,
    file_path: Path = Path("test_embeddings.npz"),
    model: str = "all-MiniLM-L6-v2"
):
    np.savez(
        file_path,
        ids=ids,
        embeddings=embeddings,
        model=model
    )


# %%
def get_embeddings2(
    df: pd.DataFrame,
    text_col: str,
    model_str: str = "all-MiniLM-L6-v2",
    text_embeddings_path: Path = Path("test_embeddings.npz"),
    # text_embeddings_path: Path = Path("../data/embeddings/text_embeddings.npy"),
    regenerate: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    # if file, loads file. no check, no filler.
    if os.path.exists(text_embeddings_path) and not regenerate:
        ids, embeddings = load_ids_embeddings(text_embeddings_path)
    else:
        model = SentenceTransformer(model_str)

        ids = df.index.to_numpy()
        embeddings = model.encode(df[text_col].tolist(), convert_to_numpy=True)
        
        save_embeddings(ids, embeddings)
    
    return ids, embeddings


# %%
def add_text_embeddings2(
    df: pd.DataFrame,
    text_col: str
) -> pd.DataFrame:
    ids, embeddings = get_embeddings2(df, text_col)
    
    vector_df = pd.DataFrame(
        embeddings,
        index=ids,
        columns=[f"{text_col}_{i}" for i in range(embeddings.shape[1])]
    )

    # df = pd.concat([df, vector_df], axis=1)
    df = df.combine_first(vector_df)

    return df


# %%
# mb just one more u know ;-;
def this_time_surely(
        df: pd.DataFrame,
        embedding_path: Path,
        text_col: str = "text"
    ):
    if os.path.exists(embedding_path):
        # load the ids, embeddings
        ids_loaded, embeddings_loaded = load_ids_embeddings()

        # check what ids are missing
        

        # generate the missing ids
        # merge the loaded and new embs
        
        # format for dataframe
        # merge with original dataframe

        # somewhere save

        # return merged
        pass
    else:
        # if there's _no_file
        # generate embs for all rows
        # format if necessary
        # save embs
        # return df
        pass


# %%
# FUNCTIONAL BTW
# well for the most part
def u_can_dance_if_u_want_to(df: pd.DataFrame, text_col: str = "title"):
    ids_loaded, embeddings_loaded = load_ids_embeddings()

    # all ids in dataframe and in emb -> to load
    intersecting_ids = df.index.intersection(ids_loaded.tolist())
    # all ids - result -> to generate
    ids_to_generate = df.index.difference(intersecting_ids)

    intersecting_ids_dict = {
        id: emb for id, emb in zip(ids_loaded, embeddings_loaded) if id in intersecting_ids
    }
    embedding_dim = next(iter(intersecting_ids_dict.values()), np.array([])).shape[0]

    intersecting_embeddings_df = pd.DataFrame.from_dict(
        intersecting_ids_dict,
        orient="index",
        columns=[f'{text_col}_{i}' for i in range(embedding_dim)]
    )

    text_to_generate = df.loc[ids_to_generate][text_col].to_list()

    new_embeddings = model.encode(
        text_to_generate,
        convert_to_numpy=True
    )

    new_embeddings_dict = dict(zip(ids_to_generate, new_embeddings))
    new_embedding_dim = next(iter(new_embeddings_dict.values()), np.array([])).shape[0]

    new_embeddings_df = pd.DataFrame.from_dict(
        new_embeddings_dict,
        orient="index",
        columns=[f'{text_col}_{i}' for i in range(new_embedding_dim)]
    )

    df_result = df.combine_first(intersecting_embeddings_df).combine_first(new_embeddings_df)

    return df_result

# test_df: pd.DataFrame = pd.read_pickle("test_df")
# test_df = test_df.set_index("id")
# test_df.loc[16] = "helloo"
# test_df.loc[192] = "helloooo"

# # result = add_text_embeddings2(test_df, "title")
# result = u_can_dance_if_u_want_to(test_df, "title")
# result

    # embedding_df_loaded = pd.DataFrame(
    #     cached_embeddings,
    #     index=cached_ids,
    #     columns=[f"embedding_{i}" for i in range(cached_embeddings.shape[1])]
    # )


# %%
dic1 = {1: 1}
dic2 = {2: 2}
# dic1.update(dic2)
dic1 | dic2

# %%
dic1

# %%
np.array([]).shape[0]

# %%
test_df: pd.DataFrame = pd.read_pickle("test_df")
test_df = test_df.set_index("id")
test_df.loc[16] = "helloo"

# result = add_text_embeddings2(test_df, "title")
result = u_can_dance_if_u_want_to(test_df, "title")
result

# %%
# clean ;-;

# %%
test_df


# %%
def smth_smth(
        df: pd.DataFrame,
        embedding_path: Path,
        text_col: str = "text"
    ) -> pd.DataFrame:

    if os.path.exists(embedding_path):
        cache = np.load(embedding_path)
        
        cache_emb_df = pd.DataFrame(
            cache["embeddings"],
            index=cache["ids"],
            columns=[f"embedding_{i}" for i in range(cache["embeddings"].shape[1])]
        )

        missing_df = df.loc[
            ~df.index.isin(cache_emb_df.index)
        ]

        new_embeddings = model.encode(
            missing_df[text_col].tolist(),
            convert_to_numpy=True
        )

        new_embedding_df = pd.DataFrame(
            new_embeddings,
            index=missing_df.index,
            columns=cache_emb_df.columns
        )

        embedding_df = cache_emb_df.combine_first(new_embedding_df)

        result_df = test_df.combine_first(embedding_df).combine_first(new_embedding_df)
    else:
        embeddings = model.encode(
            df[text_col].tolist(),
            convert_to_numpy=True
        )

        embedding_df = pd.DataFrame(
            embeddings,
            index=df.index,
            columns=[f"embedding_{i}" for i in range(embeddings.shape[1])]
        )

        result_df = df.combine_first(embedding_df)
    
    # better to check if save makes sense
    # np.savez(embeddings_path) WITH BONUS
    
    return result_df

# smth_smth()


# %%
test_df: pd.DataFrame = pd.read_pickle("test_df")
test_df = test_df.set_index("id")
test_df.loc[16] = "helloo"

smth_smth(test_df, Path("test_embeddings.npz"), "title")

# %%

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

new_embeddings = model.encode(
    missing_df["title"].tolist(),
    convert_to_numpy=True
)

new_embedding_df = pd.DataFrame(
    new_embeddings,
    index=missing_df.index,
    columns=embedding_df.columns
)

# embedding_df = pd.concat(
#     [test_df, embedding_df, new_embedding_df]
# )

result_df = test_df.combine_first(embedding_df).combine_first(new_embedding_df)

result_df

# %%
embedding_df

# %%
list(intersection_emb_dic.keys())

# %%
intersection_emb_dic.values()

# %%
embedding_df = pd.DataFrame(
    intersection_emb_dic,
    index=list(intersection_emb_dic.keys()),
    columns=[f"embedding_{i}" for i in range(384)]
)

# %%
embedding_df

# %%
intersection_emb_dic

# %%
ya = [cache_dict[x] for x in index_intersection]

# %%
ya = [cache_dict.get(x) for x in test_df.index]

# %%
ya

# %%
# test_df.index.union(hi_im_something)
# test_df.index.difference(hi_im_something)
remainder

# %%
# embeddings_df = pd.DataFrame(
#     cache["embeddings"],
#     index=cache["ids"],
#     columns=[f"embedding_{i}" for i in range(cache["embeddings"].shape[1])]
# )

# result_df = test_df.join(
#     embeddings_df
# )

# result_df

# %%
output

# %%
loaded_test_embeddings["embeddings"].shape[1]

# %%
# goal: make test_df with embeddings
# also, create situation where one row is ignored intentionally
funky_dic = dict(zip(loaded_test_embeddings["ids"], loaded_test_embeddings["embeddings"]))
# test_df["embedding"] = 

# %%
test_df.index.values

# %%
embeddings

# %%
test_df = test_df.set_index("id")
test_df

# %%
# good to know
test_df.index  # gives ids
test_df.loc[6]  # gives entry @ id 6
test_df.iloc[0]  # gives 0th row

# %%
df.shape

# %%
# end testing

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
