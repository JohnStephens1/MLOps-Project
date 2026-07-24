import numpy as np
import pandas as pd
import os

from pathlib import Path
from sentence_transformers import SentenceTransformer
from text_classifier.config import EMBEDDING_DIM, EMBEDDINGS_PATH, EMBEDDING_MODEL_STR


def save_embeddings(
    ids: np.typing.ArrayLike,
    embeddings: np.typing.ArrayLike,
    file_path: Path = EMBEDDINGS_PATH,
    model: str = EMBEDDING_MODEL_STR,
):
    """saves embeddings to file_path, including ids and model name

    Args:
        ids (np.typing.ArrayLike): ids array
        embeddings (np.typing.ArrayLike): embeddings array
        file_path (Path): file path to save location
        model (str, optional): used embedding model name. Defaults to EMBEDDING_MODEL_STR.
    """
    np.savez(file_path, ids=ids, embeddings=embeddings, model=model)


def load_ids_embeddings(
    file_path: Path = EMBEDDINGS_PATH,
    embedding_dim: int = EMBEDDING_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """loads the ids and embeddings stored in file_path

    Args:
        file_path (Path): file path to stored data

    Returns:
        tuple[np.ndarray, np.ndarray]: ids, embeddings
    """
    if not os.path.exists(file_path):
        return np.empty(0), np.empty((0, embedding_dim))

    loaded_file = np.load(file_path)

    return loaded_file["ids"], loaded_file["embeddings"]


def get_intersecting_complementing_ids(
    df: pd.DataFrame, ids_loaded: np.ndarray
) -> tuple[pd.Index, pd.Index]:
    """gets the intersection between df and saved ids, as well as the remaining ids that need to be generated

    Args:
        df (pd.DataFrame): input df
        ids_loaded (np.ndarray): the loaded ids from saved embeddings

    Returns:
        tuple[pd.Index, pd.Index]: intersecting_ids, ids_to_generate.
        intersection between df and saved ids, as well as the remaining ids that need to be generated
    """
    # all ids in input df and embeddings -> to load
    intersecting_ids = df.index.intersection(ids_loaded.tolist())

    # all ids - intersecting ids -> to generate
    ids_to_generate = df.index.difference(intersecting_ids)

    return intersecting_ids, ids_to_generate


def get_new_embeddings(
    df: pd.DataFrame,
    ids_to_generate: pd.Index,
    text_col: str,
    model_str: str = EMBEDDING_MODEL_STR,
) -> np.ndarray:
    """generates new embeddings for df based on ids_to_generate using the model defined in model_str

    Args:
        df (pd.DataFrame): input df
        ids_to_generate (pd.Index): corresponding ids to generate
        text_col (str): name of the text column
        model_str (str, optional): name of the embedding model. Defaults to EMBEDDING_MODEL_STR.

    Returns:
        np.ndarray: generated embeddings
    """
    model = SentenceTransformer(model_str)

    text_to_generate = df.loc[ids_to_generate][text_col].to_list()

    new_embeddings = model.encode(text_to_generate, convert_to_numpy=True)

    return new_embeddings


# def get_embeddings_dict(
#     ids_loaded: np.ndarray,
#     embeddings_loaded: np.ndarray,
#     intersecting_ids: pd.Index,
#     ids_to_generate: pd.Index,
#     new_embeddings: np.ndarray,
# ) -> dict[pd.Index, np.ndarray]:
#     """gets a dictionary with ids mapping to the new and loaded embeddings for df

#     Args:
#         ids_loaded (np.ndarray): loaded ids
#         embeddings_loaded (np.ndarray): loaded embeddings
#         intersecting_ids (pd.Index): intersecting ids
#         ids_to_generate (pd.Index): ids that require new generation
#         new_embeddings (np.ndarray): newly generated embeddings

#     Returns:
#         dict[pd.Index, np.ndarray]: embeddings_dict. contains all ids and embeddings for df
#     """
#     loaded_embeddings_dict = dict(zip(ids_loaded, embeddings_loaded))

#     x = {
#         id: emb for id, emb in loaded_embeddings_dict.items() if id in intersecting_ids
#     }

#     new_embeddings_dict = dict(zip(ids_to_generate, new_embeddings))

#     embeddings_dict = x | new_embeddings_dict

#     return embeddings_dict


# def get_embeddings_df(
#     embeddings_dict: dict[pd.Index, np.ndarray], text_col: str
# ) -> pd.DataFrame:
#     """creates a df based on the input dic

#     Args:
#         embeddings_dict (dict[pd.Index, np.ndarray]): dict containing ids mapped to the corresponding embeddings
#         text_col (str): name of the text column

#     Returns:
#         pd.DataFrame: embeddings_df. the dataframe built from the input dict
#     """
#     embedding_dim = next(iter(embeddings_dict.values()), np.array([])).shape[0]
#     embeddings_df = pd.DataFrame.from_dict(
#         embeddings_dict,
#         orient="index",
#         columns=[f"{text_col}_{i}" for i in range(embedding_dim)],
#     )

#     return embeddings_df


def get_and_save_generated_embeddings(
    df: pd.DataFrame,
    ids_loaded: np.ndarray,
    ids_to_generate: pd.Index,
    embeddings_loaded: np.ndarray,
    text_col: str,
    file_path: Path = EMBEDDINGS_PATH,
    embedding_dim: int = EMBEDDING_DIM,
) -> np.ndarray:
    if ids_to_generate.empty:
        return np.empty((0, embedding_dim))

    # generate embeddings for ids_to_generate
    embeddings_generated = get_new_embeddings(df, ids_to_generate, text_col)

    print(
        f"ids_to_generate: {ids_to_generate.shape}\n"
        f"embeddings_generated: {embeddings_generated.shape}"
    )
    print(f"ids_loaded: {ids_loaded.shape}\n")
    print(f"embeddings_loaded: {embeddings_loaded.shape}\n")

    # to_save = loaded + generated
    ids_to_save = np.concat([ids_loaded, ids_to_generate])
    embeddings_to_save = np.concat([embeddings_loaded, embeddings_generated])

    save_embeddings(ids_to_save, embeddings_to_save, file_path)

    return embeddings_generated


def get_intersecting_embeddings(
    ids_loaded: np.ndarray,
    embeddings_loaded: np.ndarray,
    ids_intersecting: pd.Index,
    embedding_dim: int = EMBEDDING_DIM,
) -> np.ndarray:
    print(f"ids_intersecting: {ids_intersecting.shape}")
    print(f"ids_loaded: {ids_loaded.shape}")
    print(f"embeddings_loaded: {embeddings_loaded.shape}")
    return np.array(
        [
            emb
            for id, emb in zip(ids_loaded, embeddings_loaded)
            if id in ids_intersecting
        ]
    ).reshape(-1, embedding_dim)


def get_embeddings_df(
    ids_result: pd.Index,
    embeddings_result: np.ndarray,
    text_col: str,
    embedding_dim: int = EMBEDDING_DIM,
):
    return pd.DataFrame(
        embeddings_result,
        index=ids_result,
        columns=[f"{text_col}_{i}" for i in range(embedding_dim)],
    )


def add_text_embeddings(
    df: pd.DataFrame,
    text_col: str = "title",
    file_path: Path = EMBEDDINGS_PATH,
    embedding_dim: int = EMBEDDING_DIM,
) -> pd.DataFrame:
    ids_loaded, embeddings_loaded = load_ids_embeddings(file_path)

    ids_intersecting, ids_to_generate = get_intersecting_complementing_ids(
        df, ids_loaded
    )

    embeddings_generated = get_and_save_generated_embeddings(
        df, ids_loaded, ids_to_generate, embeddings_loaded, text_col
    )

    embeddings_intersecting = get_intersecting_embeddings(
        ids_loaded, embeddings_loaded, ids_intersecting
    )

    print(
        f"ids_intersecting: {ids_intersecting.shape}\n"
        f"ids_to_generate: {ids_to_generate.shape}\n"
        f"embeddings_generated: {embeddings_generated.shape}\n"
        f"embeddings_intersecting: {embeddings_intersecting.shape}\n"
    )

    # result = intersecting + generated
    ids_result = np.concat([ids_intersecting, ids_to_generate])
    embeddings_result = np.concat([embeddings_intersecting, embeddings_generated])

    print(
        f"ids_result: {ids_result.shape}\nembeddings_result: {embeddings_result.shape}"
    )

    # create embeddings df for result
    df_embeddings = pd.DataFrame(
        embeddings_result,
        index=ids_result,
        columns=[f"{text_col}_{i}" for i in range(embedding_dim)],
    )

    # merge into input df
    df_result = df.combine_first(df_embeddings)

    return df_result

    # embeddings_loaded_dict = dict(zip(ids_loaded, embeddings_loaded))

    # if ids_to_generate.empty:
    #     embeddings_generated = np.array([])
    #     embeddings_new_dict = {}
    # else:
    #     embeddings_generated = get_new_embeddings(df, ids_to_generate, text_col)
    #     embeddings_new_dict = dict(zip(ids_to_generate, embeddings_generated))

    #     # result = intersecting + generated  # for return
    #     # full = loaded + generated  # for saving
    #     embeddings_full_dict = embeddings_loaded_dict | embeddings_new_dict
    #     embeddings_full_dict = embeddings_new_dict | embeddings_loaded_dict

    # embeddings_dict = get_embeddings_dict(
    #     ids_loaded, embeddings_loaded, ids_intersecting, ids_to_generate, embeddings_generated
    # )

    # # TODO: add to existing embeddings instead of replacing everything
    # if not ids_to_generate.empty:
    #     save_embeddings(
    #         list(embeddings_dict.keys()), list(embeddings_dict.values()), file_path
    #     )

    # embeddings_df = get_embeddings_df(embeddings_dict, text_col)

    # # merge by index with input df
    # df_result = df.combine_first(embeddings_df)

    # return df_result
