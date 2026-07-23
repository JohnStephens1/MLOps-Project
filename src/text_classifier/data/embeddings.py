import numpy as np
import pandas as pd
import os

from pathlib import Path
from sentence_transformers import SentenceTransformer
from text_classifier.config import EMBEDDINGS_PATH, EMBEDDING_MODEL_STR


def save_embeddings(
    ids: np.typing.ArrayLike,
    embeddings: np.typing.ArrayLike,
    file_path: Path,
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


def load_ids_embeddings(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """loads the ids and embeddings stored in file_path

    Args:
        file_path (Path): file path to stored data

    Returns:
        tuple[np.ndarray, np.ndarray]: ids, embeddings
    """
    if not os.path.exists(file_path):
        return np.array([]), np.array([])

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


def get_embeddings_dic(
    ids_loaded: np.ndarray,
    embeddings_loaded: np.ndarray,
    intersecting_ids: pd.Index,
    ids_to_generate: pd.Index,
    new_embeddings: np.ndarray,
) -> dict[pd.Index, np.ndarray]:
    """gets a dictionary with ids mapping to the new and loaded embeddings for df

    Args:
        ids_loaded (np.ndarray): loaded ids
        embeddings_loaded (np.ndarray): loaded embeddings
        intersecting_ids (pd.Index): intersecting ids
        ids_to_generate (pd.Index): ids that require new generation
        new_embeddings (np.ndarray): newly generated embeddings

    Returns:
        dict[pd.Index, np.ndarray]: embeddings_dic. contains all ids and embeddings for df
    """
    intersecting_ids_dic = {
        id: emb
        for id, emb in zip(ids_loaded, embeddings_loaded)
        if id in intersecting_ids
    }
    new_embeddings_dic = dict(zip(ids_to_generate, new_embeddings))
    embeddings_dic = intersecting_ids_dic | new_embeddings_dic

    return embeddings_dic


def get_embeddings_df(
    embeddings_dic: dict[pd.Index, np.ndarray], text_col: str
) -> pd.DataFrame:
    """creates a df based on the input dic

    Args:
        embeddings_dic (dict[pd.Index, np.ndarray]): dic containing ids mapped to the corresponding embeddings
        text_col (str): name of the text column

    Returns:
        pd.DataFrame: embeddings_df. the dataframe built from the input dic
    """
    embedding_dim = next(iter(embeddings_dic.values()), np.array([])).shape[0]
    embeddings_df = pd.DataFrame.from_dict(
        embeddings_dic,
        orient="index",
        columns=[f"{text_col}_{i}" for i in range(embedding_dim)],
    )

    return embeddings_df


def add_text_embeddings(
    df: pd.DataFrame,
    text_col: str = "title",
    file_path: Path = EMBEDDINGS_PATH,
) -> pd.DataFrame:
    """adds text embeddings to the input df for the text_col, loading and saving embeddings from and to file_path

    Args:
        df (pd.DataFrame): input df
        text_col (str, optional): name of the column containing text. Defaults to "title".
        file_path (Path, optional): path to the stored embeddings. Defaults to EMBEDDINGS_DIR.

    Returns:
        pd.DataFrame: df with added embeddings
    """
    ids_loaded, embeddings_loaded = load_ids_embeddings(file_path)
    intersecting_ids, ids_to_generate = get_intersecting_complementing_ids(
        df, ids_loaded
    )

    new_embeddings = (
        np.array([])
        if ids_to_generate.empty
        else get_new_embeddings(df, ids_to_generate, text_col)
    )

    embeddings_dic = get_embeddings_dic(
        ids_loaded, embeddings_loaded, intersecting_ids, ids_to_generate, new_embeddings
    )

    if not ids_to_generate.empty:
        save_embeddings(
            list(embeddings_dic.keys()), list(embeddings_dic.values()), file_path
        )

    embeddings_df = get_embeddings_df(embeddings_dic, text_col)

    # merge by index with input df
    df_result = df.combine_first(embeddings_df)

    return df_result
