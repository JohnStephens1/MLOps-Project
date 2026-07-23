import numpy as np
import pandas as pd
import os

from pathlib import Path
from sentence_transformers import SentenceTransformer
from text_classifier.config import EMBEDDINGS_DIR, EMBEDDING_MODEL_STR


def save_embeddings(
    ids: np.typing.ArrayLike,
    embeddings: np.typing.ArrayLike,
    file_path: Path,
    model: str = EMBEDDING_MODEL_STR
):
    np.savez(
        file_path,
        ids=ids,
        embeddings=embeddings,
        model=model
    )


def load_ids_embeddings(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(file_path):
        return np.array([]), np.array([])
    
    loaded_file = np.load(file_path)
    
    return loaded_file['ids'], loaded_file['embeddings']


def get_intersecting_complementing_ids(
    df: pd.DataFrame,
    ids_loaded: np.ndarray
) -> tuple[pd.Index, pd.Index]:
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
    model = SentenceTransformer(model_str)
    
    text_to_generate = df.loc[ids_to_generate][text_col].to_list()

    new_embeddings = model.encode(
        text_to_generate,
        convert_to_numpy=True
    )

    return new_embeddings


def get_embeddings_dic(
    ids_loaded: np.ndarray,
    embeddings_loaded: np.ndarray,
    intersecting_ids: pd.Index,
    ids_to_generate: pd.Index,
    new_embeddings: np.ndarray
) -> dict[pd.Index, np.ndarray]:
    intersecting_ids_dic = {
        id: emb for id, emb in zip(ids_loaded, embeddings_loaded) if id in intersecting_ids
    }
    new_embeddings_dic = dict(zip(ids_to_generate, new_embeddings))
    embeddings_dic = intersecting_ids_dic | new_embeddings_dic

    return embeddings_dic


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


def add_text_embeddings(
    df: pd.DataFrame,
    text_col: str = "title",
    file_path: Path = EMBEDDINGS_DIR,
) -> pd.DataFrame:
    ids_loaded, embeddings_loaded = load_ids_embeddings(file_path)
    intersecting_ids, ids_to_generate = get_intersecting_complementing_ids(df, ids_loaded)

    new_embeddings = np.array([]) if ids_to_generate.empty else get_new_embeddings(df, ids_to_generate, text_col)
    
    embeddings_dic = get_embeddings_dic(
        ids_loaded,
        embeddings_loaded,
        intersecting_ids,
        ids_to_generate,
        new_embeddings
    )

    if not ids_to_generate.empty:
        save_embeddings(
            list(embeddings_dic.keys()),
            list(embeddings_dic.values()),
            file_path
        )

    embeddings_df = get_embeddings_df(
        embeddings_dic,
        text_col
    )

    # merge by index with input df
    df_result = df.combine_first(embeddings_df)

    return df_result
