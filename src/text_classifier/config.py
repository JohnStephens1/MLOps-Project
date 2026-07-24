from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASET_PATH = PROJECT_ROOT / "datasets/dataset.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "data/embeddings/text_embeddings.npz"

EMBEDDING_MODEL_STR = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
