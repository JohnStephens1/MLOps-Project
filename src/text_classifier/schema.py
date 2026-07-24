import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


@dataclass
class TrainingData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.typing.ArrayLike
    y_test: np.typing.ArrayLike


@dataclass
class Artifacts:
    pipe: Pipeline
    label_encoder: LabelEncoder


@dataclass
class TrainingContext:
    dataset: TrainingData
    artifacts: Artifacts
