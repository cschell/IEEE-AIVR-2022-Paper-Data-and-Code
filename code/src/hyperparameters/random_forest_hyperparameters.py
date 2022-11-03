from dataclasses import dataclass
from typing import Optional

from src.hyperparameters.base_hyperparameters import BaseHyperparameters


@dataclass
class RandomForestHyperparameters(BaseHyperparameters):
    n_estimators: int
    min_samples_leaf: float
    min_samples_split: int = 2
    max_depth: Optional[int] = None

    @property
    def model_params(self):
        return self.__dict__.items()
