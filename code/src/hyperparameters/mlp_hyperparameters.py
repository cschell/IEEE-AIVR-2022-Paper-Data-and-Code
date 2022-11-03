from dataclasses import dataclass

from src.hyperparameters.base_hyperparameters import BaseHyperparameters


@dataclass
class MLPHyperparameters(BaseHyperparameters):
    layer_size: int
    number_of_layers: int
    activation_function: str
