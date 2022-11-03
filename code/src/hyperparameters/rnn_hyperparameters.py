from dataclasses import dataclass

from src.hyperparameters.base_hyperparameters import BaseHyperparameters


@dataclass
class RNNHyperparameters(BaseHyperparameters):
    num_rnn_layers: int
    rnn_hidden_size: int
    dropout: float
    cell_type: str
