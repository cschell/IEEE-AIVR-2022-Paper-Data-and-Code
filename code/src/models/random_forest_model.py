from sklearn.ensemble import RandomForestClassifier

from src.hyperparameters.random_forest_hyperparameters import RandomForestHyperparameters


class RandomForestModel:
    def __init__(self, hyperparameters: RandomForestHyperparameters, **sklearn_kwargs) -> None:
        self.hyperparameters = hyperparameters
        self.sklearn_model = RandomForestClassifier(**hyperparameters.model_params, **sklearn_kwargs)

    def fit(self, *args, **kwargs):
        return self.sklearn_model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.sklearn_model.predict(*args, **kwargs)