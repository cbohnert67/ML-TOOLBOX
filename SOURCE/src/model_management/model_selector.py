import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

class ModelSelector:
    def __init__(self, models: dict, X: pd.DataFrame, y: pd.Series):
        """
        Initialize the ModelSelector with the models and data.
        :param models: Dictionary of models to evaluate
        :param X: Features
        :param y: Labels
        """
        self.models = models
        self.X = X
        self.y = y

    def select_best_model(self):
        """
        Select the best model based on cross-validation scores.
        :return: Best model
        """
        best_score = -np.inf
        best_model = None
        for name, model in self.models.items():
            scores = cross_val_score(model, self.X, self.y, cv=5)
            mean_score = scores.mean()
            print(f"Model: {name}, Mean CV Score: {mean_score}")
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
        return best_model