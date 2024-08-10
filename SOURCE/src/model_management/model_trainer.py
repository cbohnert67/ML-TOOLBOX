import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator



class ModelTrainer:
    def __init__(self, model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Initialize the ModelTrainer with the model and training data.
        :param model: Machine learning model to train
        :param X_train: Training features
        :param y_train: Training labels
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        """
        Train the model.
        """
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        :param X_test: Test features
        :return: Predictions
        """
        return self.model.predict(X_test)