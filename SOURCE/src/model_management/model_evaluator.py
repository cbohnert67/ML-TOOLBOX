import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score



class ModelEvaluator:
    def __init__(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Initialize the ModelEvaluator with the model and test data.
        :param model: Trained machine learning model
        :param X_test: Test features
        :param y_test: Test labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        """
        Compute performance metrics.
        """
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")

    def cross_validate(self, cv: int = 5):
        """
        Perform cross-validation and return performance metrics.
        :param cv: Number of cross-validation folds
        """
        scores = cross_val_score(self.model, self.X_test, self.y_test, cv=cv)
        print(f"Cross-validation scores: {scores}")
        print(f"Mean cross-validation score: {scores.mean()}")