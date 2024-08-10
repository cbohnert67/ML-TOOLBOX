import joblib
from sklearn.base import BaseEstimator



class ModelSaver:
    def __init__(self, model: BaseEstimator, file_path: str):
        """
        Initialize the ModelSaver with the model and file path.
        :param model: Trained machine learning model
        :param file_path: Path to save the model
        """
        self.model = model
        self.file_path = file_path

    def save(self):
        """
        Save the model to the specified file path.
        """
        joblib.dump(self.model, self.file_path)
