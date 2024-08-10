from sklearn.base import BaseEstimator
import joblib



class ModelLoader:
    def __init__(self, file_path: str):
        """
        Initialize the ModelLoader with the file path.
        :param file_path: Path to load the model from
        """
        self.file_path = file_path

    def load(self) -> BaseEstimator:
        """
        Load a model from the specified file path.
        :return: Loaded model
        """
        return joblib.load(self.file_path)