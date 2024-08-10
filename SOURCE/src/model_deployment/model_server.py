from sklearn.base import BaseEstimator



class ModelServer:
    def __init__(self, model: BaseEstimator):
        """
        Initialize the ModelServer with the model.
        :param model: Trained machine learning model
        """
        self.model = model

    def serve(self):
        """
        Start a server to serve the model for predictions.
        """
        raise NotImplementedError("Model serving is not implemented yet")
