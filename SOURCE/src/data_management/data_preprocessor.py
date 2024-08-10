import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataPreprocessor with the data.
        :param data: DataFrame containing the data to preprocess
        """
        self.data = data

    def handle_missing_values(self, strategy: str = 'mean'):
        """
        Handle missing values in the data.
        :param strategy: Strategy to handle missing values ('mean', 'median', 'most_frequent')
        """
        imputer = SimpleImputer(strategy=strategy)
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)

    def scale_features(self, method: str = 'standard'):
        """
        Scale features in the data.
        :param method: Method to scale features ('standard', 'minmax')
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method")
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)

