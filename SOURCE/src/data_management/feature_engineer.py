import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the FeatureEngineer with the data.
        :param data: DataFrame containing the data for feature engineering
        """
        self.data = data

    def create_features(self):
        """
        Create new features based on existing data.
        """
        # Example: Create a new feature as the sum of two existing features
        #self.data['new_feature'] = self.data['feature1'] + self.data['feature2']
        pass

    def select_features(self, method: str = 'correlation'):
        """
        Select important features based on the specified method.
        :param method: Method to select features ('correlation', 'importance')
        """
        if method == 'correlation':
            corr_matrix = self.data.corr()
            # Example: Select features with correlation above a threshold
            selected_features = corr_matrix.index[abs(corr_matrix["target"]) > 0.5]
            self.data = self.data[selected_features]
        elif method == 'importance':
            # Example: Use feature importance from a model
            model = RandomForestClassifier()
            model.fit(self.data.drop('target', axis=1), self.data['target'])
            importances = model.feature_importances_
            selected_features = self.data.columns[importances > 0.1]
            self.data = self.data[selected_features]
        else:
            raise ValueError("Unsupported feature selection method")
