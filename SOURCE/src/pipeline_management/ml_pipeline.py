from sklearn.model_selection import train_test_split
from src.data_management.data_loader import DataLoader
from src.data_management.data_preprocessor import DataPreprocessor
from src.data_management.feature_engineer import FeatureEngineer
from src.model_management.model_trainer import ModelTrainer
from src.model_management.model_evaluator import ModelEvaluator



class MLPipeline:
    def __init__(self, data_loader: DataLoader, preprocessor: DataPreprocessor, feature_engineer: FeatureEngineer,
                 model_trainer: ModelTrainer, model_evaluator: ModelEvaluator):
        """
        Initialize the MLPipeline with the components.
        :param data_loader: DataLoader instance
        :param preprocessor: DataPreprocessor instance
        :param feature_engineer: FeatureEngineer instance
        :param model_trainer: ModelTrainer instance
        :param model_evaluator: ModelEvaluator instance
        """
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator

    def run_pipeline(self):
        """
        Execute the full ML pipeline.
        """
        self.data_loader.load_data()
        data = self.data_loader.get_data()
        self.preprocessor.handle_missing_values()
        self.preprocessor.scale_features()
        #self.feature_engineer.create_features()
        self.feature_engineer.select_features()
        X = self.feature_engineer.data.drop('target', axis=1)
        y = self.feature_engineer.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model_trainer.X_train = X_train
        self.model_trainer.y_train = y_train
        self.model_trainer.fit()
        self.model_evaluator.X_test = X_test
        self.model_evaluator.y_test = y_test
        self.model_evaluator.evaluate()
