import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../')))  # Add the path to the src folder to the system path

from src.data_management.data_loader import DataLoader
from src.data_management.data_preprocessor import DataPreprocessor
from src.data_management.feature_engineer import FeatureEngineer
from src.model_management.model_trainer import ModelTrainer
from src.model_management.model_evaluator import ModelEvaluator
from src.model_deployment.model_saver import ModelSaver
from src.model_deployment.model_loader import ModelLoader
from src.model_deployment.model_server import ModelServer
from src.pipeline_management.ml_pipeline import MLPipeline

# Sample code to demonstrate usage
file_path = './../../data/raw/boston_housing.csv'
absolute_file_path = os.path.abspath(file_path)
print(f"Trying to load data from: {absolute_file_path}")

data_loader = DataLoader(source=file_path)
data_loader.load_data()
data = data_loader.get_data()

# Name of the target variable in the dataset boston_housing.csv
target_variable = 'MEDV'

# Rename the target variable to 'target' for consistency
data = data.rename(columns={target_variable: 'target'})


# Vérifiez que les données sont correctement chargées
if data is None:
    print("Error: Data not loaded. Please check the file path and ensure the file exists.")
else:
    print("Data loaded:")
    print(data.head())

    preprocessor = DataPreprocessor(data=data)
    preprocessor.handle_missing_values()

    # Vérifiez que les données sont correctement prétraitées
    print("Data after preprocessing:")
    print(preprocessor.data.head())

    feature_engineer = FeatureEngineer(data=preprocessor.data)
    feature_engineer.select_features()

    # Vérifiez que les caractéristiques sont correctement générées
    print("Data after feature engineering:")
    print(feature_engineer.data.head())

    models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Regressor": SVR()
    } 

    for model_name, model in models.items():
        print(f"Model: {model_name}")
        model_trainer = ModelTrainer(model=model, X_train=pd.DataFrame(), y_train=pd.Series())
        model_evaluator = ModelEvaluator(model=model, X_test=pd.DataFrame(), y_test=pd.Series())
        pipeline = MLPipeline(data_loader, preprocessor, feature_engineer, model_trainer, model_evaluator)
        pipeline.run_pipeline()
        





    # Save the model
    #model_saver = ModelSaver(model=model, file_path='model.pkl')
    #model_saver.save()

    # Load the model
    #model_loader = ModelLoader(file_path='model.pkl')
    #loaded_model = model_loader.load()

    # Serve the model (not implemented)
    #model_server = ModelServer(model=loaded_model)
    #model_server.serve()  # This will raise NotImplementedError in this example