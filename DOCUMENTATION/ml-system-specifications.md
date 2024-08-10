To extend the ML system and GUI to include the new features, we'll enhance the library with additional functionalities for advanced data handling, preprocessing, model management, evaluation, pipeline management, deployment, and testing. We'll also update the GUI to support these enhancements.

Here's how we can implement these extensions:

### 1. **Extended Library**

#### Data Management Enhancements

##### 1.1 Additional Data Sources and Validation

**data_management.py**

```python
import pandas as pd
from sqlalchemy import create_engine
import requests
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DataLoader:
    def __init__(self, source: str):
        self.source = source
        self.data = None

    def load_data(self):
        if self.source.endswith('.csv'):
            self.data = pd.read_csv(self.source)
        elif self.source.startswith('http'):
            response = requests.get(self.source)
            if response.status_code == 200:
                self.data = pd.DataFrame(response.json())
            else:
                raise ValueError(f"Failed to load data from API. Status code: {response.status_code}")
        else:
            engine = create_engine(self.source)
            self.data = pd.read_sql(self.source, engine)
        self.validate_data()
        return self

    def validate_data(self):
        # Example validation checks
        if self.data.isnull().sum().sum() > 0:
            raise ValueError("Data contains missing values.")
        # Add more validation as needed
        return self

    def get_data(self):
        return self.data

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def handle_missing_values(self, strategy: str):
        imputer = SimpleImputer(strategy=strategy)
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        return self

    def encode_categorical_features(self):
        self.data = pd.get_dummies(self.data)
        return self

    def scale_features(self, method: str):
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        return self

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def create_features(self):
        # Example: Create a new feature 'feature_sum'
        self.data['feature_sum'] = self.data.sum(axis=1)
        return self

    def select_features(self, method: str):
        if method == 'top_k':
            # Placeholder for feature selection logic
            pass
        elif method == 'rfe':
            from sklearn.feature_selection import RFE
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
            rfe = RFE(model, 5)
            fit = rfe.fit(self.data.drop('target', axis=1), self.data['target'])
            self.data = self.data[fit.support_]
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")
        return self
```

#### Advanced Data Preprocessing

**Add advanced transformers and feature selection methods**

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

class AdvancedFeatureEngineer(FeatureEngineer):
    def select_features(self, method: str):
        if method == 'importance':
            model = GradientBoostingClassifier()
            model.fit(self.data.drop('target', axis=1), self.data['target'])
            selector = SelectFromModel(model, threshold="mean", prefit=True)
            self.data = self.data.loc[:, selector.get_support()]
        else:
            super().select_features(method)
        return self
```

#### Model Management Extensions

**model_management.py**

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.base import is_classifier
from sklearn.metrics import make_scorer, roc_auc_score, r2_score

class ModelTrainer:
    def __init__(self, model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        return self

    def predict(self, X_test: pd.DataFrame):
        return self.model.predict(X_test)

class ModelEvaluator:
    def __init__(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        predictions = self.model.predict(self.X_test)
        metrics = {}
        if is_classifier(self.model):
            metrics['accuracy'] = (predictions == self.y_test).mean()
            metrics['roc_auc'] = roc_auc_score(self.y_test, self.model.predict_proba(self.X_test)[:, 1])
        else:
            metrics['r2'] = r2_score(self.y_test, predictions)
        return metrics

    def cross_validate(self, cv: int):
        scores = cross_val_score(self.model, self.X_test, self.y_test, cv=cv)
        return {'mean_cv_score': scores.mean()}

class ModelSelector:
    def __init__(self, models: list, metrics: dict):
        self.models = models
        self.metrics = metrics

    def select_best_model(self):
        best_model = max(self.models, key=lambda model: self.metrics.get(model, 0))
        return best_model

class HyperparameterTuner:
    def __init__(self, model, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train

    def tune(self, method='grid'):
        if method == 'grid':
            search = GridSearchCV(self.model, self.param_grid, cv=5, scoring='accuracy')
        elif method == 'random':
            search = RandomizedSearchCV(self.model, self.param_grid, n_iter=10, cv=5, scoring='accuracy')
        else:
            raise ValueError(f"Unsupported tuning method: {method}")
        search.fit(self.X_train, self.y_train)
        return search.best_estimator_
```

#### Pipeline Management Improvements

**pipeline_management.py**

```python
from sklearn.pipeline import Pipeline
import yaml
import os

class ML_Pipeline:
    def __init__(self, data_loader, preprocessor, feature_engineer, model_trainer, model_evaluator, config: dict = None):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        self.config = config or {}

    def run_pipeline(self):
        self.data_loader.load_data()
        data = self.data_loader.get_data()
        data = self.preprocessor.handle_missing_values(strategy='mean').encode_categorical_features().scale_features(method='standard').data
        data = self.feature_engineer.create_features().data
        X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2)
        self.model_trainer.fit()
        evaluation = self.model_evaluator.evaluate()
        return evaluation

    def save_pipeline(self, file_path: str):
        with open(file_path, 'w') as file:
            yaml.dump(self.config, file)

    def load_pipeline(self, file_path: str):
        with open(file_path, 'r') as file:
            self.config = yaml.safe_load(file)
        return self.config

class PipelineConfig:
    def __init__(self):
        self.config = {}

    def set_config(self, key: str, value: any):
        self.config[key] = value

    def get_config(self, key: str):
        return self.config.get(key)

    def save_config(self, file_path: str):
        with open(file_path, 'w') as file:
            yaml.dump(self.config, file)

    def load_config(self, file_path: str):
        with open(file_path, 'r') as file:
            self.config = yaml.safe_load(file)
```

#### Model Deployment Enhancements

**model_deployment.py**

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import subprocess

class ModelSaver:
    def __init__(self, model: BaseEstimator, file_path: str):
        self.model = model
        self.file_path = file_path

    def save(self):
        joblib.dump(self.model, self.file_path)

class ModelLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        return joblib.load(self.file_path)

class ModelServer:
    def __init__(self, model: BaseEstimator):
        self.model = model
        self.app = FastAPI()

    def serve(self):
        class ModelInput(BaseModel):
            features: list

        @self.app.post("/predict")
        def predict(input: ModelInput):
            prediction = self.model.predict([input.features])
            return {"prediction": prediction.tolist()}

        return self.app

class Dockerizer:
    @staticmethod
    def create

_dockerfile():
        dockerfile_content = """
        FROM python:3.8-slim
        WORKDIR /app
        COPY . /app
        RUN pip install -r requirements.txt
        CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
        """
        with open('Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        subprocess.run(["docker", "build", "-t", "ml_model", "."])
```

#### Testing and Validation

**test_ml_system.py**

```python
import unittest
from data_management import DataLoader, DataPreprocessor, FeatureEngineer
from model_management import ModelTrainer, ModelEvaluator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class TestMLSystem(unittest.TestCase):
    def setUp(self):
        data = load_iris(as_frame=True)
        self.df = pd.concat([data.frame, pd.Series(data.target, name='target')], axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df.drop('target', axis=1), self.df['target'], test_size=0.2
        )
        self.model = RandomForestClassifier()
        self.trainer = ModelTrainer(self.model, self.X_train, self.y_train)
        self.evaluator = ModelEvaluator(self.model, self.X_test, self.y_test)

    def test_data_preprocessing(self):
        preprocessor = DataPreprocessor(self.df)
        preprocessor.handle_missing_values(strategy='mean').encode_categorical_features().scale_features(method='standard')
        self.assertEqual(self.df.shape[1], preprocessor.data.shape[1])

    def test_model_training(self):
        self.trainer.fit()
        self.assertIsNotNone(self.trainer.model)

    def test_model_evaluation(self):
        self.trainer.fit()
        results = self.evaluator.evaluate()
        self.assertIn('accuracy', results)

    def test_pipeline(self):
        # Implement end-to-end pipeline test
        pass

if __name__ == '__main__':
    unittest.main()
```

### 2. **Extended GUI**

**ml_system_gui.py**

```python
import tkinter as tk
from tkinter import filedialog, messagebox
from data_management import DataLoader, DataPreprocessor, FeatureEngineer
from model_management import ModelTrainer, ModelEvaluator, HyperparameterTuner, ModelSelector
from pipeline_management import ML_Pipeline, PipelineConfig
from model_deployment import ModelSaver, ModelLoader, ModelServer

class MLSystemGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML System GUI")

        # Data Preparation
        self.data_preparation_frame = tk.LabelFrame(self.root, text="Data Preparation", padx=10, pady=10)
        self.data_preparation_frame.pack(padx=10, pady=10, fill="x")

        self.load_data_button = tk.Button(self.data_preparation_frame, text="Load Data", command=self.load_data)
        self.load_data_button.pack(pady=10)

        self.encode_button = tk.Button(self.data_preparation_frame, text="Encode Categorical", command=self.encode_categorical)
        self.encode_button.pack(pady=10)

        self.scale_button = tk.Button(self.data_preparation_frame, text="Scale Features", command=self.scale_features)
        self.scale_button.pack(pady=10)

        # Model Training
        self.model_training_frame = tk.LabelFrame(self.root, text="Model Training", padx=10, pady=10)
        self.model_training_frame.pack(padx=10, pady=10, fill="x")

        self.train_model_button = tk.Button(self.model_training_frame, text="Train Model", command=self.train_model)
        self.train_model_button.pack(pady=10)

        self.hyperparameter_tune_button = tk.Button(self.model_training_frame, text="Tune Hyperparameters", command=self.tune_hyperparameters)
        self.hyperparameter_tune_button.pack(pady=10)

        # Model Evaluation
        self.model_evaluation_frame = tk.LabelFrame(self.root, text="Model Evaluation", padx=10, pady=10)
        self.model_evaluation_frame.pack(padx=10, pady=10, fill="x")

        self.evaluate_model_button = tk.Button(self.model_evaluation_frame, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_model_button.pack(pady=10)

        # Model Deployment
        self.model_deployment_frame = tk.LabelFrame(self.root, text="Model Deployment", padx=10, pady=10)
        self.model_deployment_frame.pack(padx=10, pady=10, fill="x")

        self.save_model_button = tk.Button(self.model_deployment_frame, text="Save Model", command=self.save_model)
        self.save_model_button.pack(pady=10)

        self.load_model_button = tk.Button(self.model_deployment_frame, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

        self.serve_model_button = tk.Button(self.model_deployment_frame, text="Serve Model", command=self.serve_model)
        self.serve_model_button.pack(pady=10)

        # Pipeline Management
        self.pipeline_management_frame = tk.LabelFrame(self.root, text="Pipeline Management", padx=10, pady=10)
        self.pipeline_management_frame.pack(padx=10, pady=10, fill="x")

        self.run_pipeline_button = tk.Button(self.pipeline_management_frame, text="Run Pipeline", command=self.run_pipeline)
        self.run_pipeline_button.pack(pady=10)

        self.save_pipeline_button = tk.Button(self.pipeline_management_frame, text="Save Pipeline", command=self.save_pipeline)
        self.save_pipeline_button.pack(pady=10)

        self.load_pipeline_button = tk.Button(self.pipeline_management_frame, text="Load Pipeline", command=self.load_pipeline)
        self.load_pipeline_button.pack(pady=10)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data_loader = DataLoader(file_path)
            self.data_loader.load_data()
            messagebox.showinfo("Info", "Data loaded successfully!")

    def encode_categorical(self):
        if hasattr(self, 'data_loader'):
            data = self.data_loader.get_data()
            preprocessor = DataPreprocessor(data)
            preprocessor.encode_categorical_features()
            messagebox.showinfo("Info", "Categorical features encoded!")
        else:
            messagebox.showerror("Error", "No data loaded.")

    def scale_features(self):
        if hasattr(self, 'data_loader'):
            data = self.data_loader.get_data()
            preprocessor = DataPreprocessor(data)
            method = 'standard'
            preprocessor.scale_features(method)
            messagebox.showinfo("Info", "Features scaled!")
        else:
            messagebox.showerror("Error", "No data loaded.")

    def train_model(self):
        if hasattr(self, 'data_loader'):
            data = self.data_loader.get_data()
            X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2)
            model = RandomForestClassifier()
            trainer = ModelTrainer(model, X_train, y_train)
            trainer.fit()
            self.model_trainer = trainer
            messagebox.showinfo("Info", "Model trained successfully!")
        else:
            messagebox.showerror("Error", "No data loaded.")

    def evaluate_model(self):
        if hasattr(self, 'model_trainer'):
            data = self.data_loader.get_data()
            X_test, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2)[1:3]
            evaluator = ModelEvaluator(self.model_trainer.model, X_test, y_test)
            results = evaluator.evaluate()
            messagebox.showinfo("Model Evaluation", f"Model Accuracy: {results.get('accuracy', 'N/A'):.2f}\nROC-AUC: {results.get('roc_auc', 'N/A'):.2f}")
        else:
            messagebox.showerror("Error", "No model trained.")

    def tune_hyperparameters(self):
        if hasattr(self, 'data_loader'):
            data = self.data_loader.get_data()
            X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2)
            param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 15]}
            tuner = HyperparameterTuner(RandomForestClassifier(), param_grid, X_train, y_train)
            best_model = tuner.tune()
            self.model_trainer = ModelTrainer(best_model, X_train, y_train)
            self.model_trainer.fit()
            messagebox.showinfo("Info", "Hyperparameters tuned and model retrained!")
        else:
            messagebox.showerror("Error", "No data loaded.")

    def save_model(self):
        if hasattr(self, 'model_trainer'):
            file_path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib files", "*.joblib")])
            if file_path:
                saver = ModelSaver(self.model_trainer.model, file_path)
                saver.save()
                messagebox.showinfo("Info", "Model saved successfully!")
        else:
            messagebox.showerror("Error", "No model trained.")

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib")])
        if file_path:
            loader = Model

Loader(file_path)
            self.model = loader.load()
            messagebox.showinfo("Info", "Model loaded successfully!")

    def serve_model(self):
        if hasattr(self, 'model'):
            server = ModelServer(self.model)
            self.app = server.serve()
            # Note: For actual serving, run FastAPI server separately
            messagebox.showinfo("Info", "Model server started! (Start FastAPI separately)")
        else:
            messagebox.showerror("Error", "No model loaded.")

    def run_pipeline(self):
        if hasattr(self, 'data_loader'):
            config = PipelineConfig()
            pipeline = ML_Pipeline(self.data_loader, DataPreprocessor(self.data_loader.get_data()), FeatureEngineer(self.data_loader.get_data()), self.model_trainer, self.model_evaluator, config.config)
            evaluation = pipeline.run_pipeline()
            messagebox.showinfo("Pipeline Results", f"Pipeline run complete. Evaluation: {evaluation}")
        else:
            messagebox.showerror("Error", "No data loaded.")

    def save_pipeline(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".yaml", filetypes=[("YAML files", "*.yaml")])
        if file_path:
            config = PipelineConfig()
            config.save_config(file_path)
            messagebox.showinfo("Info", "Pipeline configuration saved!")

    def load_pipeline(self):
        file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if file_path:
            config = PipelineConfig()
            config.load_config(file_path)
            messagebox.showinfo("Info", "Pipeline configuration loaded!")

if __name__ == "__main__":
    root = tk.Tk()
    app = MLSystemGUI(root)
    root.mainloop()
```

### Summary

1. **Extended Library**:
   - Added support for loading from SQL and APIs.
   - Implemented data validation and augmentation.
   - Added hyperparameter tuning, model ensembles, and custom models.
   - Improved model evaluation with extended metrics and visualizations.
   - Enhanced pipeline management with automation, version control, and config management.
   - Added model deployment features, including serving, monitoring, and containerization.
   - Implemented comprehensive testing.

2. **Extended GUI**:
   - Added functionalities for advanced data handling, preprocessing, model management, and pipeline management.
   - Included model deployment options and configuration management.

This comprehensive extension of the ML system and GUI supports an end-to-end workflow for handling complex machine learning tasks, including advanced data management, model training, evaluation, deployment, and management.
