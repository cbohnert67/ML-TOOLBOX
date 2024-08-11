import os
import pandas as pd
import requests
from io import StringIO

def download_and_save_dataset(url, file_path, column_names=None):
    """
    Download a dataset from a URL and save it to a specified file path.

    Parameters:
    - url: str, the URL to the dataset.
    - file_path: str, the path to save the dataset.
    - column_names: list of str, optional, the names of the columns.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download dataset from {url}")
    
    data = StringIO(response.text)
    df = pd.read_csv(data, header=None, names=column_names)
    df.to_csv(file_path, index=False)
    print(f"Dataset saved to {file_path}")

# Dossier de destination
raw_data_dir = os.path.join('../data', 'raw')
os.makedirs(raw_data_dir, exist_ok=True)

# Jeux de données à télécharger
datasets = {
    "wine_quality": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "file_path": os.path.join(raw_data_dir, "wine_quality.csv"),
        "column_names": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", 
                         "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
    },
    "boston_housing": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
        "file_path": os.path.join(raw_data_dir, "boston_housing.csv"),
        "column_names": ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
    },
    "auto_mpg": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
        "file_path": os.path.join(raw_data_dir, "auto_mpg.csv"),
        "column_names": ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
    },
    "concrete_compressive_strength": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
        "file_path": os.path.join(raw_data_dir, "concrete_compressive_strength.csv"),
        "column_names": ["cement", "slag", "fly ash", "water", "superplasticizer", "coarse aggregate", "fine aggregate", "age", "compressive strength"]
    }
}

# Télécharger et sauvegarder les jeux de données
for dataset_name, dataset_info in datasets.items():
    download_and_save_dataset(dataset_info["url"], dataset_info["file_path"], dataset_info.get("column_names"))