import os

# Define the directory structure
structure = {
    "src": {
        "data_management": ["data_loader.py", "data_preprocessor.py", "feature_engineer.py", "README.md"],
        "model_management": ["model_trainer.py", "model_evaluator.py", "model_selector.py", "README.md"],
        "pipeline_management": ["ml_pipeline.py", "pipeline_config.py", "README.md"],
        "model_deployment": ["model_saver.py", "model_loader.py", "model_server.py", "README.md"],
        "README.md": [""]
    }
}

# Function to create directories and files
def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            os.makedirs(os.path.dirname(os.path.join(path, content[0])), exist_ok=True)
            for file in content:
                if file:
                    os.makedirs(path, exist_ok=True)
                    open(os.path.join(path, file), 'a').close()

# Define the base path
base_path = "."

# Create the directory structure
create_structure(base_path, structure)

# Write the directory structure to README.md
readme_content = """src/
│   ├── data_management/
│   │   ├── data_loader.py
│   │   ├── data_preprocessor.py
│   │   ├── feature_engineer.py
│   │   └── README.md
│   ├── model_management/
│   │   ├── model_trainer.py
│   │   ├── model_evaluator.py
│   │   ├── model_selector.py
│   │   └── README.md
│   ├── pipeline_management/
│   │   ├── ml_pipeline.py
│   │   ├── pipeline_config.py
│   │   └── README.md
│   ├── model_deployment/
│   │   ├── model_saver.py
│   │   ├── model_loader.py
│   │   ├── model_server.py
│   │   └── README.md
│   └── README.md
"""

with open(os.path.join(base_path, "src/README.md"), "w") as readme_file:
    readme_file.write(readme_content)