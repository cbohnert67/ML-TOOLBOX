# ML Toolbox, a PyCaret Clone

Ce projet est une implémentation d'un système de Machine Learning de bout en bout similaire à PyCaret. Il couvre la préparation des données, l'entraînement des modèles, l'évaluation, le déploiement et la gestion des pipelines.

## Structure du Projet situé dans le dossier SOURCE

- `data/`: Contient les données brutes et traitées.
- `notebooks/`: Contient les notebooks Jupyter pour l'exploration et l'analyse.
- `src/`: Contient le code source organisé par gestion des données, gestion des modèles, gestion des pipelines et déploiement des modèles.
- `tests/`: Contient les tests unitaires et d'intégration.
- `.gitignore`: Fichiers et répertoires à ignorer par Git.
- `requirements.txt`: Liste des dépendances Python.

SOURCE/
├── data/
│   ├── raw/
│   │   └── README.md
│   ├── processed/
│   │   └── README.md
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
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
├── tests/
│   ├── unit/
│   │   └── README.md
│   ├── integration/
│   │   └── README.md
│   └── README.md
├── .gitignore
├── requirements.txt
└── README.md

## Installation

Installez les dépendances avec pip :

```bash
pip install -r requirements.txt