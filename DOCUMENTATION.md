# ChurnGuardPro: Complete Project Documentation, Architecture & Interview Guide

This document provides an exhaustive, beginner-friendly walkthrough of the **ChurnGuardPro** Customer Churn Prediction project. It covers core machine learning concepts, detailed line-by-line code explanations, Jupyter notebook prototyping steps, frontend UI architecture, and a curated list of technical interview questions and answers to help you explain this project confidently in interviews.

---

## 📖 Table of Contents
1. [Core ML & MLOps Concepts](#-core-ml--mlops-concepts)
2. [Jupyter Notebook & Exploratory Data Analysis (EDA)](#-jupyter-notebook--exploratory-data-analysis-eda)
3. [System Architecture & Directory Map](#-system-architecture--directory-map)
4. [Linkage & System Execution Flow (How files work together)](#-linkage--system-execution-flow-how-files-work-together)
5. [Line-by-Line Code Explanation](#-line-by-line-code-explanation)
    - [Utilities & Setup (`setup.py`, `logger.py`, `exception.py`, `utils.py`)](#utilities--setup)
    - [Ingestion Component (`data_ingestion.py`)](#ingestion-component)
    - [Transformation Component (`data_transformation.py`)](#transformation-component)
    - [Trainer Component (`model_tranier.py`)](#trainer-component)
    - [Prediction Pipeline (`prediction_pipeline.py`)](#prediction-pipeline)
    - [Flask Server Backend (`app.py`)](#flask-server-backend)
6. [Frontend Design & SSE Real-time Logs (`index.html`, `style.css`, `script.js`)](#-frontend-design--sse-real-time-logs)
7. [Curated Technical Interview Questions & Answers](#-curated-technical-interview-questions--answers)

---

## 🧠 Core ML & MLOps Concepts

Before looking at the code, it is important to understand the business and engineering concepts this project implements:

### 1. What is Customer Churn?
Customer churn occurs when a customer cancels their subscription or service. In business, acquiring a new customer is up to **5 times more expensive** than retaining an existing one. Pre-emptively identifying which customers are at risk of churning allows retention teams to intervene with promotions or targeted support.

### 2. Class Imbalance & SMOTE
In real-world churn datasets, the majority of customers do not churn (e.g., 75% Non-Churn vs. 25% Churn). If a model is trained directly on this imbalanced data, it will learn to predict the majority class ("No Churn") to achieve high accuracy, while failing to identify actual churning customers (poor **Recall**).
* **SMOTE (Synthetic Minority Over-sampling Technique)** solves this by analyzing the feature space of the minority class (Churn) and generating *synthetic* data points along the line segments joining its k-nearest neighbors.

### 3. Data Leakage
Data leakage happens when information from the test dataset is inadvertently used to train the machine learning model. For example, if you calculate the mean/median of the *entire* dataset to scale features before splitting it into train/test, information from the test set leaks into the training set.
* **Prevention:** Always fit transformers (like `StandardScaler` or imputers) **only** on the training set, and use those fitted metrics to transform the test set.

### 4. MLOps (Machine Learning Operations) with MLflow
In development, data scientists run many experiments with different algorithms and parameters. **MLflow** acts as an experiment tracking system. It records parameters (e.g., learning rates, number of estimators), performance metrics (Accuracy, Recall, F1), and serializes the trained model files (artifacts) to a local SQLite database (`mlflow.db`) for versioning and auditing.

---

## 📊 Jupyter Notebook & Exploratory Data Analysis (EDA)

The file `customer-churn.ipynb` contains the original exploratory work. Here is a step-by-step breakdown of how the data was analyzed and modeled:

### Step 1: Initial Imports & Loading Data
* **Libraries:** Imports `pandas` for data manipulation, `matplotlib.pyplot` and `seaborn` for plotting, and various classifiers from `sklearn`.
* **Data Inspection:** Reads `Telco-Customer-Churn.csv` using `pd.read_csv()`, checking data types and general statistics.

### Step 2: Data Cleaning
* **TotalCharges Parsing:** Evaluates why the `TotalCharges` column is imported as an object (string) rather than a float. The notebook reveals empty spaces (`" "`) representing brand new customers with `tenure == 0`. It resolves this using:
  ```python
  df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
  df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
  ```
* **ID Dropping:** The unique `customerID` string column is removed (`df.drop()`) as it provides no predictive value.

### Step 3: Visual EDA
* **Target Distribution:** Uses a pie chart to check the class balance of `Churn`, showing that **26.5%** of customers churned while **73.5%** did not.
* **Feature Categorization:** Plots categorical relations using seaborn `countplot()`. Key insights discovered:
  - Customers on a **Month-to-month contract** churn significantly more than those on one- or two-year contracts.
  - **Fiber optic** users churn at a much higher rate than DSL or non-internet users.
  - Customers using **Electronic check** as their payment method represent the highest churn density.

### Step 4: Preprocessing & Feature Engineering
* **Binary Encoding:** Maps the target `Churn` column to binary values: `{'Yes': 1, 'No': 0}`.
* **One-Hot Encoding:** Applies `pd.get_dummies()` to convert categorical attributes (like contract types, gender, and internet service options) into binary columns (`0` or `1`).
* **Feature Scaling:** Fits `StandardScaler()` only on the training feature matrix to scale continuous variables (`tenure`, `MonthlyCharges`, and `TotalCharges`) to a mean of `0` and a variance of `1`.

### Step 5: Model Experimentation & Evaluation
* Benchmarks different classifiers on the preprocessed training set:
  - **Logistic Regression:** Serves as a baseline model.
  - **Random Forest:** Checks for non-linear feature interactions and outputs the initial feature importance ranking using `model.feature_importances_`.
  - **Gradient Boosting Classifier:** Fitted with `n_estimators=150` and `max_depth=4`.
  - **XGBoost & CatBoost:** Benchmarks boosting libraries.
  - **SMOTE Oversampling:** Demonstrates how applying `SMOTE` increases the model's **Recall** (ability to identify actual churners) from ~52% to over ~69%.
  - **GridSearchCV:** Explains tuning hyperparameters like `max_depth` and `learning_rate` across multiple cross-validation folds.

---

## 🏗️ System Architecture & Directory Map

```
                          +-----------------------------------+
                          |     Raw CSV (data/Telco-...)      |
                          +-----------------+-----------------+
                                            |
                                  [ 1. Data Ingestion ]
                                            |
                          +-----------------+-----------------+
                          |  train.csv      |  test.csv       |
                          +--------+--------+--------+--------+
                                   |                 |
                        [ 2. Data Transformation (SMOTE) ]
                                   |                 |
                          +--------v--------+        |
                          |  X_train_res    |        |
                          |  y_train_res    |        |
                          +--------+--------+        |
                                   |                 |
                          [ 3. Model Training ] <----+
                                   |
                     +-------------+-------------+
                     |                           |
             +-------v-------+           +-------v-------+
             | Models/ (pkl) |           |   MLflow DB   |
             +-------+-------+           +---------------+
                     |
            [ 4. Flask Serving ] <------- HTTP POST /predict (Payload)
                     |
            [ 5. Bento Dashboard ]
```

---

## 🔗 Linkage & System Execution Flow (How files work together)

Understanding how files link and communicate is key to explaining this project from scratch. Here is the interaction map:

### 1. Installation & Package Binding
* **`requirements.txt` ➔ Pip Environment:** Lists dependencies like Flask, pandas, MLflow, and scikit-learn.
* **`setup.py` ➔ Python Interpreter:** Registers the `src/` folder as a local importable library named `customer_churn_prediction`. This lets python scripts run `from src.customer_churn_prediction.pipelines...` seamlessly without path import errors.

### 2. The Offline ML Training Flow (Data Prep to Model Save)
* **`main.py`** is the command-line orchestrator that imports and executes **`training_pipeline.py`**.
* **`training_pipeline.py`** coordinates three sequential steps:
  1. Instantiates `DataIngestion` (`data_ingestion.py`) to split the raw dataset inside `data/` into training and testing partitions within `artifacts/`.
  2. Instantiates `DataTransformation` (`data_transformation.py`) to load those CSVs, clean missing values, map targets, execute dummy one-hot column alignment, scale features, balance training classes via SMOTE, and save the configurations to `models/scaler.pkl` and `models/model_columns.pkl`.
  3. Instantiates `ModelTrainer` (`model_tranier.py`) to train the Gradient Boosting Classifier model, log metrics/parameters/artifacts to the MLflow local database (`mlflow.db`), and export `models/churn_model.pkl` alongside dashboard telemetry endpoints `metrics.json` and `feature_importance.json`.

### 3. The Web Dashboard Serving Flow (Inference API)
* **`app.py`** starts a Flask web server listening on port `5001`. It is the central junction linking UI interactions to Python code:
  * **Serving Layout:** Renders `templates/index.html` on requests to `/`.
  * **Static Asset Linkage:** `index.html` loads `static/style.css` (styling rules) and `static/script.js` (javascript interactions).
  * **Telemetry API Endpoint:** `index.html` and `script.js` fetch metrics from the `/metrics` and `/feature_importance` backend routes, which load metadata directly from `models/metrics.json` and `models/feature_importance.json` (created by the training pipeline).
  * **Prediction Routing:** When a user submits features on the UI, `script.js` posts a JSON payload to the `/predict` route. `app.py` passes this request to the `CustomData` class, transforms it into a dataframe, and invokes `PredictPipeline` (`prediction_pipeline.py`).
  * **Inference Pipeline:** `PredictPipeline` reads `models/scaler.pkl` and `models/model_columns.pkl` to scale and align dummy inputs to match training dimensions, runs inference via `models/churn_model.pkl`, and returns classification outputs and probabilities back to Flask to update the SVG risk meter.
  * **Log Streaming Stream:** Clicking "Retrain" posts a request to `/train`. `app.py` spawns a background subprocess running `python main.py` and pipes stdout to `models/training.log`. The `/train/stream` route reads this file line-by-line and pushes real-time text updates directly to the terminal container in `script.js` using Server-Sent Events.

---

## 💻 Line-by-Line Code Explanation

### Utilities & Setup

#### 1. `setup.py`
Allows the modular `src` folder to be packaged and installed as a local library in editable mode.
```python
from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Reads requirements.txt and filters out '-e .' if present."""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove trailing newline characters from each package name
        requirements = [req.replace("\n", "") for req in requirements]

        # Filter out '-e .' which is used internally to trigger setup.py
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
    return requirements

setup(
    name='customer_churn_prediction',
    version='0.0.1',
    author='Navya',
    author_email='navya@example.com',
    packages=find_packages(), # Automatically discovers packages inside the project directory
    install_requires=get_requirements('requirements.txt'),
)
```

#### 2. `src/customer_churn_prediction/logger.py`
Configures structured file-logging for runtime verification.
```python
import logging
import os
from datetime import datetime

# Generate a unique log filename based on the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create a log directory inside the workspace
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Set basic config for severity level, outputs, and message format
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
```

#### 3. `src/customer_churn_prediction/exception.py`
Captures python exceptions and returns line-level diagnostic error telemetry.
```python
import sys

def error_message_detail(error, error_detail: sys):
    """Formats a detailed error message containing script name and line number."""
    # Extract execution traceback info
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename # File name where exception occurred
    line_number = exc_tb.tb_lineno # Line number of the error
    
    error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    """Custom Exception class derived from python's base Exception."""
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
```

#### 4. `src/customer_churn_prediction/utils.py`
Core utilities to serialize and de-serialize python helper models.
```python
import os
import sys
import pickle
from src.customer_churn_prediction.exception import CustomException

def save_object(file_path, obj):
    """Serializes a python object to disk using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Loads a serialized python object from disk."""
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
```

---

### Ingestion Component

#### `src/customer_churn_prediction/components/data_ingestion.py`
Ingests raw datasets and executes stratified splits.
```python
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.customer_churn_prediction.exception import CustomException

class DataIngestion:
    def __init__(self):
        # Configure local relative files paths
        self.raw_data_path = "data/Telco-Customer-Churn.csv"
        self.train_data_path = "artifacts/train.csv"
        self.test_data_path = "artifacts/test.csv"

    def initiate_data_ingestion(self):
        try:
            # Load raw Telco csv dataset
            df = pd.read_csv(self.raw_data_path)
            
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            
            # Stratify train_test_split to maintain equal churn ratios in both subsets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn'])
            
            # Export partitioned subsets
            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)
            
            return self.train_data_path, self.test_data_path
        except Exception as e:
            raise CustomException(e, sys)
```

---

### Transformation Component

#### `src/customer_churn_prediction/components/data_transformation.py`
Applies preprocessors, handles dummy categories encoding, aligns schemas, and performs SMOTE.
```python
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.customer_churn_prediction.utils import save_object
from src.customer_churn_prediction.exception import CustomException

class DataTransformation:
    def __init__(self):
        # Define artifact destinations
        self.preprocessor_obj_file_path = "models/scaler.pkl"
        self.columns_obj_file_path = "models/model_columns.pkl"

    def clean_data(self, df):
        """Standardizes TotalCharges and drops redundant keys."""
        try:
            df = df.copy()
            # Convert TotalCharges to float representation, marking spaces as NaN
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            # Impute NaN spaces using median value
            df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
            
            # Drop customer ID column as it contains no predictive information
            if 'customerID' in df.columns:
                df = df.drop(columns=['customerID'])
                
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df = self.clean_data(train_df)
            test_df = self.clean_data(test_df)

            target_column = 'Churn'
            
            # Encode target variable from strings ('Yes'/'No') to integers (1/0)
            train_df[target_column] = train_df[target_column].map({'Yes': 1, 'No': 0})
            test_df[target_column] = test_df[target_column].map({'Yes': 1, 'No': 0})

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Categorical vs Numerical feature identification
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            # Execute pandas One-Hot Dummy encoding manually
            X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
            X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

            # Align feature dimensions: inserts columns with 0 if present in train but missing in test
            X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

            # Cast boolean dummy variables to int64 to ensure MLflow/sklearn typing compatibility
            for col in X_train_encoded.columns:
                if X_train_encoded[col].dtype == bool:
                    X_train_encoded[col] = X_train_encoded[col].astype(np.int64)
                    X_test_encoded[col] = X_test_encoded[col].astype(np.int64)

            # Feature Scaling (Fit only on training data to prevent data leakage)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_encoded)
            X_test_scaled = scaler.transform(X_test_encoded)

            # Balance classes using SMOTE
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

            # Save the preprocessor scaler and column schema configurations
            save_object(self.preprocessor_obj_file_path, scaler)
            save_object(self.columns_obj_file_path, X_train_encoded.columns.tolist())

            return X_train_res, y_train_res, X_test_scaled, y_test, self.preprocessor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)
```

---

### Trainer Component

#### `src/customer_churn_prediction/components/model_tranier.py`
Trains the core model and tracks metrics/artifacts with MLflow.
```python
import os
import sys
import json
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.customer_churn_prediction.utils import save_object, load_object
from src.customer_churn_prediction.exception import CustomException

class ModelTrainer:
    def __init__(self):
        self.model_file_path = "models/churn_model.pkl"
        self.metrics_file_path = "models/metrics.json"
        self.columns_file_path = "models/model_columns.pkl"
        self.feature_importance_file_path = "models/feature_importance.json"

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)
            
            # Hyperparameter configs matching the optimized notebook settings
            params = {
                'n_estimators': 150,
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'random_state': 42
            }

            # Configure SQL-backed MLflow tracking DB
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("Telco_Customer_Churn_Prediction")

            # Initiate an MLflow logging run
            with mlflow.start_run() as run:
                model = GradientBoostingClassifier(**params)
                model.fit(X_train, y_train)

                # Generate predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                # Compute core evaluation metrics
                metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred)),
                    'recall': float(recall_score(y_test, y_pred)),
                    'f1': float(f1_score(y_test, y_pred)),
                    'roc_auc': float(roc_auc_score(y_test, y_prob))
                }

                # Log configurations and parameters to the MLflow database
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")

                # Serialize trained model object to disk for Flask serving
                save_object(self.model_file_path, model)

                # Export metrics to JSON format for UI telemetry retrieval
                with open(self.metrics_file_path, 'w') as f:
                    json.dump(metrics, f, indent=2)

                # Extract and save top 5 feature importances
                try:
                    columns = load_object(self.columns_file_path)
                    importances = model.feature_importances_.tolist()
                    # Map column names to feature importance coefficients
                    feat_imp = sorted(zip(columns, importances), key=lambda x: x[1], reverse=True)[:5]
                    with open(self.feature_importance_file_path, 'w') as f:
                        json.dump(feat_imp, f, indent=2)
                except Exception as ex:
                    print(f"Warning: Failed to save feature importances: {ex}")

                return metrics
        except Exception as e:
            raise CustomException(e, sys)
```

---

### Prediction Pipeline

#### `src/customer_churn_prediction/pipelines/prediction_pipeline.py`
Exposes core prediction pipeline methods, mapping UI JSON payloads to feature dataframes.
```python
import os
import sys
import pandas as pd
import numpy as np
from src.customer_churn_prediction.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Fetch pre-compiled models and encoders
        self.model_path = "models/churn_model.pkl"
        self.preprocessor_path = "models/scaler.pkl"
        self.columns_path = "models/model_columns.pkl"

    def predict(self, features_df):
        try:
            model = load_object(self.model_path)
            scaler = load_object(self.preprocessor_path)
            model_columns = load_object(self.columns_path)

            # Explicitly match identical categorical keys used in training dummy variables
            categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                'PaperlessBilling', 'PaymentMethod']

            # Cast dynamic inputs to appropriate numeric/string formats
            features_df['SeniorCitizen'] = features_df['SeniorCitizen'].astype(int)
            features_df['tenure'] = features_df['tenure'].astype(float)
            features_df['MonthlyCharges'] = features_df['MonthlyCharges'].astype(float)
            features_df['TotalCharges'] = features_df['TotalCharges'].astype(float)

            # One-Hot Encode features using pandas get_dummies
            features_encoded = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)

            # Re-index columns list to match identical training sequence schema
            features_aligned = features_encoded.reindex(columns=model_columns, fill_value=0)

            # Cast boolean values to int64
            for col in features_aligned.columns:
                if features_aligned[col].dtype == bool:
                    features_aligned[col] = features_aligned[col].astype(np.int64)

            # Apply fitted scaler transformations
            scaled_features = scaler.transform(features_aligned)

            # Generate prediction classifications and probability weights
            preds = model.predict(scaled_features)
            probs = model.predict_proba(scaled_features)[:, 1]

            return preds[0], probs[0]
        except Exception as e:
            raise Exception(f"Prediction failed: {e}")

class CustomData:
    """Helper class to map user inputs to a structured pandas DataFrame."""
    def __init__(self, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
                 MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
                 TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                 PaymentMethod, MonthlyCharges, TotalCharges):
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise Exception(f"Converting data to dataframe failed: {e}")
```

---

### Flask Server Backend

#### `app.py`
Provides backend API endpoints, handles real-time retraining log streams, and serves static files.
```python
import os
import sys
import json
import time
import subprocess
from flask import Flask, render_template, request, jsonify, Response
from src.customer_churn_prediction.pipelines.prediction_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

# Constants pointing to local files
METRICS_FILE = "models/metrics.json"
FEATURE_IMPORTANCE_FILE = "models/feature_importance.json"
TRAINING_LOG_FILE = "models/training.log"

def get_saved_metrics():
    """Loads metrics from disk, falling back to defaults if not found."""
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"accuracy": 0.78, "precision": 0.57, "recall": 0.69, "f1": 0.62, "roc_auc": 0.84}

@app.route('/')
def home():
    """Serves the main bento dashboard template."""
    metrics = get_saved_metrics()
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles POST predictions from the front-end dashboard form."""
    try:
        data = request.get_json()
        
        # Instantiate CustomData helper with requested payload parameters
        custom_data = CustomData(
            gender=data.get('gender'),
            SeniorCitizen=data.get('SeniorCitizen'),
            Partner=data.get('Partner'),
            Dependents=data.get('Dependents'),
            tenure=data.get('tenure'),
            PhoneService=data.get('PhoneService'),
            MultipleLines=data.get('MultipleLines'),
            InternetService=data.get('InternetService'),
            OnlineSecurity=data.get('OnlineSecurity'),
            OnlineBackup=data.get('OnlineBackup'),
            DeviceProtection=data.get('DeviceProtection'),
            TechSupport=data.get('TechSupport'),
            StreamingTV=data.get('StreamingTV'),
            StreamingMovies=data.get('StreamingMovies'),
            Contract=data.get('Contract'),
            PaperlessBilling=data.get('PaperlessBilling'),
            PaymentMethod=data.get('PaymentMethod'),
            MonthlyCharges=data.get('MonthlyCharges'),
            TotalCharges=data.get('TotalCharges')
        )
        
        df = custom_data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        
        # Execute prediction pipeline
        pred, prob = predict_pipeline.predict(df)
        
        return jsonify({
            "status": "success",
            "prediction": "Churn" if pred == 1 else "No Churn",
            "churn_probability": round(float(prob), 3)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/train', methods=['POST'])
def train_model():
    """Triggers the model training script in a background subprocess."""
    try:
        os.makedirs("models", exist_ok=True)
        # Clear previous logs if they exist
        if os.path.exists(TRAINING_LOG_FILE):
            open(TRAINING_LOG_FILE, 'w').close()
            
        # Spawn 'main.py' as a background subprocess, logging output to training.log
        with open(TRAINING_LOG_FILE, 'w') as log_file:
            subprocess.Popen(
                [sys.executable, "-u", "main.py"],
                stdout=log_file,
                stderr=log_file,
                bufsize=1
            )
        return jsonify({"status": "success", "message": "Training started"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/train/stream')
def train_stream():
    """Streams training log console outputs back to the client using Server-Sent Events."""
    def generate():
        # Open training.log and yield new lines as they are written
        if not os.path.exists(TRAINING_LOG_FILE):
            with open(TRAINING_LOG_FILE, 'w') as f:
                f.write("Initializing logs...\n")
                
        with open(TRAINING_LOG_FILE, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.5) # Wait for new entries
                    continue
                yield f"data: {line}\n\n"
                if "Model Training completed" in line or "Error executing pipeline" in line:
                    break
                    
    # Return response as 'text/event-stream' to notify client of real-time server events
    return Response(generate(), mimetype='text/event-stream')

@app.route('/metrics')
def get_metrics_endpoint():
    """Exposes current metrics as an API endpoint."""
    return jsonify(get_saved_metrics())

@app.route('/feature_importance')
def get_feature_importance():
    """Exposes top feature importances for frontend charting."""
    if os.path.exists(FEATURE_IMPORTANCE_FILE):
        try:
            with open(FEATURE_IMPORTANCE_FILE, 'r') as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    # Fallback default feature importances
    return jsonify([
        ["Contract_Two year", 0.18],
        ["tenure", 0.15],
        ["InternetService_Fiber optic", 0.12],
        ["Contract_One year", 0.08],
        ["TotalCharges", 0.07]
    ])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
```

---

## 🎨 Frontend Design & SSE Real-time Logs

The frontend architecture consists of three files: `templates/index.html` (layout structure), `static/style.css` (visual system and tokens), and `static/script.js` (dynamic chart bindings and event handlers).

### 1. Layout Structure (`templates/index.html`)
The page utilizes a sidebar navigation system mapped to distinct content divisions via `tab-content` class variables:
* **Sidebar (`<aside class="sidebar">`):** Displays the branding logo area, navigation tab links (Overview, Predictor, Model Training, EDA Analysis), and a system status footer showing developer details (Navya).
* **Overview Tab Content:** Prepares bento card metrics slots for model performance rates (Accuracy, Precision, Recall, F1) alongside HTML5 canvas tags (`<canvas>`) targeted by Chart.js.
* **Predictor Tab Content:** Contains a structured demographic form. When submitted, the backend prediction outputs are displayed inside a dynamic radial SVG circular gauge:
  ```html
  <svg class="progress-ring" width="160" height="160">
      <circle class="progress-ring-circle-bg" stroke="rgba(255,255,255,0.05)" stroke-width="12" fill="transparent" r="70" cx="80" cy="80"/>
      <circle class="progress-ring-circle" id="risk-circle" stroke="var(--primary)" stroke-width="12" fill="transparent" r="70" cx="80" cy="80"/>
  </svg>
  ```
* **Model Training Content:** Renders a terminal console box (`<div class="terminal-body" id="train-terminal">`) to display streaming build output.

---

### 2. Styling System (`static/style.css`)
Implements a premium dark theme styling architecture:
* **Design Tokens & Variables:** Root configurations standardize modern aesthetics, using Outfit typography, translucent glassmorphism cards, and specific HSL accent colors:
  ```css
  :root {
      --bg-primary: #121214; 
      --bg-secondary: rgba(24, 24, 28, 0.75); 
      --border-color: rgba(255, 255, 255, 0.1);
      --primary: #38bdf8; /* Sky blue */
      --success: #10b981; /* Emerald green */
      --danger: #ef4444; /* Ruby red */
      --font-stack: 'Outfit', -apple-system, sans-serif;
  }
  ```
* **Bento Grid Layout:** Implements CSS Grid to construct responsive dashboards:
  ```css
  .bento-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      grid-auto-rows: minmax(130px, auto);
      gap: 1.5rem;
  }
  .chart-cell-2x2 { grid-column: span 2; grid-row: span 2; }
  ```
* **Micro-Animations:** Adds smooth hover effects using transition properties:
  ```css
  .bento-cell:hover {
      transform: translateY(-4px);
      box-shadow: 0 0 25px 2px rgba(255, 255, 255, 0.15);
  }
  ```

---

### 3. JavaScript Logic (`static/script.js`)
Handles interactivity, Chart.js renderings, asynchronous fetch operations, and Server-Sent Events (SSE).

* **Tab Switching Control:** Listens for sidebar nav clicks to toggle the active tab view:
  ```javascript
  document.querySelectorAll('.nav-item').forEach(item => {
      item.addEventListener('click', function(e) {
          e.preventDefault();
          // Remove active classes from all navs and contents
          document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
          document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active-content'));
          // Set current active link and tab content
          this.classList.add('active');
          document.getElementById(this.getAttribute('data-tab')).classList.add('active-content');
      });
  });
  ```
* **Chart.js Renders:** Creates 4 charts (Pie, Bar, Horizontal Bar, Line) using Chart.js configurations. For example, rendering the Churn by Internet Type horizontal bar chart:
  ```javascript
  new Chart(ctx, {
      type: 'bar',
      data: {
          labels: ['DSL', 'Fiber Optic', 'No Service'],
          datasets: [{ data: [18.9, 41.8, 7.4], backgroundColor: '#38bdf8' }]
      },
      options: { indexAxis: 'y', responsive: true }
  });
  ```
* **Predictor Submission & Circle Gauge Math:** Prevents default submission, serializes form fields into a JSON payload, posts to `/predict`, and calculates the circle's SVG stroke offset based on the returned probability:
  ```javascript
  const circle = document.getElementById('risk-circle');
  const radius = circle.r.baseVal.value;
  const circumference = 2 * Math.PI * radius;
  
  circle.style.strokeDasharray = `${circumference} ${circumference}`;
  
  function setProgress(percent) {
      const offset = circumference - (percent * circumference);
      circle.style.strokeDashoffset = offset;
  }
  ```
  - Low Risk (<30%): Highlights the gauge in Green.
  - Medium Risk (30%–70%): Highlights the gauge in Orange.
  - High Risk (>70%): Highlights the gauge in Red.
* **Log Streaming EventSource (SSE):** Listens to `/train/stream` and updates the console log output:
  ```javascript
  function triggerModelTraining() {
      fetch('/train', { method: 'POST' }).then(() => {
          const terminal = document.getElementById('train-terminal');
          const eventSource = new EventSource('/train/stream');
          
          eventSource.onmessage = function(e) {
              terminal.innerHTML += e.data + '<br>';
              terminal.scrollTop = terminal.scrollHeight; // Auto-scroll
          };
      });
  }
  ```

---

## 🙋 Curated Technical Interview Questions & Answers

These questions are commonly asked by ML/Data Science interviewers:

### Q1: What is Data Leakage and how did you prevent it in your pipelines?
* **Answer:** Data leakage happens when training data contains information about the target variable that won't be available during inference (e.g. leaking test data statistics into training features). In this project, I prevented it by strictly fitting my preprocessing `StandardScaler` **only on the training dataset** (`scaler.fit_transform`). The validation features are scaled using only the already-fitted transform properties (`scaler.transform`).

### Q2: Why did you use SMOTE? How does it differ from simple random oversampling?
* **Answer:** Telco Customer Churn datasets are highly imbalanced, meaning the model might default to predicting "No Churn" to optimize accuracy while missing churners. Simple random oversampling duplicates minority samples, which can lead to overfitting. **SMOTE** avoids this by creating **synthetic samples**. It identifies the k-nearest neighbors for minority samples, calculates feature space distances, and places new synthetic coordinate points along those lines.

### Q3: How do you handle categorical values during production inference? What if the user submits a missing category?
* **Answer:** During training, `pd.get_dummies` is used, and the final list of columns is serialized to disk as `model_columns.pkl`. During inference, the input payload is dummy-encoded and then **re-indexed** against the saved list of training columns (`features_aligned = features_encoded.reindex(columns=model_columns, fill_value=0)`). If a category is missing in the production payload, it safely defaults to `0`, ensuring consistent input shapes for the standard scaler.

### Q4: Why did you configure MLflow with a SQL database backend (`sqlite:///mlflow.db`) instead of the default local filesystem storage?
* **Answer:** By default, MLflow logs runs directly as text files on the filesystem. However, this structure is prone to deprecation issues and lock conflicts on Windows. Using a SQLite database backend enables structured SQL queries, relational run-history storage, and faster experiment logging.

### Q5: How did you implement real-time log streaming from Python to the UI?
* **Answer:** I used **Server-Sent Events (SSE)** via Flask's `Response` object and the `text/event-stream` mime-type. Clicking "Retrain" launches a background `subprocess.Popen` running `main.py` which writes stdout to `training.log`. Flask reads this log file line-by-line and yields chunks back to the client. On the frontend, a JavaScript `EventSource` listens to this stream and appends incoming lines to the terminal UI dynamically.

### Q6: Why did you choose a Gradient Boosting Classifier for this task?
* **Answer:** Gradient Boosting belongs to the boosting ensemble class. It fits sequential decision trees to minimize residual errors made by preceding trees. Telco churn data consists of non-linear categorical and numerical relationships (e.g., interaction between monthly charges, contract type, and tenure) that linear models struggle to map. In our notebook benchmarks, Gradient Boosting outperformed simpler models on ROC-AUC and Recall.

### Q7: If a model has high Accuracy but low Recall, what does that mean in the context of Customer Churn? Which metric is more important here?
* **Answer:** High Accuracy but low Recall means the model is predicting "No Churn" for almost everyone. It looks highly accurate because the majority of customers don't churn, but it **fails to detect the actual churners**. In churn prediction, **Recall is more important**. Failing to identify a customer who is about to churn (False Negative) costs the business subscription revenue, whereas misidentifying a loyal customer as a churn risk (False Positive) only costs a small promotional discount.
