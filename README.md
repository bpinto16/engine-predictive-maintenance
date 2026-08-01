# Capstone Project of Engine Predictive Maintenance by Bryan Pinto.

# Engine Predictive Maintenance using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)]()
[![DagsHub](https://img.shields.io/badge/DagsHub-MLOps-green)]()
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployment-yellow)]()

## Overview

Engine failures often occur without warning, resulting in unplanned downtime, increased maintenance costs, and reduced operational efficiency.

This project implements a complete Machine Learning and MLOps pipeline that predicts whether an engine is operating normally or requires maintenance based on six real-time sensor measurements.

The project goes beyond model development by implementing:

- Automated data registration
- Data preprocessing
- Experiment tracking
- Hyperparameter optimization
- Model selection
- Model versioning
- CI/CD automation
- Cloud deployment
- Interactive prediction application

---

# Features

- End-to-end MLOps workflow
- Automated GitHub Actions pipeline
- Dataset versioning using Hugging Face Dataset Hub
- Model versioning using Hugging Face Model Hub
- MLflow experiment tracking
- DagsHub remote experiment logging
- Hyperparameter tuning
- Multiple ensemble algorithms
- Recall-oriented threshold optimization
- Sensor validation before prediction
- Streamlit web application
- Cloud deployment using Hugging Face Spaces

---

# Project Structure

```
engine-predictive-maintenance/

│
├── .github/
│   └── workflows/
│       └── pipeline.yml
│
├── predictive_maintenance_project/
│
│   ├── data/
│   │     engine_data.csv
│   │
│   ├── model_building/
│   │     data_register.py
│   │     preparation.py
│   │     train.py
│   │
│   ├── deployment/
│   │     app.py
│   │     Dockerfile
│   │
│   ├── hosting/
│   │     hosting.py
│   │
│   └── requirements.txt
│
└── README.md
```

---

# Machine Learning Pipeline

The complete workflow consists of the following stages.

```
Raw Dataset
      │
      ▼
Dataset Registration
(Hugging Face Dataset Hub)
      │
      ▼
Data Cleaning
      │
      ▼
Train/Test Split
      │
      ▼
Feature Range Generation
      │
      ▼
Upload Processed Dataset
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Search
      │
      ▼
MLflow Tracking
      │
      ▼
Best Model Selection
      │
      ▼
Model Upload
(Hugging Face Model Hub)
      │
      ▼
Streamlit Deployment
(Hugging Face Spaces)
```

---

# Dataset

The model predicts engine condition using six sensor measurements.

| Feature | Description |
|----------|-------------|
| Engine rpm | Engine rotational speed |
| Lub oil pressure | Lubrication oil pressure |
| Fuel pressure | Fuel delivery pressure |
| Coolant pressure | Cooling system pressure |
| Lub oil temperature | Lubrication oil temperature |
| Coolant temperature | Engine coolant temperature |

Target:

```
0 = Normal Engine
1 = Faulty Engine
```

---

# Data Preparation

The preprocessing pipeline performs:

- Missing value detection
- Duplicate removal
- Stratified train/test split
- Physical sensor limit generation
- Operating range calculation
- Feature validation metadata creation
- Upload processed datasets to Hugging Face Dataset Hub

Unlike many predictive maintenance projects, statistical outliers are intentionally retained because they may represent genuine engine failures rather than erroneous observations.

---

# Model Development

Several supervised learning algorithms are evaluated.

- Decision Tree
- Bagging Classifier
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

Each model is trained inside a Scikit-Learn Pipeline consisting of:

```
StandardScaler
        │
        ▼
Classifier
```

This avoids data leakage during cross-validation.

---

# Hyperparameter Optimization

Each algorithm undergoes automated parameter search using:

- GridSearchCV
- Stratified K-Fold Cross Validation

Model performance is compared using:

- Recall
- Precision
- F1 Score
- ROC-AUC
- Average Precision
- Accuracy

---

# Threshold Optimization

Instead of using the default probability threshold of 0.5, this project optimizes the operating threshold.

The objective is to minimize costly false negatives while maintaining acceptable false positive rates.

```
Default Threshold = 0.50

Optimized Threshold = 0.39
```

This improves maintenance sensitivity by prioritizing fault detection.

---

# Experiment Tracking

Experiments are automatically logged using MLflow.

Tracked information includes:

- Parameters
- Metrics
- Cross-validation scores
- Model metadata
- Feature preprocessing pipeline

Remote tracking is synchronized with DagsHub.

---

# Model Validation

Before making predictions, the application performs two validation stages.

## Stage 1

Physical Limits

Detects impossible sensor values.

Example:

```
Engine RPM = -25
```

Prediction is blocked.

---

## Stage 2

Operating Limits

Detects unusual but physically possible readings.

Example:

```
Coolant Temperature
outside normal operating envelope
```

Prediction proceeds with a warning.

---

# Deployment

The trained model is downloaded dynamically from Hugging Face Model Hub.

The Streamlit application provides:

- Interactive sliders
- Sensor validation
- Prediction probability
- Maintenance recommendation
- Warning messages
- Operating range reference

---

# CI/CD Pipeline

GitHub Actions automates the complete workflow.

```
Push to main
      │
      ▼
Register Dataset
      │
      ▼
Data Preparation
      │
      ▼
Model Training
      │
      ▼
Experiment Logging
      │
      ▼
Model Upload
      │
      ▼
Deploy Streamlit Application
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<username>/engine-predictive-maintenance.git

cd engine-predictive-maintenance
```

Install dependencies.

```bash
pip install -r predictive_maintenance_project/requirements.txt
```

---

# Environment Variables

Create the following environment variables.

```text
HF_TOKEN=<your_huggingface_token>

DAGSHUB_USERNAME=<username>

DAGSHUB_TOKEN=<token>

MLFLOW_TRACKING_URI=http://localhost:5000
```

---

# Usage

## 1. Register Dataset

```bash
python predictive_maintenance_project/model_building/data_register.py
```

---

## 2. Prepare Dataset

```bash
python predictive_maintenance_project/model_building/preparation.py
```

---

## 3. Train Models

```bash
python predictive_maintenance_project/model_building/train.py
```

---

## 4. Launch Web Application

```bash
streamlit run predictive_maintenance_project/deployment/app.py
```

---

# Example

Input

| Sensor | Value |
|----------|------|
| Engine RPM | 780 |
| Lub Oil Pressure | 3.2 |
| Fuel Pressure | 7.4 |
| Coolant Pressure | 2.1 |
| Lub Oil Temp | 81 |
| Coolant Temp | 78 |

Prediction

```
Engine Status

Faulty

Probability: 0.87

Recommendation:
Schedule preventive maintenance.
```

---

# Technologies Used

- Python
- Scikit-Learn
- XGBoost
- MLflow
- DagsHub
- Hugging Face Hub
- Hugging Face Spaces
- Streamlit
- GitHub Actions
- Docker
- Pandas
- NumPy

---

# Future Improvements

- SHAP Explainability
- Feature Importance Dashboard
- Drift Detection

---

# Author

**Bryan Pinto**

Machine Learning • MLOps • Predictive Maintenance • Cloud Deployment
