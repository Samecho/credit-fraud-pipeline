# End-to-End Credit Card Fraud Detection Pipeline

## Overview

This project implements an end-to-end machine learning pipeline for detecting fraudulent credit card transactions.

The focus is not only on model performance, but on building a reproducible and production-ready workflow, including:

* Data preprocessing
* Model experimentation and tracking
* Model selection
* API development
* Containerization
* Cloud deployment (Azure)

The final model is exposed as a FastAPI service and deployed in a Docker container to Microsoft Azure.

---

## Tech Stack

**Language**

* Python 3.9+

**Data & ML**

* Pandas
* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)

**Experiment Tracking**

* MLflow

**Backend / API**

* FastAPI
* Pydantic

**Testing**

* Pytest

**Production & Deployment**

* Docker
* Gunicorn
* Uvicorn
* Microsoft Azure (Container Deployment)

**Version Control**

* Git

---

## Dataset

* Source: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* Features V1–V28 are PCA-transformed due to confidentiality.

To reproduce locally:

1. Download `creditcard.csv`
2. Place it under:

```
data/raw/creditcard.csv
```

---

# Phase 1 – Model Training & Experimentation

The training pipeline performs preprocessing, model experimentation, and hyperparameter tuning.

## Run Training Pipeline

From project root:

```
python src/train.py
```

### What the training pipeline does

* Loads and preprocesses data (`src/pipeline.py`)
* Applies SMOTE to handle class imbalance
* Runs experiments tracked by MLflow:

  * RandomForest baseline
  * Default XGBoost
  * Tuned XGBoost (GridSearchCV optimizing Recall)
* Saves the final model to:

```
models/champion_model.pkl
```

## View MLflow Experiments (Optional)

```
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

# Phase 2 – API Service (FastAPI)

The trained model is served through a FastAPI application.

### Start locally (development mode)

```
uvicorn src.api.main:app --reload
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

---

# Phase 3 – Docker Containerization

The API is containerized for reproducibility and deployment.

## Build Docker Image

```
docker build -t credit-fraud-api .
```

## Run Container Locally

```
docker run -p 8000:8000 --name fraud-api-container credit-fraud-api
```

The service will be available at:

```
http://127.0.0.1:8000/docs
```

---

# Phase 4 – Cloud Deployment (Azure)

The Dockerized FastAPI service has been successfully deployed to Microsoft Azure as a containerized application.

### Deployment Highlights

* Docker image built locally
* Image pushed to Azure Container Registry
* Service deployed to Azure container service
* Public endpoint exposed for inference
* Production server: Gunicorn + Uvicorn workers

This demonstrates:

* Container-based deployment workflow
* Cloud-native API serving
* Environment portability between local and cloud

---

# Testing

Run automated tests:

```
pytest
```

Test coverage includes:

* API endpoint validation
* Model loading verification
* Prediction response structure

---

# What This Project Demonstrates

* Handling extreme class imbalance with SMOTE
* Experiment tracking with MLflow
* Hyperparameter tuning with GridSearchCV
* Model selection based on Recall optimization
* Production API design using FastAPI
* Containerized deployment with Docker
* Cloud deployment to Azure
* Reproducible ML workflow

---

# Future Improvements

* Add LightGBM comparison
* Implement CI/CD pipeline (GitHub Actions)
* Add integration tests for containerized API
* Add monitoring & logging for production usage
