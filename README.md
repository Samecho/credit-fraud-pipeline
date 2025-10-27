# End-to-End Credit Card Fraud Detection Pipeline

## Overview

This project implements a complete, production-ready machine learning pipeline to detect fraudulent credit card transactions. The focus is not just on building an accurate model, but on establishing a robust, reproducible, and deployable MLOps workflow. The entire process, from data preprocessing and model training to API creation and containerization, is covered.

## Tech Stack

* Python 3.9+
* Data Handling: Pandas
* Machine Learning: Scikit-learn, XGBoost, Imbalanced-learn (for SMOTE)
* Experiment Tracking: MLflow
* API Framework: FastAPI with Pydantic
* Testing: Pytest
* Containerization: Docker
* Production Server: Gunicorn, Uvicorn
* Version Control: Git

## Setup and Installation

1. Clone the repository:
   `git clone https://github.com/Samecho/credit-fraud-pipeline`  
   `cd credit-fraud-pipeline`

2. Create and activate a virtual environment:  
   For Windows:  
   `python -m venv .venv`  
   `.\.venv\Scripts\activate`  

   For macOS/Linux
   `python3 -m venv .venv`
   `source .venv/bin/activate`

3. Install development dependencies:
   `pip install -r requirements.txt`

## Data

The project uses the "Credit Card Fraud Detection" dataset from Kaggle. Due to confidentiality, the primary features (V1-V28) are PCA-transformed.

* Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
* Action: Download creditcard.csv and place it inside the data/raw/ directory.

# Phase 1: Training the Champion Model

This phase covers data processing, model experimentation, and hyperparameter tuning to produce a final, production-ready model file.

## How to Run the Training Pipeline

The main training script executes a series of experiments and saves the best-performing model.

1. Execute the training script:
   From the project root directory, run:
   `python src/train.py`

2. What this script does:
   * Loads and preprocesses data using functions from src/pipeline.py.
   * Applies SMOTE to the training set to handle class imbalance.
   * Runs three separate experiments tracked by MLflow:
       1. A baseline RandomForestClassifier.
       2. An out-of-the-box XGBoost model.
       3. A final XGBoost model tuned with GridSearchCV to maximize Recall.
   * Saves the final champion model as models/champion_model.pkl.

3. Review Experiment Results (Optional):
   To view a detailed dashboard comparing all experiment runs, start the MLflow UI:
   mlflow ui
   Then, navigate to http://127.0.0.1:5000 in your browser.

# Phase 2: Running the Application with Docker

This phase covers building the Docker image and running the containerized FastAPI service.

## Prerequisites

* Docker Desktop is installed and running.
* You have successfully run the training pipeline at least once to generate the models/champion_model.pkl file.

## Step 1: Build the Docker Image

This command reads the Dockerfile and packages the application, the model, and all production dependencies into a self-contained image named credit-fraud-api.

`docker build -t credit-fraud-api .`

## Step 2: Run the Docker Container

This command starts the image as a running container, making the API service available.

`docker run -p 8000:8000 --name fraud-api-container credit-fraud-api`

* `-p 8000:8000`: Maps port 8000 on your computer to port 8000 inside the container.
* `--name fraud-api-container`: Assigns a convenient name to the running container.

Your terminal will now display live logs from the Gunicorn server. Leave this terminal running.

## Step 3: Access and Test Your API

Your service is now running in a fully isolated Linux container.

1. Open your web browser.
2. Navigate to http://127.0.0.1:8000. You should see the welcome message.
3. Navigate to http://127.0.0.1:8000/docs. This is the interactive API documentation (Swagger UI) where you can directly test the /predict endpoint.

## Step 4: Run Automated Tests (Optional)

While the container is running, you can verify its health by running the test suite from a new terminal.

1. Open a new terminal window.
2. Activate the virtual environment: `.\.venv\Scripts\activate`
3. Run pytest:  
   `pytest`  
   You should see all tests pass successfully.

## Step 5: Stop the Application

When you are finished, write in terminal:
   Stop the container if it's still running:  
   `docker stop fraud-api-container`

   Remove the container:  
   `docker rm fraud-api-container`

## Future TODOs

- Add LightGBM and compare with XGBoost  
- Deploy FastAPI API via Docker + Render  
- Add CI/CD (GitHub Actions) and endpoint tests  

