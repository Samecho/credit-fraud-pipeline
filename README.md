# End-to-End Credit Card Fraud Detection Pipeline

## Overview
A complete machine learning system for detecting fraudulent credit card transactions.  
Focus: accuracy, reproducibility, and MLOps automation — from data ingestion to deployment prep.

## Tech Stack
- Python 3.9+
- Pandas, Scikit-learn, XGBoost, Imbalanced-learn (SMOTE), gridsearchCV, FastAPI
- MLflow (experiment tracking)
- Virtual environment (venv)

## Project Structure
credit-fraud-pipeline/  
├── .gitignore  
├── README.md  
├── requirements.txt  
├── data/raw/creditcard.csv (not committed)  
├── mlruns/ (MLflow tracking data)  
├── models/ (saved model artifacts)  
├── notebooks/01_data_exploration.ipynb  
├── src/   
│   ├── api.py
│   ├── pipeline.py (data processing)  
│   └── train.py (training + MLflow logging)  
└── tests/ (future)

## Setup
1. Clone repo  
   `git clone https://github.com/Samecho/credit-fraud-pipeline`  
   `cd credit-fraud-pipeline`

2. Create virtual environment  
   - Windows: `python -m venv .venv && .venv\Scripts\activate`  
   - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`

3. Install dependencies  
   `pip install -r requirements.txt`

## Data
Dataset: [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Features: V1–V28 (PCA), Time, Amount  
- Target: Class (1 = Fraud, 0 = Normal)  
- Place `creditcard.csv` in `data/raw/`.

## Workflow (Phase 1)
1. EDA: checked imbalance, scaled Time & Amount  
2. Preprocessing: scaling, stratified split, SMOTE  
3. MLflow tracking for all runs  
4. Models: RandomForest (baseline), XGBoost (default + tuned)  
5. GridSearchCV tuned XGBoost for Recall priority  
6. Registered tuned model in MLflow Model Registry

## Results
| Model | Recall | Notes |
|--------|---------|-------|
| RandomForest | ~0.78 | Baseline |
| XGBoost (default) | ~0.81 | Better |
| XGBoost (tuned) | ~0.82 | Final |

Best Params:  
`{'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 300, 'subsample': 1.0}`  

All experiments tracked in MLflow.

## Run the Pipeline
`python src/train.py`  

To view MLflow UI:  
`mlflow ui` → visit http://127.0.0.1:5000

## In progress (Phase 2)
- Implement FastAPI (`src/api.py`) to serve MLflow model  
- Add tests (Pytest)  
- Add config + logging system
