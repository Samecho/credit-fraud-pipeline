print("Importing packages...")

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from pipeline import load_and_preprocess_data, split_data, apply_smote 

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit-fraud-pipeline")

X, y = load_and_preprocess_data('data/raw/creditcard.csv')
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_smote, y_train_smote = apply_smote(X_train, y_train)

# 1 
print("Running Experiment 1: RandomForest......")
with mlflow.start_run(run_name="RandomForest"):
    n_estimators = 100
    random_state = 0
    
    print("Saving parameters to MLflow...")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    print("Training model...")
    model_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model_rf.fit(X_train_smote, y_train_smote)

    print("Evaluating model...")
    y_pred_rf = model_rf.predict(X_test)
    y_pred_proba_rf = model_rf.predict_proba(X_test)[:, 1]
    
    print("Saving metrics to MLflow...")
    mlflow.log_metric("recall", recall_score(y_test, y_pred_rf))
    mlflow.log_metric("precision", precision_score(y_test, y_pred_rf))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_rf))
    mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_proba_rf))

    print("Saving model to MLflow...")
    mlflow.sklearn.log_model(model_rf, "random_forest_model")

    print("Model report of Experiment 1:")
    print(classification_report(y_test, y_pred_rf))

# 2
print("Running Experiment 2: XGBoost (Default)......")
with mlflow.start_run(run_name="XGBoost"):
    n_estimators = 100
    random_state = 0

    print("Saving parameters to MLflow...")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    
    print("Training model...")
    model_xgb = XGBClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model_xgb.fit(X_train_smote, y_train_smote)

    print("Evaluating model...")
    y_pred_xgb = model_xgb.predict(X_test)
    y_pred_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

    print("Saving metrics to MLflow...")
    mlflow.log_metric("recall", recall_score(y_test, y_pred_xgb))
    mlflow.log_metric("precision", precision_score(y_test, y_pred_xgb))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_xgb))
    mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_proba_xgb))

    print("Saving model to MLflow...")
    mlflow.sklearn.log_model(model_xgb, "xgboost_model")
    
    print("Model report of Experiment 2:")
    print(classification_report(y_test, y_pred_xgb))

# 3
print("Running Experiment 3: Hyperparameter Tuning for XGBoost......")
with mlflow.start_run(run_name="XGBoost_Tuned"):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': list(range(4, 9)),
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb = XGBClassifier(random_state=0, n_jobs=-1)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='recall', verbose=1)

    print("Starting GridSearch...")
    grid_search.fit(X_train_smote, y_train_smote)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {best_params}")

    print("Saving parameters to MLflow...")
    mlflow.log_params(best_params)
    
    print("Evaluating best model...")
    y_pred_tuned = best_model.predict(X_test)
    y_pred_proba_tuned = best_model.predict_proba(X_test)[:, 1]

    print("Saving metrics to MLflow...")
    mlflow.log_metric("recall", recall_score(y_test, y_pred_tuned))
    mlflow.log_metric("precision", precision_score(y_test, y_pred_tuned))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred_tuned))
    mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_proba_tuned))

    print("Saving model to MLflow...")
    mlflow.sklearn.log_model(best_model, "tuned_xgboost_model")

    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/tuned_xgboost_model"
    mlflow.register_model(model_uri=model_uri, name="fraud-detector-model")

    print("Model report of Experiment 3:")
    print(classification_report(y_test, y_pred_tuned))

print("Execution complete!")