print("Importing packages...")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn

print("Loading dataset...")
creditcard = pd.read_csv('data/raw/creditcard.csv')

print("Scaling dataset...")
scaler = StandardScaler()
creditcard['scaled_amount'] = scaler.fit_transform(creditcard[['Amount']])
creditcard['scaled_time'] = scaler.fit_transform(creditcard[['Time']])

y = creditcard['Class']
X = creditcard.drop(['Class', 'Time', 'Amount'], axis=1)

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

print("Oversampling dataset...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

with mlflow.start_run():
    n_estimators = 100
    random_state = 0
    
    print("Saving parameters to MLflow...")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("oversampling_method", "SMOTE")

    print("Training model...")
    final_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    final_model.fit(X_train_smote, y_train_smote)

    print("Evaluating model...")
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("Saving metrics to MLflow...")
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1) 
    mlflow.log_metric("auc", auc)

    print("Saving model to MLflow...")
    mlflow.sklearn.log_model(final_model, name="random_forest_model")

    print("Model report:")
    print(classification_report(y_test, y_pred))
 
print("Execution complete!")