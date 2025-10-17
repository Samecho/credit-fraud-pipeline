print("Importing packages...")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

import joblib

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

print("Training model...")
final_model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
final_model.fit(X_train_smote, y_train_smote)

print("Evaluating model...")
y_pred = final_model.predict(X_test)
print("Model report:")
print(classification_report(y_test, y_pred))

print("Saving model to models/best_model.pkl file...")
joblib.dump(final_model, 'models/best_model.pkl')

print("Execution complete!")