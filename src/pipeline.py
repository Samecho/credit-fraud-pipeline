import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)

    print("Scaling dataset...")
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler.fit_transform(df[['Time']])
    
    y = df['Class']
    X = df.drop(['Class', 'Time', 'Amount'], axis=1)
    
    return X, y

def split_data(X, y):
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    print("Oversampling dataset...")
    smote = SMOTE(random_state=0)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote