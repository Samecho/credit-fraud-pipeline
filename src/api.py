from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd

class TransactionFeatures(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    scaled_amount: float
    scaled_time: float

app = FastAPI(title="Credit Fraud Detection API")

model_name = "fraud-detector-model"
model_stage = "Production"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")
print(f"Successfully loaded model '{model_name}' version from stage '{model_stage}'.")

@app.get("/")
def read_root():
    return {"message": "Credit Fraud Detection API Executing..."}

@app.post("/predict")
def predict(features: TransactionFeatures):
    try:
        
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        prediction = model.predict(df)
        
        is_fraud = int(prediction[0])
        
        return {"is_fraud": is_fraud}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))