from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

VALID_DATA = {
    "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378, "V5": -0.338,
    "V6": 0.462, "V7": 0.239, "V8": 0.098, "V9": 0.363, "V10": 0.090,
    "V11": -0.551, "V12": -0.617, "V13": -0.991, "V14": -0.311, "V15": 1.468,
    "V16": -0.470, "V17": 0.207, "V18": 0.025, "V19": 0.403, "V20": 0.251,
    "V21": -0.018, "V22": 0.277, "V23": -0.110, "V24": 0.066, "V25": 0.128,
    "V26": -0.189, "V27": 0.133, "V28": -0.021,
    "scaled_amount": 0.244, "scaled_time": -1.996
}

KNOWN_FRAUD_DATA = {
    "V1": -2.312, "V2": 1.951, "V3": -1.609, "V4": 3.997, "V5": -0.522,
    "V6": -1.426, "V7": -2.537, "V8": 1.391, "V9": -2.770, "V10": -2.772,
    "V11": 3.202, "V12": -2.899, "V13": -0.595, "V14": -4.289, "V15": 0.389,
    "V16": -1.140, "V17": -2.830, "V18": -0.016, "V19": 0.416, "V20": 0.126,
    "V21": 0.517, "V22": -0.035, "V23": -0.465, "V24": 0.320, "V25": 0.044,
    "V26": 0.177, "V27": 0.261, "V28": -0.143,
    "scaled_amount": -0.353, "scaled_time": -1.023
}

def test_predict_normal_transaction():
    response = client.post("/predict", json=VALID_DATA)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["is_fraud"] == 0

def test_predict_fraud_transaction():
    response = client.post("/predict", json=KNOWN_FRAUD_DATA)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["is_fraud"] == 1

def test_predict_invalid_data():
    invalid_data = VALID_DATA.copy()
    del invalid_data["V10"]
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422