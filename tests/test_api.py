#Testes da API FastAPI — endpoints /health e /predict.
from fastapi.testclient import TestClient
from churn_prediction.api.main import app
client = TestClient(app)
SAMPLE_PAYLOAD = {
    "gender": "Male",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 89.9,
    "total_charges": 1024.5,
}

def testHealth():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def testPredict():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200


def testPredictFormat():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data


def testProbabilidade():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0


def testPrediction():

    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert data["churn_prediction"] in [0, 1]


def testRiskLevel():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert data["risk_level"] in ["low", "medium", "high"]


def testPayloadInvalido():
    response = client.post("/predict", json={"invalid": "payload"})
    assert response.status_code == 422