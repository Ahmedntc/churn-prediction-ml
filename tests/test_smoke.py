#Smoke test — verifica que o modelo carrega e faz uma predição sem quebrar.
from churn_prediction.inference.predictor import ChurnPredictor
SAMPLE_PAYLOAD = {
    "Gender": "Male",
    "Senior Citizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 89.9,
    "Total Charges": 1024.5,
}
def test_smokeLoadModel():
    predictor = ChurnPredictor("modeldumps/pipeline_mlp_lr0_01_bs64_patience10.joblib")
    assert predictor.model is not None
    
def test_smokeRetornaResult():
    predictor = ChurnPredictor("modeldumps/pipeline_mlp_lr0_01_bs64_patience10.joblib")
    result = predictor.predict(SAMPLE_PAYLOAD)
    assert result is not None
    
def test_smokeOutput():
    predictor = ChurnPredictor("modeldumps/pipeline_mlp_lr0_01_bs64_patience10.joblib")
    result = predictor.predict(SAMPLE_PAYLOAD)
    assert "churn_probability" in result
    assert "churn_prediction" in result
    assert "risk_level" in result