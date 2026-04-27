from churn_prediction.api.schemas import PredictRequest


def test_predict_request_schema():
    payload = PredictRequest(
        **{
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
    )

    assert payload.tenure == 12
    assert payload.monthly_charges == 89.9