#Testes de schema — validação do Pydantic e Pandera.
import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.pandas import Check, Column, DataFrameSchema
from pydantic import ValidationError

from churn_prediction.api.schemas import PredictRequest

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
DATAFRAME_SCHEMA = DataFrameSchema({
    "Tenure Months": Column(float, Check.ge(0)),
    "Monthly Charges": Column(float, Check.ge(0)),
    "Total Charges": Column(float, Check.ge(0)),
    "Gender": Column(str),
    "Senior Citizen": Column(int, Check.isin([0, 1])),
    "Partner": Column(str),
    "Dependents": Column(str),
    "Phone Service": Column(str),
    "Multiple Lines": Column(str),
    "Internet Service": Column(str),
    "Online Security": Column(str),
    "Online Backup": Column(str),
    "Device Protection": Column(str),
    "Tech Support": Column(str),
    "Streaming TV": Column(str),
    "Streaming Movies": Column(str),
    "Contract": Column(str),
    "Paperless Billing": Column(str),
    "Payment Method": Column(str),
})

def test_schema_valido():
    payload = PredictRequest(**SAMPLE_PAYLOAD)
    assert payload.tenure == 12
    assert payload.monthly_charges == 89.9


def test_predict_cobranca_neg():
    invalid = SAMPLE_PAYLOAD.copy()
    invalid["Monthly Charges"] = -10.0
    with pytest.raises(ValidationError):
        PredictRequest(**invalid)

def test_pandera_valido():
    df = pd.DataFrame([{
        "Tenure Months": 12.0,
        "Monthly Charges": 89.9,
        "Total Charges": 1024.5,
        "Gender": "Male",
        "Senior Citizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
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
    }])
    DATAFRAME_SCHEMA.validate(df)

def test_pandera_cobranca_neg():
    df = pd.DataFrame([{
        "Tenure Months": 12.0,
        "Monthly Charges": -10.0,
        "Total Charges": 1024.5,
        "Gender": "Male",
        "Senior Citizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
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
    }])
    with pytest.raises(pa.errors.SchemaError):
        DATAFRAME_SCHEMA.validate(df)
