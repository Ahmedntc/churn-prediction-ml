from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    gender: str = Field(..., alias="Gender")
    senior_citizen: int = Field(..., alias="Senior Citizen", ge=0, le=1)
    partner: str = Field(..., alias="Partner")
    dependents: str = Field(..., alias="Dependents")
    tenure: int = Field(..., alias="Tenure Months")
    phone_service: str = Field(..., alias="Phone Service")
    multiple_lines: str = Field(..., alias="Multiple Lines")
    internet_service: str = Field(..., alias="Internet Service")
    online_security: str = Field(..., alias="Online Security")
    online_backup: str = Field(..., alias="Online Backup")
    device_protection: str = Field(..., alias="Device Protection")
    tech_support: str = Field(..., alias="Tech Support")
    streaming_tv: str = Field(..., alias="Streaming TV")
    streaming_movies: str = Field(..., alias="Streaming Movies")
    contract: str = Field(..., alias="Contract")
    paperless_billing: str = Field(..., alias="Paperless Billing")
    payment_method: str = Field(..., alias="Payment Method")
    monthly_charges: float = Field(..., alias="Monthly Charges", ge=0)
    total_charges: float = Field(..., alias="Total Charges", ge=0)

    model_config = {
        "populate_by_name": True,
    }


class PredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
