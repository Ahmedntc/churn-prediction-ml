import joblib
import pandas as pd


class ChurnPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, payload: dict) -> dict:
        df = pd.DataFrame([payload])

        df["Tenure Months"] = pd.to_numeric(df["Tenure Months"])
        df["Monthly Charges"] = pd.to_numeric(df["Monthly Charges"])
        df["Total Charges"] = pd.to_numeric(df["Total Charges"])

        categorical_cols = [
            "Gender",
            "Senior Citizen",
            "Partner",
            "Dependents",
            "Phone Service",
            "Multiple Lines",
            "Internet Service",
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Contract",
            "Paperless Billing",
            "Payment Method",
        ]

        for col in categorical_cols:
            df[col] = df[col].astype(str).str.strip()

        prediction = int(self.model.predict(df)[0])

        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(df)[0][1])
        else:
            probability = float(prediction)

        return {
            "churn_probability": probability,
            "churn_prediction": prediction,
            "risk_level": self._get_risk_level(probability),
        }

    @staticmethod
    def _get_risk_level(probability: float) -> str:
        if probability >= 0.7:
            return "high"
        if probability >= 0.4:
            return "medium"
        return "low"
