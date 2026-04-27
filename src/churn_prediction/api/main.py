from fastapi import FastAPI

from churn_prediction.api.middleware import LatencyMiddleware
from churn_prediction.api.schemas import PredictRequest, PredictResponse
from churn_prediction.inference.predictor import ChurnPredictor
from churn_prediction.utils.logging import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="API para previsão de churn de clientes de telecomunicações.",
    version="0.1.0",
)

app.add_middleware(LatencyMiddleware)

predictor = ChurnPredictor(
    model_path="models/randomforest.joblib"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    logger.info("Recebida requisição de predição")

    result = predictor.predict(
        payload.model_dump(
            by_alias=True,
        )
    )

    logger.info("Predição realizada com sucesso")

    return PredictResponse(**result)