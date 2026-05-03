from pathlib import Path

from fastapi import FastAPI

from churn_prediction.api.middleware import LatencyMiddleware
from churn_prediction.api.schemas import PredictRequest, PredictResponse
from churn_prediction.inference.predictor import ChurnPredictor
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MODEL_PATH = (
    _PROJECT_ROOT / "modeldumps" / "pipeline_mlp_lr0_01_bs64_patience10.joblib"
)

app = FastAPI(
    title="Churn Prediction API",
    description="API para previsão de churn de clientes de telecomunicações.",
    version="0.1.0",
)

app.add_middleware(LatencyMiddleware)

_predictor: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    """Carrega o modelo sob demanda para import/testes não dependerem do artefato."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor(model_path=str(_DEFAULT_MODEL_PATH))
    return _predictor


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    logger.info("Recebida requisição de predição")

    result = get_predictor().predict(
        payload.model_dump(
            by_alias=True,
        )
    )

    logger.info("Predição realizada com sucesso")

    return PredictResponse(**result)
