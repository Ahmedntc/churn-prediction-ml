import pathlib

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from models.arqMlp import ChurnMLPClassifier
from src.data.preprocess import build_preprocessor, load_data, prepare_feats, split_data
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

DATA_PATH = pathlib.Path("dataframe/processed/telco_clean.csv")
MODELS_DIR = pathlib.Path("modeldumps")
MODELS_DIR.mkdir(exist_ok=True)
EXPERIMENT_NAME = "churn-mlp"

# Testado com 1, 0.1, 0.01, 0.001
# 0.01 foi o melhor
LEARNING_RATE = 1e-2
MAX_EPOCHS = 100
# Testado com 3, 5 e 10
# 10 foi o melhor
PATIENCE = 10
# Testado com 8, 16, 32, 64, 128
# 64 foi o melhor
BATCH_SIZE = 64

FIXED_SEED = 12


def set_metricas(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


if __name__ == "__main__":
    np.random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)

    df = load_data(DATA_PATH)
    x, y = prepare_feats(df)
    x_train, x_test, y_train, y_test = split_data(x, y)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", ChurnMLPClassifier(
            hidden_dims=[64, 32, 16],
            dropout_rate=0.3,
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            random_state=FIXED_SEED,
        )),
    ])

    mlflow.set_experiment(EXPERIMENT_NAME)
    lr_str = str(LEARNING_RATE).replace(".", "_")

    with mlflow.start_run(run_name=f"MLP_lr{lr_str}_bs{BATCH_SIZE}_patience{PATIENCE}"):
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
        probs = pipeline.predict_proba(x_test)[:, 1]
        metrics = set_metricas(y_test.values, preds, probs)
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "hidden_dims": [64, 32, 16],
            "dropout_rate": 0.3,
        })
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        joblib.dump(pipeline, MODELS_DIR / f"pipeline_mlp_lr{lr_str}_bs{BATCH_SIZE}_patience{PATIENCE}.joblib")

        logger.info("roc_auc=%.4f | recall=%.4f | f1=%.4f | pr_auc=%.4f", metrics['roc_auc'], metrics['recall'], metrics['f1'], metrics['pr_auc'])
