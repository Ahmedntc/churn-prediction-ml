import pathlib
import torch
import torch.nn as nn
import numpy as np
import joblib
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.preprocess import buildPreprocessor, loadData, prepareFeats, splitData
from models.arqMlp import ChurnMLPClassifier

DATA_PATH = pathlib.Path("data/processed/telco_clean.csv")
MODELS_DIR = pathlib.Path("models")
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


def setMetricas(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
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

    df = loadData(DATA_PATH)
    X, y = prepareFeats(df)
    X_train, X_test, y_train, y_test = splitData(X, y)

    pipeline = Pipeline([
        ("preprocessor", buildPreprocessor()),
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
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]
        metrics = setMetricas(y_test.values, preds, probs)
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

        print(f"roc_auc={metrics['roc_auc']:.4f} | recall={metrics['recall']:.4f} | f1={metrics['f1']:.4f} | pr_auc={metrics['pr_auc']:.4f}")
        