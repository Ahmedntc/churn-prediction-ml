from __future__ import annotations
import hashlib
import pathlib
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data.preprocess import (
    buildPreprocessor,
    loadData,
    prepareFeats,
    splitData,
)
# ── Configurações ─────────────────────────────────────────────────────────────
DATA_PATH = pathlib.Path("data/processed/telco_clean.csv")
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(exist_ok=True)
FIXED_SEED = 12
EXPERIMENT_NAME = "churn-baselines"
CV_FOLDS = 5


# ── Funções auxiliares ────────────────────────────────────────────────────────

def dsHash(path: pathlib.Path) -> str:
    #Gerar hash do arquivo CSV para versionamento no MLflow.
    return hashlib.md5(path.read_bytes()).hexdigest()


def computarMetricas(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Recall — não perder churners
    Precision — evitar desperdiçar ações de retenção em quem não ia cancelar
    F1 — equilíbrio entre os dois
    ROC-AUC — performance geral do modelo
    PR-AUC — performance em datasets desbalanceados como o nosso.
    """
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }

#Treina pipeline, avalia e registra tudo no MLflow.
def logTraining(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict,
    data_hash: str,
) -> None:


    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=name):

        # Validação cruzada
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=FIXED_SEED)
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=["roc_auc", "average_precision", "f1", "recall"],
        )
        for key, values in cv_results.items():
            if key.startswith("test_"):
                metric_name = key.replace("test_", "cv_")
                mlflow.log_metric(metric_name, float(values.mean()))
                mlflow.log_metric(f"{metric_name}_std", float(values.std()))

        # Treino final e avaliação no teste 
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = computarMetricas(y_test.values, y_pred, y_prob)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        mlflow.log_params(params)
        mlflow.log_param("dataset_hash", data_hash)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_state", FIXED_SEED)

        model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(pipeline, artifact_path=name)

        print(f"[{name}] test | roc_auc={metrics['roc_auc']:.4f} | pr_auc={metrics['pr_auc']:.4f} | recall={metrics['recall']:.4f} | f1={metrics['f1']:.4f}")



if __name__ == "__main__":

    np.random.seed(FIXED_SEED)

    df = loadData(DATA_PATH)
    X, y = prepareFeats(df)
    X_train, X_test, y_train, y_test = splitData(X, y)
    data_hash = dsHash(DATA_PATH)

    dummy_pipeline = Pipeline([
        ("preprocessor", buildPreprocessor()),
        ("classifier", DummyClassifier(strategy="most_frequent", random_state=FIXED_SEED)),
    ])
    logTraining(
        name="DummyClassifier",
        pipeline=dummy_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params={"strategy": "most_frequent"},
        data_hash=data_hash,
    )

    regLog_pipeline = Pipeline([
        ("preprocessor", buildPreprocessor()),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=FIXED_SEED,
                solver="lbfgs",
            ),
        ),
    ])
    logTraining(
        name="LogisticRegression",
        pipeline=regLog_pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        params={
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        data_hash=data_hash,
    )