from __future__ import annotations

import hashlib
import pathlib

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.tree import DecisionTreeClassifier

from src.data.preprocess import (
    build_preprocessor,
    load_data,
    prepare_feats,
    split_data,
)
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

DATA_PATH = pathlib.Path("dataframe/processed/telco_clean.csv")
MODELS_DIR = pathlib.Path("modeldumps")
MODELS_DIR.mkdir(exist_ok=True)
FIXED_SEED = 12
EXPERIMENT_NAME = "churn-baselines"
CV_FOLDS = 5
def ds_hash(path: pathlib.Path) -> str:
    #Gerar hash do arquivo CSV para versionamento no MLflow.
    return hashlib.md5(path.read_bytes()).hexdigest()


def computar_metricas(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Recall  não perder churners que é o mais importante para o negócio mesmo que isso signifique ter mais falsos positivos
    Precision  evitar desperdiçar ações de retenção em quem não ia cancelar
    F1  equilíbrio entre os dois
    ROC-AUC  performance geral do modelo
    PR-AUC  performance em datasets desbalanceados como o nosso.
    """
    return {
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }

#Treina pipeline, avalia e registra tudo no MLflow.
def log_training(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
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
            x_train,
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
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        y_prob = pipeline.predict_proba(x_test)[:, 1]
        metrics = computar_metricas(y_test.values, y_pred, y_prob)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        mlflow.log_params(params)
        mlflow.log_param("dataset_hash", data_hash)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_state", FIXED_SEED)

        model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(pipeline, artifact_path=name)

        logger.info("[%s] test | roc_auc=%.4f | pr_auc=%.4f | recall=%.4f | f1=%.4f", name, metrics['roc_auc'], metrics['pr_auc'], metrics['recall'], metrics['f1'])



if __name__ == "__main__":
    #baselines lineares e arvores

    np.random.seed(FIXED_SEED)
    df = load_data(DATA_PATH)
    x, y = prepare_feats(df)
    x_train, X_test, y_train, y_test = split_data(x, y)
    data_hash = ds_hash(DATA_PATH)

    dummy_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DummyClassifier(strategy="most_frequent", random_state=FIXED_SEED)),
    ])
    log_training(
        name="DummyClassifier",
        pipeline=dummy_pipeline,
        x_train=x_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        params={"strategy": "most_frequent"},
        data_hash=data_hash,
    )

    reg_log_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
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
    log_training(
        name="LogisticRegression",
        pipeline=reg_log_pipeline,
        x_train=x_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        params={
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
        },
        data_hash=data_hash,
    )

    arv_decisao_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=10,
            random_state=FIXED_SEED,
        )),
    ])
    log_training(
        name="DecisionTree",
        pipeline=arv_decisao_pipeline,
        x_train=x_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        params={"max_depth": 10, "class_weight": "balanced"},
        data_hash=data_hash,
    )

    rand_forest_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=10,
            random_state=FIXED_SEED,
        )),
    ])
    log_training(
        name="RandomForest",
        pipeline=rand_forest_pipeline,
        x_train=x_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        params={"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"},
        data_hash=data_hash,
    )

    grad_boost_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=FIXED_SEED,
        )),
    ])
    log_training(
        name="GradientBoosting",
        pipeline=grad_boost_pipeline,
        x_train=x_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test,
        params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
        data_hash=data_hash,
    )
