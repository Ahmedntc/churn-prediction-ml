from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FIXED_SEED = 12
#colunas numericas
NUMERIC_FEATURES: list[str] = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]
#coplunas categoricas
CATEGORICAL_FEATURES: list[str] = [
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

def loadData(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepareFeats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    #x sao todas  as features menos o churn value ou seja é o que o modelo vai usar para aprender, y é a coluna target
    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"].astype(int)
    return X, y


def buildPreprocessor() -> ColumnTransformer:

    #Colocando as variaveis numericas em uma mesma escalar para evoitar que uma variavel tenha mais peso que a outra
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    # tranformando as variaveis categoricas em variaveis binarias 
    categorical_pipeline = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def splitData(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        #seed fixa
        random_state=FIXED_SEED,
        stratify=y,
    )