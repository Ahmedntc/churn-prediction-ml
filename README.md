# Churn Prediction - Tech Challenge

## 📌 Objetivo

Este projeto tem como objetivo prever o churn (cancelamento) de clientes de uma operadora de telecomunicações utilizando técnicas de Machine Learning.

O projeto cobre todo o ciclo de vida de ML:

- Análise exploratória de dados (EDA)
- Modelos baseline (Scikit-Learn)
- Rede neural (MLP com PyTorch)
- Tracking de experimentos com MLflow
- API de inferência com FastAPI

---

## 📂 Estrutura do Projeto

```text
churn-prediction-ml/
│
├── dataframe/                # Dataset limpo e raw
├── notebooks/           # Análises exploratórias (EDA) e notebook comparativo entre a nossa mlp e baselines
├── modeldumps/              # Modelos treinados (.joblib / .pt) são salvos aqui
├── tests/               # Testes automatizados (pytest)
├── docs/                # Documentação
│
├── src/
│   └── churn_prediction/
│       ├── api/         # API FastAPI (endpoints, schemas, middleware)
│       ├── inference/   # Lógica de predição
│       ├── pipelines/   # Pré-processamento (sklearn)
│       └── utils/       # Utilitários (ex: logging)
│   └── data/ # Onde ocorre o preprocessamento de dados, manipulamos o dataset limpo, fazemos o split entre test e treino, preparamos as feats
│   └── models/ # Treinamento das baselines e da mlp  
├── README.md
├── pyproject.toml
└── .gitignore

🚀 Como rodar o projeto
    1. Criar ambiente virtual
        python3 -m venv .venv

    2. Ativar ambiente virtual
    Windows (PowerShell):

      .venv\Scripts\Activate.ps1

    Linux/Mac:
    source .venv/bin/activate

    3. Instalar dependências
    pip install -e ".[dev]"


🧪 Rodar testes
pytest -v

Resultado esperado:
6 passed


🤖 Treinar modelos baseline
python3 -m src.models.trainBaseline
python3 -m src.models.trainMlp

📊 Visualizar métricas no MLflow
1 > bash: mlflow ui
2 > Acesse no navegador:
http://localhost:5000

🌐 Rodar API de inferência
uvicorn churn_prediction.api.main:app --reload

🔍 Endpoints disponíveis
Health Check
GET /health

Resposta:

{
  "status": "ok"
}

Documentação interativa
http://127.0.0.1:8000/docs
Predição de churn
POST /predict

Exemplo de payload:

{
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
  "Total Charges": 1024.5
}
