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
│       └── utils/       # Utilitários (ex: logging)
│   └── data/ # Onde ocorre o preprocessamento de dados, manipulamos o dataset limpo, fazemos o split entre test e treino, preparamos as feats
│   └── models/ # Treinamento das baselines e da mlp  
├── README.md
├── pyproject.toml
└── .gitignore
```

## Documentação (Etapa 4 — entrega final)
- **[Model Card](docs/MODEL_CARD.md)** — desempenho, limitações, vieses e cenários de falha.
- **[Arquitetura de deploy](docs/ARQUITETURA_DEPLOY.md)** — batch vs. tempo real e justificativa.
- **[Plano de monitoramento](docs/MONITORAMENTO.md)** — métricas, alertas e playbook.

## Arquitetura (resumo)

Dados em `dataframe/` → pré-processamento e treino em `src/` → artefatos em `modeldumps/` (não versionados) e experimentos no **MLflow** → inferência **em tempo real** via **FastAPI** (`churn_prediction.api.main`). Detalhes em [docs/ARQUITETURA_DEPLOY.md](docs/ARQUITETURA_DEPLOY.md).

## Setup

### Pré-requisitos

- Python 3.10+
- GPU com CUDA (opcional, mas recomendado para o treinamento da MLP)

### Instalação

**1. Clone o repositório**
```bash
git clone https://github.com/Ahmedntc/churn-prediction-ml
cd churn-prediction-ml
```

**2. Crie e ative o ambiente virtual**

Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Instale as dependências**
```bash
make install
```

## Execução

> ⚠️ Sempre ative o ambiente virtual antes de rodar qualquer comando `make`.

### Treinar os modelos baseline
```bash
make train-baselines
```
Treina DummyClassifier, Regressão Logística, Decision Tree, Random Forest e Gradient Boosting. Todos os experimentos são registrados automaticamente no MLflow.

### Treinar a MLP
```bash
make train-mlp
```
Treina a rede neural MLP com PyTorch com os melhores hiperparâmetros encontrados (batch_size=64, lr=0.01, patience=10). O experimento é registrado no MLflow.

### Visualizar experimentos no MLflow
```bash
mlflow ui
```
Acesse `http://localhost:5000` no browser para visualizar todos os experimentos, métricas e artefatos.

### Rodar os testes
```bash
make test
```

### Verificar linting
```bash
make lint
```
### Todos os passos anteriores
```bash
make all
```

### Subir a API
```bash
make run
```
A API estará disponível em `http://localhost:8000`.

### Fazer uma requisição de exemplo
Em outro terminal com a API rodando:
```bash
make request
```
## Endpoints

| Método | Caminho | Descrição |
|--------|---------|-----------|
| GET | `/health` | Verificação de saúde |
| POST | `/predict` | Predição de churn (corpo JSON validado pelo Pydantic) |

Documentação interativa (Swagger): http://127.0.0.1:8000/docs

**Exemplo — `GET /health`**

```json
{ "status": "ok" }
```

**Exemplo — corpo de `POST /predict`**

```json
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
```

## Entrega opcional (bônus)

- **Deploy em nuvem** (AWS, Azure ou GCP) com URL pública da API — documente o endpoint em [docs/ARQUITETURA_DEPLOY.md](docs/ARQUITETURA_DEPLOY.md) e cole a URL aqui: *[opcional]*
