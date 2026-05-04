# Churn Prediction ML

Projeto de previsão de churn para uma operadora de telecomunicações, desenvolvido como Tech Challenge da Fase 1 do programa de pós-graduação em Machine Learning Engineering da FIAP.

O objetivo é identificar clientes com risco de cancelamento do serviço, permitindo que a empresa tome ações proativas de retenção.

## Grupo

- Ahmed Bakri
- Leandro Feitosa
- Luiz Fernando

## Estrutura do Projeto

```
churn-prediction-ml/
├── src/
│   ├── data/
│   │   └── preprocess.py          # Pipeline de pré-processamento
│   ├── models/
│   │   ├── arqMlp.py              # Arquitetura da MLP (PyTorch) e wrapper sklearn
│   │   ├── trainBaseline.py       # Treinamento dos modelos baseline
│   │   └── trainMlp.py            # Treinamento da MLP
│   ├── churn_prediction/
│   │   ├── api/
│   │   │   ├── main.py            # FastAPI — endpoints /predict e /health
│   │   │   ├── schemas.py         # Schemas Pydantic
│   │   │   └── middleware.py      # Middleware de latência
│   │   ├── inference/
│   │   │   └── predictor.py       # Lógica de inferência
│   │   └── utils/
│   │       └── logging.py         # Logging estruturado
│   └── utils/
│       └── logging.py             # Logging compartilhado
├── data/
│   ├── raw/                       # Dataset original
│   └── processed/
│       └── telco_clean.csv        # Dataset limpo (gerado pelo EDA)
├── models/                        # Modelos treinados (não versionados)
├── modeldumps/                    # Pipeline final para inferência (não versionado)
├── notebooks/
│   ├── eda.ipynb                  # Análise exploratória de dados
│   └── comparativo_modelos.ipynb  # Tabela comparativa de modelos
├── tests/
│   ├── test_api.py                # Testes da API
│   ├── test_schema.py             # Testes de schema (Pydantic + Pandera)
│   └── test_smoke.py              # Smoke tests
├── docs/                          # Documentação adicional
├── conftest.py                    # Configuração do pytest
├── Makefile                       # Comandos de automação
├── pyproject.toml                 # Dependências e configuração
└── pytest.ini                     # Configuração do pytest
```

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

## API

### Endpoints

**GET /health**
Verifica se a API está no ar.
```json
{"status": "ok"}
```

**POST /predict**
Recebe os dados de um cliente e retorna a previsão de churn.

Exemplo de body:
```json
{
  "gender": "Female",
  "senior_citizen": 1,
  "partner": "No",
  "dependents": "No",
  "tenure": 3,
  "phone_service": "Yes",
  "multiple_lines": "Yes",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 95.5,
  "total_charges": 286.5
}
```

Exemplo de resposta:
```json
{
  "churn_probability": 0.86,
  "churn_prediction": 1,
  "risk_level": "high"
}
```

O campo `risk_level` pode ser `low` (< 0.4), `medium` (0.4–0.7) ou `high` (> 0.7).

## Modelos e Resultados

| Modelo | ROC-AUC | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| DummyClassifier | 0.50 | 0.00 | 0.00 | 0.27 |
| DecisionTree | 0.74 | 0.74 | 0.57 | 0.47 |
| GradientBoosting | 0.85 | 0.53 | 0.58 | 0.66 |
| LogisticRegression | 0.85 | 0.80 | 0.63 | 0.67 |
| RandomForest | 0.86 | 0.75 | 0.64 | 0.68 |
| **MLP (PyTorch)** | **0.85** | **0.84** | **0.61** | **0.67** |

A MLP foi escolhida como modelo final por apresentar o maior Recall (0.84) — métrica principal do projeto — e o maior custo de churn evitado estimado ($205k) na análise de negócio.

## Arquitetura de Deploy

O projeto adota arquitetura **real-time** via API REST. Isso permite que o time de retenção consulte o risco de churn de um cliente específico no momento do atendimento, sem precisar aguardar processamento em lote.

O pipeline completo (pré-processamento + modelo) é serializado com `joblib` e carregado pela API na inicialização, garantindo baixa latência nas predições.

## Reprodutibilidade

- Seeds fixas: `FIXED_SEED = 12` em todos os scripts
- Dataset versionado por hash MD5 registrado no MLflow
- Dependências fixadas no `pyproject.toml`
- Para reproduzir os resultados: ative o venv, rode `make install` e `make train-baselines && make train-mlp`