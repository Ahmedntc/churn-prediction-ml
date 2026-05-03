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

- **[Passo a passo do que falta](docs/PASSO_A_PASSO_ETAPA_4.md)** — checklist em português (Brasil) para fechar a entrega.
- **[Model Card](docs/MODEL_CARD.md)** — desempenho, limitações, vieses e cenários de falha.
- **[Arquitetura de deploy](docs/ARQUITETURA_DEPLOY.md)** — batch vs. tempo real e justificativa.
- **[Plano de monitoramento](docs/MONITORAMENTO.md)** — métricas, alertas e playbook.

## Arquitetura (resumo)

Dados em `dataframe/` → pré-processamento e treino em `src/` → artefatos em `modeldumps/` (não versionados) e experimentos no **MLflow** → inferência **em tempo real** via **FastAPI** (`churn_prediction.api.main`). Detalhes em [docs/ARQUITETURA_DEPLOY.md](docs/ARQUITETURA_DEPLOY.md).

## Como rodar o projeto

Execute os comandos na **raiz** do repositório (`churn-prediction-ml/`, onde está o `pyproject.toml`).

1. **Criar ambiente virtual**

   ```bash
   python -m venv .venv
   ```

2. **Ativar o ambiente**

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Linux/macOS: `source .venv/bin/activate`

3. **Instalar dependências**

   ```bash
   pip install -e ".[dev]"
   ```

   Se a instalação falhar ao gerar bytecode no Windows, use: `pip install -e ".[dev]" --no-compile`.

4. **Gerar o modelo servido pela API** (arquivos `.joblib` ficam em `modeldumps/` e não vão para o Git):

   ```bash
   python -m src.models.trainBaseline
   python -m src.models.trainMlp
   ```

## Testes

```bash
pytest -v
```

Resultado esperado: **6 passed** (na raiz do projeto).

## MLflow

```bash
mlflow ui
```

Abra no navegador: http://localhost:5000

## API de inferência

Na raiz do projeto, com o ambiente ativado:

```bash
uvicorn churn_prediction.api.main:app --reload
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

## Entrega — vídeo (obrigatório)

1. Grave o vídeo de **até 5 minutos** usando o método **STAR** (Situação, Tarefa, Ação, Resultado), conforme o enunciado do Tech Challenge.
2. Publique em plataforma de sua escolha (YouTube não listado, Google Drive, Loom, etc.).
3. **Cole o link aqui no README** (substitua o texto abaixo) e/ou coloque na descrição do repositório no GitHub:

**Link do vídeo:** *[inserir URL]*

## Entrega opcional (bônus)

- **Deploy em nuvem** (AWS, Azure ou GCP) com URL pública da API — documente o endpoint em [docs/ARQUITETURA_DEPLOY.md](docs/ARQUITETURA_DEPLOY.md) e cole a URL aqui: *[opcional]*
