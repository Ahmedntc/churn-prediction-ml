# Arquitetura de deploy — churn (API)

Documento de arquitetura: modo **batch** (treino e artefatos) versus **tempo real** (inferência) e respectiva justificativa de uso.

---

## 1. Modo escolhido: **inferência em tempo real (on-line)**

Neste projeto a entrega principal de inferência é uma **API HTTP** construída com **FastAPI**:

- Endpoint **`POST /predict`**: recebe um JSON com as features de **um** cliente (ou pode ser estendido para lote no corpo da requisição) e devolve probabilidade, classe e nível de risco.
- Endpoint **`GET /health`**: indica se o processo está vivo (útil para balanceadores e Kubernetes).

Isso caracteriza um padrão **síncrono / tempo real**: o score é calculado **no momento da requisição**.

---

## 2. Por que tempo real (e não só batch)?

| Critério | Justificativa |
|----------|----------------|
| **Uso típico de churn** | Campanhas de retenção e telas operacionais costumam precisar de score **no ato** (ex.: cliente no call center, app do gerente). |
| **Latência** | A MLP é pequena; inferência em CPU/GPU única costuma ser rápida para um registro — adequado a requisições pontuais. |
| **Acoplamento com o produto** | API desacopla **treino** (batch, offline) de **serviço** (online), alinhado ao desafio (“modelo servido via API”). |

**Batch (fora da API)** continua existindo no fluxo de **MLOps**:

- Treino e re-treino dos modelos (`trainMlp.py`, `trainBaseline.py`) são jobs **offline**.
- Geração de relatórios ou scores em massa **poderia** ser um job agendado (Airflow, cron, notebook) escrevendo em banco — **não** é o caminho principal implementado aqui, mas é compatível com o mesmo artefato `joblib`.

---

## 3. Diagrama lógico (visão simplificada)

```text
                    ┌─────────────────┐
                    │  Dados brutos   │
                    │  (CSV / lake)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Pré-processamento│
                    │ + treino (batch) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌────────────┐ ┌────────────┐ ┌────────────┐
       │  MLflow    │ │ modeldumps │ │ Notebooks │
       │ (tracking) │ │  (.joblib) │ │   / EDA   │
       └────────────┘ └──────┬─────┘ └────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  FastAPI + Uvicorn│
                    │  /health /predict │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Clientes       │
                    │ (app, CRM, etc.) │
                    └─────────────────┘
```

---

## 4. Deploy em produção (referência)

**Ambiente local (desenvolvimento):**

```bash
uvicorn churn_prediction.api.main:app --reload
```

**Produção (conceitual):**

- Container **Docker** com a mesma imagem Python + dependências do `pyproject.toml`.
- Servidor de aplicação **Uvicorn** (ou Gunicorn com workers Uvicorn) atrás de **proxy reverso** (HTTPS).
- Variável de ambiente ou volume para o caminho do **`joblib`** (padrão atual: relativo à raiz do repositório).
- **Nuvem (opcional):** AWS (ECS/Fargate, Lambda com adaptador), Azure Container Apps ou Google Cloud Run, com endpoint HTTPS público e registo da URL na documentação do projeto.

---

## 5. Decisão resumida

- **Treino e experimentação:** batch / offline, rastreados no MLflow.  
- **Inferência entregue no desafio:** **tempo real** via FastAPI, por adequação ao caso de uso de churn operacional e ao enunciado do Tech Challenge.

Se a estratégia de negócio passar a priorizar apenas pontuação em lote (por exemplo, scores diários gravados em base de dados), este documento deve ser revisto para refletir o novo desenho e a justificativa.
