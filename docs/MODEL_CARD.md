# Model Card — Previsão de churn (MLP + pipeline)

**Versão do documento:** 0.2.0  
**Projeto:** churn-ml (Tech Challenge — Fase 1)  
**Idioma:** Português (Brasil)  
**Data de referência das métricas:** 03/05/2026 (experimento MLflow `churn-mlp`, conjunto de teste).

---

## 1. Visão geral do modelo

| Campo | Descrição |
|--------|-----------|
| **Tipo de modelo** | Pipeline Scikit-Learn: pré-processamento tabular + classificador **MLP** em **PyTorch** (`ChurnMLPClassifier`), com early stopping e batching. |
| **Tarefa** | Classificação binária: **churn** vs. não churn. |
| **Entrada** | Um registro (JSON) com features alinhadas ao Telco Customer Churn, mesmo esquema do treino. |
| **Saída** | Probabilidade de churn, classe predita e faixa de risco (`low` / `medium` / `high`) na API. |
| **Artefato na API** | `modeldumps/pipeline_mlp_lr0_01_bs64_patience10.joblib` (gerado por `python -m src.models.trainMlp`; **não versionado no Git**). |

**Baselines:** `python -m src.models.trainBaseline` — experimento MLflow `churn-baselines`.

---

## 2. Dados de treino

| Campo | Valor |
|--------|--------|
| **Fonte** | Telco Customer Churn (telecomunicações, dados tabulares). |
| **Arquivo** | `dataframe/processed/telco_clean.csv` |
| **Registros (N)** | 7.032 |
| **Taxa aproximada de churn (alvo)** | ~26,6% (classe positiva minoritária — usar PR-AUC e recall no contexto de negócio). |
| **Split** | Treino/teste estratificado (`src/data/preprocess.py` — `splitData`). |
| **Versionamento** | Hash MD5 do CSV nas runs de baseline no MLflow. |

---

## 3. Desempenho (métricas no conjunto de teste — MLP)

Valores da **última run** consultada no experimento **`churn-mlp`** (métricas `test_*` logadas após `fit` no holdout de teste).

| Métrica | Valor | Nota |
|---------|--------|------|
| **ROC-AUC** | **0,8496** | Discriminação geral entre classes. |
| **PR-AUC** | **0,6686** | Adequada com classe churn minoritária. |
| **Recall** | **0,8449** | Proporção de churners reais identificados (alto recall costuma ser prioridade em retenção). |
| **F1** | **0,6118** | Equilíbrio precision/recall. |

**MLflow — run ID (referência):** `72a5d812b0ca4831892cb2fe4b5e10d7`.

**Baselines:** experimento MLflow `churn-baselines` — DummyClassifier, Regressão Logística, Decision Tree, Random Forest, Gradient Boosting; validação cruzada estratificada e métricas de teste conforme `src/models/trainBaseline.py`.

---

## 4. Limitações

1. **Generalização:** modelo treinado neste recorte do Telco; outra base ou época exige validação e provavelmente retreino.  
2. **Features fora do treino:** categorias novas ou valores extremos podem degradar predição ou falhar no pré-processador.  
3. **Label fixo:** “churn” segue a definição do dataset; mudança de produto/política altera o significado do alvo.  
4. **Threshold 0,5:** a API usa a decisão padrão do classificador; campanhas de negócio podem exigir outro corte na probabilidade (não exposto como parâmetro na API atual).

---

## 5. Vieses e equidade

1. **Desbalanceamento:** churn ~27%; métricas globais podem mascarar performance em subgrupos.  
2. **Atributos sensíveis:** o dataset inclui, por exemplo, **gênero** e **idade (Senior Citizen)** — uso em decisões automatizadas de cliente exige governança legal/ética; aqui o foco é acadêmico.  
3. **Subamostragem:** grupos com poucas observações podem ter estimativas instáveis; recomenda-se análise segmentada no notebook de EDA quando houver volume suficiente.

---

## 6. Cenários de falha

| Cenário | O que acontece | Mitigação |
|---------|----------------|-----------|
| **Arquivo `.joblib` ausente** | Erro ao chamar `/predict` (modelo carregado sob demanda). | Rodar `python -m src.models.trainMlp` na raiz do projeto. |
| **JSON inválido** | HTTP **422** (validação Pydantic). | Ver exemplos em `README.md` e `/docs` da API. |
| **Drift** | Queda de PR-AUC/recall em dados novos. | Plano em [`MONITORAMENTO.md`](MONITORAMENTO.md); retreino. |
| **Ambiente diferente** | Pequenas variações numéricas. | `pyproject.toml`, seeds (`FIXED_SEED = 12` no treino), mesmo Python quando possível. |

---

## 7. Reprodutibilidade

- **Seed:** `12` em `src/models/trainMlp.py` (`FIXED_SEED`).  
- **Hiperparâmetros MLP:** `lr=0,01`, `batch_size=64`, `patience=10`, `max_epochs=100`, `hidden_dims=[64,32,16]`, `dropout_rate=0,3` — todos logados no MLflow.  
- **Comando:** na pasta raiz do projeto `churn-prediction-ml/`, com venv ativo: `python -m src.models.trainMlp`.

---

## 8. Equipe e contato

| Campo | Valor |
|--------|--------|
| **Responsáveis** | Ahmed Mohamad Bakri; Luiz Fernando Carvalho; Leandro Feitosa da Silva |
| **Contato** | leandros.feitosa@hotmail.com · Discord: leandro073918 |
