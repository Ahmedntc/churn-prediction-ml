# Plano de monitoramento — API e modelo de churn

Plano operacional: **métricas** a acompanhar, **alertas** sugeridos e **playbook** de resposta a incidentes.

As integrações com APM, Prometheus, Grafana ou serviços equivalentes são **recomendações para ambiente de produção**; o repositório concentra-se no serviço FastAPI e no rastreio de experimentos no MLflow.

---

## 1. Métricas a acompanhar

### 1.1 Serviço (API)

| Métrica | O que mede | Onde / como |
|---------|------------|-------------|
| **Disponibilidade** | % de tempo com `/health` OK | Synthetic checks, load balancer |
| **Latência p50 / p95 / p99** | Tempo de resposta de `/predict` | APM, logs estruturados (middleware de latência já existe no projeto) |
| **Taxa de erro HTTP** | 4xx vs 5xx | Gateway / reverse proxy |
| **Throughput** | Requisições por segundo | Métricas do processo Uvicorn / container |

### 1.2 Dados e modelo (qualidade / drift)

| Métrica | O que mede | Notas |
|---------|------------|--------|
| **Volume de requisições** | Queda ou pico anormal | Pode indicar incidente ou bot |
| **Taxa de 422 (validação)** | Payloads fora do schema | Muitos 422 podem indicar cliente da API desatualizado ou ataque |
| **Distribuição de features** | PSI, KS ou histogramas vs. treino | Requer logging amostral ou batch de comparação com baseline de treino |
| **Performance proxy** | Feedback de label (churn de fato em 30 dias) | Em produção real, comparar predição com desfecho; calcular PR-AUC/recall em janela móvel |

### 1.3 MLflow (offline)

- Continuar registrando **runs** de retreino com **mesmas métricas** (`test_*`) para comparar versões do modelo ao longo do tempo.

---

## 2. Alertas sugeridos

| Alerta | Condição (exemplo) | Severidade |
|--------|---------------------|------------|
| **API fora do ar** | `/health` falha N vezes seguidas | Crítica |
| **Latência alta** | p95 de `/predict` > limiar acordado (ex.: 2 s) por 5 min | Alta |
| **Erro 5xx** | Taxa > 1% em 5 min | Crítica |
| **Drift de dados** | PSI > 0,25 em feature chave (ex.: `Monthly Charges`) | Média — revisar retreino |
| **Queda de qualidade** | Recall ou PR-AUC em dados rotulados recentes abaixo do mínimo acordado | Alta |

Os limiares numéricos (latência, taxa de erro, PSI) devem ser calibrados com o negócio e com testes de carga antes de produção.

---

## 3. Playbook de resposta (o que fazer quando…)

### 3.1 Health falhando / API indisponível

1. Verificar se o processo Uvicorn/container está em execução.  
2. Ver logs da aplicação (erro de import, porta, crash no startup).  
3. Se o erro for ao **carregar o modelo** na primeira predição: confirmar existência do `joblib` no caminho esperado (`modeldumps/...`).  
4. Rollback para versão anterior da imagem/deploy, se aplicável.

### 3.2 Latência alta sem erro

1. Checar CPU/memória do host e número de workers.  
2. Verificar se há concorrência excessiva ou modelo carregado várias vezes por worker.  
3. Escalar horizontalmente (mais réplicas) ou otimizar batching se houver endpoint em lote.

### 3.3 Muitos 422 (validação)

1. Comparar contrato da API (`PredictRequest`) com o cliente que integra.  
2. Atualizar documentação OpenAPI e versão da API (`version` no FastAPI).  
3. Se for tráfego malicioso, rate limiting no gateway.

### 3.4 Suspeita de drift ou queda de métricas de negócio

1. Congelar **promoção** de novos modelos até análise.  
2. Rodar notebook de EDA em janela recente; comparar distribuições.  
3. Agendar **retreino** com dados atualizados e validar no holdout + MLflow.  
4. Atualizar **Model Card** com nova versão e limitações observadas.

### 3.5 Incidente de segurança (vazamento / abuso)

1. Revogar chaves de API e restringir rede (VPC, IP allowlist).  
2. Auditar logs de acesso.  
3. Seguir política institucional de incidentes.

---

## 4. Papéis e responsabilidades

| Papel | Responsabilidade |
|-------|-------------------|
| **Dono do modelo** | Aprovar retreinos, limiares de decisão e atualização do Model Card |
| **Dono da API** | Deploy, SLOs de latência e disponibilidade |
| **Dados** | Qualidade da fonte, tratamento de PII e pipelines de ingestão |

---

## 5. Evolução

Em produção, recomenda-se ligar este plano a **dashboards** (Grafana, Datadog, Amazon CloudWatch ou equivalente) e a rotinas de **plantão (on-call)**, conforme a maturidade operacional da equipe.
