# Churn Prediction - Tech Challenge

## 📌 Objetivo

Este projeto tem como objetivo prever o churn (cancelamento) de clientes de uma operadora de telecomunicações 
utilizando técnicas de Machine Learning.

---

## 📂 Estrutura do Projeto
# churn-prediction-ml
churn-prediction-ml/
│
├── data/ # Dataset
├── notebooks/ # Análises exploratórias
├── src/ # Código fonte
├── models/ # Modelos treinados
├── tests/ # Testes automatizados
├── docs/ # Documentação
│
├── README.md
├── pyproject.toml
└── .gitignore

 Como rodar o projeto
1. Ativar o ambiente virtual
source venv/bin/activate
2. Instalar as dependências
pip install -e ".[dev]"
3. Rodar o pré-processamento e baselines
python3 -m src.models.train_baseline
4. Visualizar os experimentos no MLflow
mlflow ui
Acesse http://localhost:5000 no browser.