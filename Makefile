.PHONY: install lint test train-baselines train-mlp run request all

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v --tb=short

train-baselines:
	python -m src.models.trainBaseline

train-mlp:
	python -m src.models.trainMlp

run:
	uvicorn src.churn_prediction.api.main:app --host 0.0.0.0 --port 8000 --reload

request:
	curl -X POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  -d '{"gender":"Female","senior_citizen":1,"partner":"No","dependents":"No","tenure":3,"phone_service":"Yes","multiple_lines":"Yes","internet_service":"Fiber optic","online_security":"No","online_backup":"No","device_protection":"No","tech_support":"No","streaming_tv":"Yes","streaming_movies":"Yes","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":95.5,"total_charges":286.5}'

all: install train-baselines train-mlp test lint 