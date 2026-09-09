.PHONY: install install-dev train calibrate evaluate test lint serve docker-build docker-up

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

train:
	python scripts/train.py

calibrate:
	python scripts/calibrate.py

evaluate:
	python scripts/evaluate.py

test:
	pytest

lint:
	ruff check .

serve:
	uvicorn api.main:app --reload

docker-build:
	docker build -t m4-forecasting .

docker-up:
	docker compose up --build