# credit-risk-scoring-service
#
# Every target works with a stock Python 3.9+ interpreter and no third-party
# packages. On Windows without GNU make, use .\run.ps1 instead (same targets).

PYTHON ?= python
ROWS ?= 6000
SEED ?= 20260101
DATA ?= data/applications.jsonl
MODEL ?= models/model.json
HOST ?= 127.0.0.1
PORT ?= 8080
THRESHOLD ?=

.PHONY: help setup data train run install test test-verbose smoke bench lint clean clean-data \
        check docker-build docker-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the output directories
	@mkdir -p data models reports

data: setup ## Generate the synthetic dataset (ROWS, SEED)
	$(PYTHON) scripts/generate_dataset.py --rows $(ROWS) --seed $(SEED) --output $(DATA)

train: ## Train the scorecard (needs `make data` first)
	$(PYTHON) scripts/train_model.py --data $(DATA) --model $(MODEL) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD) --threshold-mode fixed,)

run: ## Start the HTTP service
	RISKSCORE_HOST=$(HOST) RISKSCORE_PORT=$(PORT) \
	RISKSCORE_DB_PATH=$(DB_PATH) RISKSCORE_MODEL_PATH=$(MODEL) \
	PYTHONPATH=src $(PYTHON) -m riskscore.server

install: ## Install the package into the current environment (editable)
	$(PYTHON) -m pip install -e .

test: ## Run the unit and integration test suite
	$(PYTHON) -m unittest discover -s tests -t . -v

test-quiet: ## Run the test suite with a summary only
	$(PYTHON) -m unittest discover -s tests -t .

smoke: ## End-to-end check in a temporary directory
	$(PYTHON) scripts/smoke_check.py

bench: ## Time the training run and 1000 in-process scorings
	@$(PYTHON) scripts/bench.py

lint: ## Byte-compile everything to catch syntax errors
	$(PYTHON) -m compileall -q src scripts tests

check: lint smoke ## Lint plus the end-to-end smoke check
	@echo "all checks passed"

clean: ## Remove caches and generated reports
	@rm -rf reports/*.json reports/*.md
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	@find . -name '*.pyc' -delete

clean-data: ## Remove generated data and model artifacts
	@rm -rf data models reports
	@echo "removed data/, models/ and reports/"

docker-build: ## Build the container image
	docker build -t credit-risk-scoring-service:0.1.0 .

docker-run: ## Run the service in Docker on PORT
	docker run --rm -p $(PORT):8080 -v "$(PWD)/models:/app/models:ro" \
		credit-risk-scoring-service:0.1.0
