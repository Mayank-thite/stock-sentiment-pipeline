PYTHON ?= .venv/bin/python

.PHONY: setup data prices-api run-local run-sample run-kafka kafka-up kafka-down test lint powerbi clean

setup:            ## create the virtualenv and install pinned packages
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

data:             ## download the two public datasets into data/raw
	$(PYTHON) scripts/download_data.py

prices-api:       ## optional: fetch prices from the public API and compare with the static file
	$(PYTHON) -m pipeline.ingest_prices --check

run-local: data   ## full pipeline, no Docker
	$(PYTHON) -m pipeline.run --local

run-sample:       ## full pipeline on the small committed sample, no download
	$(PYTHON) -m pipeline.run --local --sample

kafka-up:
	docker compose up -d

kafka-down:
	docker compose down -v

run-kafka: data   ## full pipeline through Kafka (run make kafka-up first)
	$(PYTHON) -m pipeline.run --kafka

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

powerbi:          ## rebuild the Power BI star schema from the last run
	$(PYTHON) -m pipeline.export_powerbi

clean:
	rm -rf data/work data/work_sample data/metastore data/spark-warehouse .pytest_cache
