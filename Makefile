PY ?= python

.PHONY: setup test lint experiment scarce spy

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

experiment:   # ~2.5 minutes
	$(PY) scripts/run_experiment.py

scarce:       # same, with only 250 training days
	$(PY) scripts/run_experiment.py --train-days 250

spy:          # uses data/SPY.csv if present, else downloads (pip install -e ".[real-data]")
	$(PY) scripts/run_experiment.py --ticker SPY $(if $(wildcard data/SPY.csv),--csv data/SPY.csv,)
