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

spy:          # needs: pip install -e ".[real-data]" and internet access
	$(PY) scripts/run_experiment.py --ticker SPY
