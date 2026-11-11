PY ?= python

.PHONY: setup test lint spy

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

spy:          # downloads SPY 2000-2026 into data/ on first run, then uses the cached copy
	$(PY) scripts/run_spy.py
