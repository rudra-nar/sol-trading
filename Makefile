.PHONY: test lint typecheck all clean

all: lint typecheck test

test:
	python -m pytest tests/ -v --tb=short

lint:
	python -m ruff check src/ tests/

typecheck:
	python -m mypy src/sol_ew/core/ src/sol_ew/cli.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
