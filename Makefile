PYTHON = uv run python
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint lint-strict test

all: install

install:
	@uv sync

run:
	@$(PYTHON) -m src

debug:
	@$(PYTHON) -m pdb -m src

clean:
	@echo "Remove temporary files or caches"
	rm -rf .venv .uv_cache .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	@echo "--- Running Flake8 ---"
	@uv run flake8 . --exclude .venv,llm_sdk
	@echo "--- Running MyPy ---"
	@uv run mypy . $(MYPY_FLAGS) --exclude "(tests/|llm_sdk/)"

lint-strict:
	@echo "--- Running Flake8 ---"
	@uv run flake8 . --exclude .venv,llm_sdk 
	@echo "--- Running MyPy ---"
	@uv run mypy . $(MYPY_FLAGS) --exclude "(tests/|llm_sdk/)"

test:
	@echo "Launching the entire suite of tests..."
	@uv run pytest -v



