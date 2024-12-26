# -------------------------------------------------------------
# Common Usage:
#   make install            # Install main an dev requirements
#   make lint               # Run flake8
#   make format             # Run black
#   make test               # Run tests with pytest
#   make coverage           # Run test coverage
# -------------------------------------------------------------

.PHONY: help python_check install lint format test coverage

help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  install         - install main and dev requirements"
	@echo "  lint            - run flake8 linting"
	@echo "  format          - run black formatting"
	@echo "  test            - run tests with pytest"
	@echo "  coverage        - run pytest coverage"

python_check:
	@which python3 > /dev/null || (echo "Error: Python 3 is not installed. Please install Python 3 to continue. Instructions: \
	For macOS: Install Homebrew (https://brew.sh) and run 'brew install python'. \
	For Linux: Use your package manager (e.g., 'sudo apt install python3' on Ubuntu/Debian).\
	For Windows: Download Python from https://www.python.org/downloads/ and ensure it is added to your PATH." && exit 1)
	@echo "Using $$(python3 --version)"
install:
	pip install --upgrade pip
	@echo "Installing requirements..."
	pip install -r requirements.txt -r requirements_dev.txt

lint:
	flake8 src tests

format:
	black src tests

test:
	python -m pytest -v

coverage:
	pytest --cov=src --cov-report=term-missing tests/