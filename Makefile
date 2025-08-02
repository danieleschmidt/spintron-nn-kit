# SpinTron-NN-Kit Makefile
# Ultra-low-power neural inference framework for spin-orbit-torque hardware

.PHONY: help install install-dev test lint format clean docs build benchmark simulate
.DEFAULT_GOAL := help

PYTHON := python
PIP := pip
VENV := venv
ACTIVATE := $(VENV)/bin/activate

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

help: ## Display this help message
	@echo "$(BLUE)SpinTron-NN-Kit Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# Environment Setup
venv: ## Create virtual environment
	@echo "$(YELLOW)Creating virtual environment...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Virtual environment created. Activate with: source $(ACTIVATE)$(RESET)"

install: ## Install package
	@echo "$(YELLOW)Installing spintron-nn-kit...$(RESET)"
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	@echo "$(YELLOW)Installing spintron-nn-kit with dev dependencies...$(RESET)"
	$(PIP) install -e ".[dev]"
	pre-commit install

install-all: ## Install package with all optional dependencies
	@echo "$(YELLOW)Installing spintron-nn-kit with all dependencies...$(RESET)"
	$(PIP) install -e ".[all]"

# Code Quality
format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(RESET)"
	black spintron_nn/ tests/ scripts/
	isort spintron_nn/ tests/ scripts/

format-check: ## Check code formatting
	@echo "$(YELLOW)Checking code formatting...$(RESET)"
	black --check spintron_nn/ tests/ scripts/
	isort --check-only spintron_nn/ tests/ scripts/

lint: ## Run linting
	@echo "$(YELLOW)Running linters...$(RESET)"
	flake8 spintron_nn/ tests/ scripts/
	pylint spintron_nn/

typecheck: ## Run type checking
	@echo "$(YELLOW)Running type checking...$(RESET)"
	mypy spintron_nn/

security-check: ## Run security checks
	@echo "$(YELLOW)Running security checks...$(RESET)"
	bandit -r spintron_nn/ -f json -o security_report.json

pre-commit: ## Run pre-commit hooks
	@echo "$(YELLOW)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

# Testing
test: ## Run all tests
	@echo "$(YELLOW)Running tests...$(RESET)"
	pytest tests/ -v --cov=spintron_nn --cov-report=html --cov-report=term-missing

test-unit: ## Run unit tests only
	@echo "$(YELLOW)Running unit tests...$(RESET)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(RESET)"
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests only
	@echo "$(YELLOW)Running end-to-end tests...$(RESET)"
	pytest tests/e2e/ -v

test-fast: ## Run fast tests (exclude slow markers)
	@echo "$(YELLOW)Running fast tests...$(RESET)"
	pytest tests/ -v -m "not slow"

test-coverage: ## Generate test coverage report
	@echo "$(YELLOW)Generating coverage report...$(RESET)"
	pytest tests/ --cov=spintron_nn --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(RESET)"

# Hardware and Simulation
simulate: ## Run hardware simulation
	@echo "$(YELLOW)Running hardware simulation...$(RESET)"
	$(PYTHON) scripts/run_simulation.py

simulate-spice: ## Run SPICE simulation
	@echo "$(YELLOW)Running SPICE simulation...$(RESET)"
	$(PYTHON) scripts/run_spice_simulation.py

generate-hardware: ## Generate Verilog hardware
	@echo "$(YELLOW)Generating hardware...$(RESET)"
	$(PYTHON) scripts/generate_hardware.py

analyze-power: ## Run power analysis
	@echo "$(YELLOW)Analyzing power consumption...$(RESET)"
	$(PYTHON) scripts/analyze_power.py

# Benchmarking
benchmark: ## Run benchmarks
	@echo "$(YELLOW)Running benchmarks...$(RESET)"
	$(PYTHON) scripts/run_benchmarks.py

benchmark-energy: ## Run energy efficiency benchmarks
	@echo "$(YELLOW)Running energy benchmarks...$(RESET)"
	$(PYTHON) scripts/benchmark_energy.py

benchmark-accuracy: ## Run accuracy benchmarks
	@echo "$(YELLOW)Running accuracy benchmarks...$(RESET)"
	$(PYTHON) scripts/benchmark_accuracy.py

# Documentation
docs: ## Build documentation
	@echo "$(YELLOW)Building documentation...$(RESET)"
	cd docs && make html
	@echo "$(GREEN)Documentation built in docs/_build/html/$(RESET)"

docs-serve: ## Serve documentation locally
	@echo "$(YELLOW)Serving documentation...$(RESET)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(YELLOW)Cleaning documentation...$(RESET)"
	cd docs && make clean

# Build and Distribution
build: ## Build package
	@echo "$(YELLOW)Building package...$(RESET)"
	$(PYTHON) -m build

build-wheel: ## Build wheel distribution
	@echo "$(YELLOW)Building wheel...$(RESET)"
	$(PYTHON) -m build --wheel

build-sdist: ## Build source distribution
	@echo "$(YELLOW)Building source distribution...$(RESET)"
	$(PYTHON) -m build --sdist

# Docker
docker-build: ## Build Docker image
	@echo "$(YELLOW)Building Docker image...$(RESET)"
	docker build -t spintron-nn-kit:latest .

docker-run: ## Run Docker container
	@echo "$(YELLOW)Running Docker container...$(RESET)"
	docker run -it --rm -v $(PWD):/workspace spintron-nn-kit:latest

docker-test: ## Run tests in Docker
	@echo "$(YELLOW)Running tests in Docker...$(RESET)"
	docker run --rm -v $(PWD):/workspace spintron-nn-kit:latest make test

# Utilities
clean: ## Clean build artifacts and cache files
	@echo "$(YELLOW)Cleaning up...$(RESET)"
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*.pyd' -delete
	find . -name '.coverage' -delete
	find . -name '*.orig' -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/
	rm -rf docs/_build/
	rm -rf simulation/
	rm -rf work/
	rm -rf *.vcd
	rm -rf *.wlf
	@echo "$(GREEN)Cleanup complete$(RESET)"

clean-all: clean ## Clean everything including venv
	@echo "$(YELLOW)Cleaning everything including virtual environment...$(RESET)"
	rm -rf $(VENV)/
	@echo "$(GREEN)Deep cleanup complete$(RESET)"

# Development workflow
dev-setup: venv install-dev ## Complete development environment setup
	@echo "$(GREEN)Development environment ready!$(RESET)"
	@echo "$(BLUE)Activate with: source $(ACTIVATE)$(RESET)"

check: format-check lint typecheck test-fast ## Run all code quality checks

ci: format-check lint typecheck security-check test ## Run CI pipeline

release-check: check docs build ## Check if ready for release
	@echo "$(GREEN)Release checks passed!$(RESET)"

# Hardware tools integration
cadence-setup: ## Setup Cadence tools environment
	@echo "$(YELLOW)Setting up Cadence environment...$(RESET)"
	scripts/setup_cadence.sh

synopsys-setup: ## Setup Synopsys tools environment
	@echo "$(YELLOW)Setting up Synopsys environment...$(RESET)"
	scripts/setup_synopsys.sh

xilinx-setup: ## Setup Xilinx tools environment
	@echo "$(YELLOW)Setting up Xilinx environment...$(RESET)"
	scripts/setup_xilinx.sh

# Database and metrics
init-db: ## Initialize metrics database
	@echo "$(YELLOW)Initializing metrics database...$(RESET)"
	$(PYTHON) scripts/init_database.py

metrics-collect: ## Collect performance metrics
	@echo "$(YELLOW)Collecting metrics...$(RESET)"
	$(PYTHON) scripts/collect_metrics.py

metrics-report: ## Generate metrics report
	@echo "$(YELLOW)Generating metrics report...$(RESET)"
	$(PYTHON) scripts/generate_metrics_report.py

# Example workflows
example-keyword-spotting: ## Run keyword spotting example
	@echo "$(YELLOW)Running keyword spotting example...$(RESET)"
	$(PYTHON) examples/keyword_spotting/main.py

example-vision: ## Run vision model example
	@echo "$(YELLOW)Running vision example...$(RESET)"
	$(PYTHON) examples/vision/main.py

example-benchmark: ## Run benchmark example
	@echo "$(YELLOW)Running benchmark example...$(RESET)"
	$(PYTHON) examples/benchmarking/main.py

# Quick start
quick-start: dev-setup ## Quick development setup and run tests
	@echo "$(BLUE)Running quick start...$(RESET)"
	make test-fast
	@echo "$(GREEN)Quick start complete! You're ready to develop.$(RESET)"