# LLuminary Project Makefile
# Standardized commands for development, testing, and deployment

.PHONY: help install dev-install lint format type-check test test-unit test-integration test-docker clean build docs

# Default target when just running 'make'
.DEFAULT_GOAL := help

# Python and Docker commands
PYTHON := python
PIP := pip
DOCKER := docker
PYTEST := tests/ -v
DOCKER_TAG := lluminary-test

# Help command that lists all available targets with descriptions
help:
	@echo "LLuminary Project Make Commands"
	@echo "==============================="
	@echo ""
	@echo "Development:"
	@echo "  make install              Install package"
	@echo "  make dev-install          Install package with development dependencies"
	@echo "  make format               Format code with black"
	@echo "  make lint                 Lint code with ruff"
	@echo "  make type-check           Type check with mypy"
	@echo ""
	@echo "Testing:"
	@echo "  make test                 Run all tests"
	@echo "  make test-unit            Run unit tests"
	@echo "  make test-integration     Run integration tests"
	@echo "  make test-docker          Run tests in Docker"
	@echo "  make test-docker-unit     Run unit tests in Docker"
	@echo "  make test-docker-file     Run tests for a specific file (usage: make test-docker-file FILE=path/to/test_file.py)"
	@echo "  make test-docker-integration Run integration tests in Docker"
	@echo "  make docker-build-cached  Build Docker image with caching (for CI)"
	@echo "  make docker-build-matrix-cached Build matrix Docker image with caching (for CI)"
	@echo ""
	@echo "Building and Documentation:"
	@echo "  make build                Build package"
	@echo "  make docs                 Build documentation"
	@echo "  make clean                Clean all build artifacts"

# Installation targets
install:
	$(PIP) install -e .

dev-install:
	$(PIP) install -e ".[dev]"

aws-install: dev-install
	$(PIP) install -e ".[aws]"

# Code quality targets
format:
	black src/ tests/

lint:
	ruff check src/

type-check:
	mypy src/

# Testing targets
test: dev-install
	$(PYTHON) -m $(PYTEST) tests/ -v

test-unit: dev-install
	$(PYTHON) -m $(PYTEST) tests/unit/ -v

test-integration: dev-install
	$(PYTHON) -m $(PYTEST) tests/integration/ -v

# Docker testing targets
docker-build:
	$(DOCKER) build -t $(DOCKER_TAG) -f Dockerfile.test .

# For GitHub Actions with buildx and caching
docker-build-cached:
	$(DOCKER) buildx build --load -t $(DOCKER_TAG) -f Dockerfile.test \
		--cache-from=type=local,src=$(CACHE_FROM) \
		--cache-to=type=local,dest=$(CACHE_TO),mode=max .

docker-build-matrix:
	$(DOCKER) build -t $(DOCKER_TAG) -f Dockerfile.matrix .

# For GitHub Actions with buildx and caching
# This target expects Dockerfile.matrix to exist (created dynamically in CI)
docker-build-matrix-cached:
	$(DOCKER) buildx build --load -t $(DOCKER_TAG) -f Dockerfile.matrix \
		--cache-from=type=local,src=$(CACHE_FROM) \
		--cache-to=type=local,dest=$(CACHE_TO),mode=max .

# For local development matrix testing
docker-create-matrix-file:
	@echo "Creating Dockerfile.matrix for Python $(PYTHON_VERSION)"
	@echo 'FROM python:$(PYTHON_VERSION)-slim' > Dockerfile.matrix
	@echo '' >> Dockerfile.matrix
	@echo 'WORKDIR /app' >> Dockerfile.matrix
	@echo '' >> Dockerfile.matrix
	@echo '# Install development dependencies' >> Dockerfile.matrix
	@echo 'RUN apt-get update && apt-get install -y \\' >> Dockerfile.matrix
	@echo '    git \\' >> Dockerfile.matrix
	@echo '    && rm -rf /var/lib/apt/lists/*' >> Dockerfile.matrix
	@echo '' >> Dockerfile.matrix
	@echo '# Copy requirements and install dependencies' >> Dockerfile.matrix
	@echo 'COPY requirements.txt .' >> Dockerfile.matrix
	@echo 'RUN pip install -r requirements.txt' >> Dockerfile.matrix
	@echo 'RUN pip install pytest-xdist' >> Dockerfile.matrix
	@echo '' >> Dockerfile.matrix
	@echo '# Copy source code and tests' >> Dockerfile.matrix
	@echo 'COPY src/ /app/src/' >> Dockerfile.matrix
	@echo 'COPY tests/ /app/tests/' >> Dockerfile.matrix
	@echo 'COPY pyproject.toml pytest.ini mypy.ini ./' >> Dockerfile.matrix
	@echo '' >> Dockerfile.matrix
	@echo '# Install the package in development mode' >> Dockerfile.matrix
	@echo 'RUN pip install -e .' >> Dockerfile.matrix
	@echo '' >> Dockerfile.matrix
	@echo '# Set environment variables for consistent test behavior' >> Dockerfile.matrix
	@echo 'ENV PYTHONPATH=/app' >> Dockerfile.matrix
	@echo 'ENV PYTHONDONTWRITEBYTECODE=1' >> Dockerfile.matrix
	@echo 'ENV PYTHONUNBUFFERED=1' >> Dockerfile.matrix
	@echo '' >> Dockerfile.matrix
	@echo '# Run tests by default' >> Dockerfile.matrix
	@echo 'ENTRYPOINT ["python", "-m", "pytest"]' >> Dockerfile.matrix
	@echo 'CMD ["tests/"]' >> Dockerfile.matrix

test-docker: docker-build
	$(DOCKER) run --rm $(DOCKER_TAG) $(PYTEST)

test-docker-unit:
	$(DOCKER) run --rm $(DOCKER_TAG) tests/unit/ -v

test-docker-integration:
	$(DOCKER) run --rm $(DOCKER_TAG) tests/integration/ -v

test-docker-file:
	$(DOCKER) run --rm $(DOCKER_TAG) $(FILE)

# Convenience targets for local matrix testing
test-matrix-python: docker-create-matrix-file docker-build-matrix
	$(DOCKER) run --rm lluminary-test:matrix tests/ -v

# Full set of local CI simulation
test-ci-local: lint type-check docker-build test-docker-unit test-docker-integration

# Building targets
build: clean
	$(PYTHON) -m build

docs:
	cd docs && make html

# Cleaning targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Add AWS-specific test targets
test-aws: aws-install
	$(PYTHON) -m $(PYTEST) tests/unit/test_aws_utils.py -v
	$(PYTHON) -m $(PYTEST) tests/unit/test_cohere_aws.py -v

test-docker-aws: docker-build
	$(DOCKER) run --rm $(DOCKER_TAG) tests/unit/test_aws_utils.py tests/unit/test_cohere_aws.py -v

# Combined testing with quality checks
check: lint type-check test

check-docker: docker-build
	$(DOCKER) run --rm $(DOCKER_TAG) bash -c "pip install ruff && python -m ruff check src/ && python -m mypy src/ && python -m pytest tests/ -v"
