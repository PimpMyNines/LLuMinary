# Contributing to LLuMinary

Thank you for considering contributing to LLuMinary! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Development Setup

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

Run tests using pytest:
```bash
python -m pytest tests/
```

Run tests with coverage:
```bash
python -m pytest tests/ --cov=lluminary --cov-report=term
```

## Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- ruff for linting

You can run these tools manually or let pre-commit handle them automatically.

## Pull Request Process

1. Update the README.md and documentation with details of changes if appropriate
2. Update the CHANGELOG.md with details of changes
3. The PR should work for Python 3.8, 3.9, and 3.10
4. Ensure all tests pass and coverage doesn't decrease
5. The PR will be merged once it receives approval from maintainers
