# GitHub Project Management Tool Tests

This directory contains the test suite for the GitHub Project Management Tool.

## Structure

- `unit/`: Unit tests for individual components
- `integration/`: Integration tests involving multiple components
- `functional/`: Functional and end-to-end tests
- `conftest.py`: Shared test fixtures and utilities

## Running Tests

From the project root directory, run:

```bash
# Run all tests
python -m pytest

# Run specific test files
python -m pytest tests/unit/test_issue_manager.py

# Run tests with coverage
python -m pytest --cov=src
```

## Test Fixtures

Test fixtures for mocking GitHub API responses are available in the `fixtures` directory.

## Adding Tests

When adding new features, please ensure that appropriate tests are added to maintain test coverage.
