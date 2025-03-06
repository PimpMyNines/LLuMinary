# LLMHandler Build and Release Process

This document outlines the implementation plan for proper packaging, CI/CD pipeline, and documentation for the LLMHandler project.

## Overview

Based on the current project status (70% test coverage), we need to implement a robust build and release process to make the project production-ready. This plan addresses the identified gaps in packaging, CI/CD pipeline, and documentation.

## 1. Package Structure Refinement

### 1.1 Update pyproject.toml
- Create a complete `pyproject.toml` with:
  - Proper build system configuration
  - Project metadata (name, version, description, authors)
  - Comprehensive classifiers
  - Dependencies (core and optional)
  - Development dependencies
  - Documentation dependencies
  - Tool configurations (black, isort, mypy, pytest)

### 1.2 Implement Proper Versioning
- Create a dedicated version module (`llm_handler/version.py`)
- Implement single-source versioning
- Configure version bumping automation
- Set up semantic versioning

### 1.3 Add Package Metadata
- Complete README.md with comprehensive documentation
- Add LICENSE file
- Create CONTRIBUTING.md guidelines
- Add CHANGELOG.md for tracking changes

### 1.4 Package Data Configuration
- Configure package data inclusion in pyproject.toml
- Add py.typed marker for type checking support
- Ensure non-Python files are properly included

## 2. CI/CD Pipeline Implementation

### 2.1 GitHub Actions Workflow
- Create a comprehensive workflow with multiple jobs:
  - Quality checks (linting, formatting, type checking)
  - Testing (multiple Python versions)
  - Build verification
  - Documentation generation
  - Publishing to PyPI

### 2.2 Quality Checks Integration
- Configure linting with ruff
- Set up code formatting with black and isort
- Implement type checking with mypy
- Add pre-commit hooks for local development

### 2.3 Testing Integration
- Configure pytest with coverage reporting
- Set up matrix testing for multiple Python versions
- Implement test result reporting
- Add coverage upload to Codecov

### 2.4 Release Automation
- Implement semantic versioning automation
- Set up changelog generation
- Configure release artifact creation
- Implement PyPI publishing

## 3. Documentation Enhancement

### 3.1 API Documentation
- Add comprehensive docstrings to all modules, classes, and functions
- Generate API reference documentation
- Create usage examples for all features

### 3.2 Documentation Build System
- Set up Sphinx documentation
- Configure automatic documentation deployment
- Implement ReadTheDocs integration

### 3.3 Architecture Documentation
- Create component diagrams
- Document provider interfaces
- Add sequence diagrams for key workflows

### 3.4 User Guides
- Create installation guide
- Write quickstart tutorial
- Develop advanced usage guides
- Add troubleshooting section

## Implementation Steps

### Phase 1: Package Structure Refinement

1. **Create version.py module**:
```python
"""Version information."""

__version__ = "0.1.0"
```

2. **Update pyproject.toml** with the following structure:
```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-handler"
version = {attr = "llm_handler.version.__version__"}
description = "A library for handling interactions with LLM services"
readme = "README.md"
authors = [
    {name = "PimpMyNines", email = "info@pimpmy9s.com"}
]
license = {text = "MIT"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Software Development :: Libraries",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
keywords = ["llm", "ai", "machine learning", "anthropic", "openai"]
requires-python = ">=3.8"
dependencies = [
    "anthropic>=0.5.0",
    "openai>=0.27.0",
    "pydantic>=2.0.0",
    "tenacity>=8.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.0.0",
    "mypy>=1.0.0",
    "ruff>=0.0.100",
    "pre-commit>=3.0.0",
]
docs = [
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=2.0.0",
]

[project.urls]
Homepage = "https://github.com/PimpMyNines/llm-handler"
Documentation = "https://github.com/PimpMyNines/llm-handler#readme"
Repository = "https://github.com/PimpMyNines/llm-handler.git"
Issues = "https://github.com/PimpMyNines/llm-handler/issues"

[tool.setuptools]
packages = ["llm_handler"]
include-package-data = true

[tool.setuptools.package-data]
llm_handler = ["py.typed"]

[tool.black]
line-length = 88
target-version = ["py38"]

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "--cov=llm_handler --cov-report=term-missing --cov-report=xml"

[tool.ruff]
line-length = 88
target-version = "py38"
select = ["E", "F", "B"]
ignore = []
```

3. **Create CHANGELOG.md**:
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of LLM Handler
- Support for OpenAI, Anthropic, Google, and Cohere providers
- Text generation, embeddings, streaming, reranking, and image input support
- Token counting and cost tracking
- Classification functionality
- Tool validation and function calling
```

4. **Create CONTRIBUTING.md**:
```markdown
# Contributing to LLM Handler

Thank you for considering contributing to LLM Handler! This document provides guidelines and instructions for contributing.

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
python -m pytest tests/ --cov=llm_handler --cov-report=term
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
```

### Phase 2: CI/CD Pipeline Implementation

1. **Create GitHub Actions workflow**:
```yaml
name: Python Package

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install ".[dev]"
    - name: Lint with ruff
      run: |
        ruff llm_handler tests
    - name: Check formatting with black
      run: |
        black --check llm_handler tests
    - name: Check imports with isort
      run: |
        isort --check-only --profile black llm_handler tests
    - name: Type check with mypy
      run: |
        mypy llm_handler

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install ".[dev]"
    - name: Test with pytest
      run: |
        pytest
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false

  build:
    needs: [quality, test]
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build package
      run: |
        python -m build
    - name: Check package
      run: |
        twine check dist/*
    - name: Store build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/

  publish:
    needs: [build]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
    - uses: actions/checkout@v3
    - name: Download build artifacts
      uses: actions/download-artifact@v3
      with:
        name: dist
        path: dist/
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install twine
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: ${{ secrets.PYPI_USERNAME }}
        TWINE_PASSWORD: ${{ secrets.PYPI_PASSWORD }}
      run: |
        twine upload dist/*
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
```

2. **Create pre-commit configuration**:
```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files

-   repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
    -   id: black

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
        args: ["--profile", "black"]

-   repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.262
    hooks:
    -   id: ruff
        args: ["--fix"]

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
    -   id: mypy
        additional_dependencies: [types-requests]
```

### Phase 3: Documentation Enhancement

1. **Set up Sphinx documentation**:
```
mkdir -p docs/source
```

2. **Create Sphinx configuration**:
```python
# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Import version
from llm_handler.version import __version__

# Project information
project = 'LLM Handler'
copyright = '2025, PimpMyNines'
author = 'PimpMyNines'
release = __version__

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
```

3. **Create index.rst**:
```rst
LLM Handler Documentation
=========================

A versatile Python library for interacting with multiple LLM providers through a unified interface.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   providers
   advanced
   contributing
   changelog

Indices and tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

## Timeline and Prioritization

### Week 1: Package Structure Refinement
- Day 1-2: Set up version.py and update pyproject.toml
- Day 3-4: Create CHANGELOG.md, CONTRIBUTING.md, and LICENSE
- Day 5: Verify package structure and test installation

### Week 2: CI/CD Pipeline Implementation
- Day 1-2: Set up GitHub Actions workflow
- Day 3: Configure pre-commit hooks
- Day 4-5: Test CI/CD pipeline and fix any issues

### Week 3: Documentation Enhancement
- Day 1-2: Set up Sphinx documentation
- Day 3-4: Create basic documentation structure
- Day 5: Test documentation build and deployment

### Week 4: Finalization and Testing
- Day 1-2: Complete any remaining tasks
- Day 3-4: Comprehensive testing of all components
- Day 5: Final review and release preparation

## Success Metrics

- Package successfully builds and installs
- CI/CD pipeline runs successfully on all PRs
- Documentation builds correctly and is accessible
- Test coverage remains at or above current levels
- Package can be published to PyPI

## Release Process

1. **Prepare Release**
   - Update version in `llm_handler/version.py`
   - Update CHANGELOG.md with release notes
   - Create a PR for the release

2. **Review and Approve**
   - Ensure all tests pass
   - Review documentation
   - Verify package builds correctly

3. **Tag and Release**
   - Merge the release PR
   - Create a tag with the version number (e.g., `v0.1.0`)
   - Push the tag to trigger the release workflow

4. **Monitor Release**
   - Verify GitHub Actions workflow completes successfully
   - Confirm package is published to PyPI
   - Check documentation is updated

5. **Post-Release**
   - Announce the release
   - Update version to next development version
   - Create issues for planned features in the next release
