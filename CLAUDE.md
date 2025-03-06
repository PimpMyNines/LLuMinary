# LLMHandler Development Guide

## Commands
- Run all tests: `python -m pytest tests/ -v`
- Run unit tests: `python -m pytest tests/unit/ -v`
- Run integration tests: `python -m pytest tests/integration/ -v`
- Run specific test: `python -m pytest tests/path/to/test_file.py::TestClass::test_function -v`
- Run tests by pattern: `python -m pytest tests/ -k "openai" -v`
- Run tests by marker: `python -m pytest tests/ -m "image" -v`
- Run tests with coverage: `python -m pytest tests/ --cov=src/llmhandler --cov-report=term`
- Install development dependencies: `pip install -e ".[dev]"`
- Install AWS dependencies: `pip install -e ".[aws]"`
- Build package: `python -m build`
- Check package: `twine check dist/*`
- Build docs: `sphinx-build -b html docs/source docs/build/html`

## Code Style Guidelines
- Python version: 3.10+
- Type hints: Required for all function parameters and return values, use Optional/Union
- Naming: CamelCase for classes, snake_case for functions/variables, UPPER_SNAKE_CASE for constants
- Imports: Group in order: standard libs, third-party, local; alphabetical within groups
- Documentation: Google-style docstrings for all public classes and methods
- Error handling: Use custom exception classes from exceptions.py
- Design: Follow single responsibility principle and provider registry pattern
- Tests: Unit tests required for all new functionality (target 90%+ coverage)

## Package Structure Guidelines
- Use pyproject.toml for all package configuration (PEP 621)
- Organize code into logical modules with clear separation of concerns
- Package all non-Python files using package_data in pyproject.toml
- Follow single-source versioning pattern via version.py
- Include proper typing support with py.typed marker
- Document all public APIs with appropriate docstrings

## Provider Integration
- Maintain consistent interfaces across providers (OpenAI, Anthropic, Google, Cohere, Bedrock)
- Support: text generation, embeddings, streaming, reranking, image input, function calling
- Always implement token counting and cost tracking for each provider
- Follow provider_template.py pattern when adding new providers
- Implement proper error handling with standardized exception types

## Current Status
- 203 unit tests now passing:
  - 13 for LLMHandler
  - 20 for Base LLM class
  - 170 for other components including providers, tools, and CLI
- 74 integration tests now implemented and enabled:
  - 7 for embeddings functionality
  - 9 for streaming across providers
  - 8 for document reranking
  - 5 for cost tracking and comparison
  - 5 for advanced features (response processing, function calling)
  - 3 for CLI functionality
  - 3 for tool registry integration
  - 8 for cross-provider testing
  - 6 for classification integration
  - 4 for image generation
  - 3 for rate limiting behavior
  - 4 for provider error types
  - 4 for dynamic model selection
  - 5 for optional parameters
- Current test coverage improved from 35% to 70% (goal is 90%)
- Recently implemented:
  - Better error handling with LLMHandlerError hierarchy
  - Comprehensive testing for Google Provider (80%+ coverage)
  - Comprehensive testing for Bedrock Provider (75%+ coverage)
  - Added rate limiting behavior tests (recovery, backoff)
  - Implemented provider error type and mapping tests
  - Added dynamic model selection tests (fallback, routing)
  - Implemented optional parameter tests
  - Fixed tests for router (93% coverage)
  - Added comprehensive tool registry tests (66% coverage)
  - Verified comprehensive Cohere provider tests (90%+ coverage)
  - Enhanced documentation with test markers and organization

## Testing Organization
- Integration tests are organized by feature and use appropriate markers:
  - `integration`: All integration tests
  - `image`: Image input processing tests
  - `image_generation`: Image generation tests
  - `classification`: Classification functionality tests
  - `tools`: Tool registry tests
  - `cli`: CLI functionality tests
  - `cross_provider`: Tests that verify behavior across providers
  - `streaming`: Streaming functionality tests
  - `cost`: Cost tracking tests
  - `rate_limiting`: Tests for rate limiting behaviors
  - `provider_errors`: Tests for provider error handling
  - `dynamic_model_selection`: Tests for model selection
  - `optional_parameters`: Tests for provider-specific parameters

## Packaging & Distribution Best Practices
- Use `python -m build` for building both wheel and sdist
- Verify packages with `twine check` before publishing
- Set up automated GitHub Actions for testing and publishing
- Implement proper versioning with semantic versioning
- Maintain backward compatibility when possible
- Include comprehensive README and documentation
- Configure CI checks for linting, type checking, and formatting
- Set up proper PyPI publishing credentials in GitHub Secrets

## GitHub Actions CI/CD Pipeline
A robust CI/CD pipeline should include:
- Testing on multiple Python versions (3.8, 3.9, 3.10)
- Code quality checks (linting, formatting, type hints)
- Test coverage reporting
- Package building and validation
- Automated PyPI publishing for tagged releases
- Documentation building and publishing

## Consolidated Task List
1. Complete provider test coverage
   - OpenAI Provider (40% → 75%+) - CRITICAL
   - Anthropic Provider (38% → 75%+) - CRITICAL
   - Provider Template (47% → 75%+)
2. Improve packaging configuration
   - Update pyproject.toml with complete metadata
   - Configure package data inclusion
   - Set up proper entry points
   - Implement single-source versioning
3. Implement CI/CD pipeline
   - Complete GitHub Actions workflow with testing, linting, and type checking
   - Set up version bumping and tagging
   - Configure publishing to PyPI
   - Implement semantic versioning automation
4. Enhance documentation
   - Add comprehensive docstrings
   - Generate API documentation with Sphinx
   - Create usage examples
   - Add architecture diagrams
5. Implement quality assurance tools
   - Add linting with ruff
   - Add formatting with black and isort
   - Add type checking with mypy
   - Set up pre-commit hooks
6. Standardize provider implementations
   - Document provider testing patterns
   - Enhance error handling across providers
   - Implement clearer parameter validation
   - Create unified approach to provider-specific features
