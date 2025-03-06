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
- Build package: `python setup.py sdist bdist_wheel`

## Code Style Guidelines
- Python version: 3.10+
- Type hints: Required for all function parameters and return values, use Optional/Union
- Naming: CamelCase for classes, snake_case for functions/variables, UPPER_SNAKE_CASE for constants
- Imports: Group in order: standard libs, third-party, local; alphabetical within groups
- Documentation: Google-style docstrings for all public classes and methods
- Error handling: Use custom exception classes from exceptions.py
- Design: Follow single responsibility principle and provider registry pattern
- Tests: Unit tests required for all new functionality (target 90%+ coverage)

## Provider Integration
- Maintain consistent interfaces across providers (OpenAI, Anthropic, Google, Cohere, Bedrock)
- Support: text generation, embeddings, streaming, reranking, image input, function calling
- Always implement token counting and cost tracking for each provider
- Follow provider_template.py pattern when adding new providers

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

## Next Steps
- Continue implementing tests for provider implementations:
  - Bedrock Provider (15% → 75%+)
  - Provider Template (0% → 75%+)
  - Complete Google Provider tests (14% → 75%+)
- Add type checking with mypy
- Improve robustness and error handling in integration tests
- Implement better integration between tools and LLM providers
- Set up CI/CD pipeline for running tests automatically
- Add performance benchmarking tests
- Implement provider-specific feature tests for untested capabilities