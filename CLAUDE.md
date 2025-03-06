# LLMHandler Development Guide

## Project-Specific Commands
- Run tests by pattern: `python -m pytest tests/ -k "openai" -v`
- Run tests by marker: `python -m pytest tests/ -m "image" -v`
- Run tests with coverage: `python -m pytest tests/ --cov=src/llmhandler --cov-report=term`
- Install AWS dependencies: `pip install -e ".[aws]"`
- Build docs: `sphinx-build -b html docs/source docs/build/html`

## Provider Integration Guidelines
- Maintain consistent interfaces across providers (OpenAI, Anthropic, Google, Cohere, Bedrock)
- Support: text generation, embeddings, streaming, reranking, image input, function calling
- Always implement token counting and cost tracking for each provider
- Follow provider_template.py pattern when adding new providers
- Implement proper error handling with standardized exception types

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

## Current Project Status
- **Current coverage**: 70% (target: 90%+)
- **Passing tests**: 221 unit tests + 74 integration tests = 295 total tests
- **Progress**: Classification, Tools Validator, Cohere Provider (90%+), Google Provider (80%+), Bedrock Provider (75%+), all Integration tests completed
- **Critical Provider Status**: OpenAI (65%), Anthropic (38%) - require attention
- **Last updated**: March 13, 2025

## Test Coverage by Module
| Module | Coverage | Priority |
|--------|----------|----------|
| Router | 93% | ✅ Done |
| Handler | 73% | ✅ Good |
| Base LLM | 76% | ✅ Good |
| AWS Utils | 88% | ✅ Good |
| Exceptions | 67% | ✅ Good |
| Tool Registry | 66% | ✅ Good |
| Classification Components | 90%+ | ✅ Done |
| Classification CLI | 90%+ | ✅ Done |
| Anthropic Provider | 38% | 🔴 CRITICAL |
| OpenAI Provider | 65% | 🟡 Medium |
| Google Provider | 80%+ | ✅ Done |
| Bedrock Provider | 75%+ | ✅ Done |
| Tools Validators | 90% | ✅ Done |
| CLI (other components) | N/A | ✅ Done |
| Cohere Provider | 90%+ | ✅ Done |
| Provider Template | 47% | 🟡 Medium |

## Priority Tasks

### 1. Provider Test Coverage (High Priority)
- 🔴 **CRITICAL**: Complete OpenAI Provider tests (65% → 75%+)
  - Implement authentication tests with proper environment variable fallback
  - Add embedding functionality tests
  - ✅ Add streaming tests with complex scenarios
  - ✅ Add reranking functionality tests
  - ✅ Implement token counting and cost tracking tests
  - ✅ Add image generation tests
  - ✅ Implement error handling and recovery tests
- 🔴 **CRITICAL**: Complete Anthropic Provider tests (38% → 75%+)
  - Add embeddings tests
  - Implement authentication flow tests
  - Expand error handling tests
  - Add tests for thinking budget behavior
  - Add timeout handling tests
- 🟡 Complete Provider Template tests (47% → 75%+)
  - Implement token counting tests
  - Fix embedding and reranking test implementation
  - Add image processing test fixtures
  - Implement tool validation and formatting tests
  - Add provider registration and discovery tests

### 2. Package Structure Refinement (High Priority)
- Update pyproject.toml
  - Add complete classifiers
  - Configure package data inclusion
  - Add development dependencies
  - Set up proper entry points
- Implement proper versioning
  - Implement single-source versioning
  - Set up version bumping automation
- Add package metadata
  - Complete documentation in README.md
  - Add licenses and other required files

### 3. CI/CD Pipeline Implementation (High Priority)
- Complete GitHub Actions workflow
  - Add testing step before publishing
  - Implement linting and type checking
  - Set up version bumping and tagging
  - Configure publishing to PimpMyNines organization
- Implement release automation
  - Add changelog generation
  - Implement semantic versioning automation
  - Add release artifacts creation
- Add quality checks
  - Configure black, isort, and ruff for code formatting
  - Add mypy for type checking
  - Set up pre-commit hooks for local development

### 4. Documentation Enhancement (Medium Priority)
- Enhance API documentation
  - Add comprehensive docstrings
  - Generate API documentation
  - Create usage examples
- Set up documentation build
  - Implement sphinx documentation
  - Add automatic documentation deployment
- Add architecture diagrams
  - Create component diagrams
  - Document provider interfaces
  - Add sequence diagrams for key workflows

### 5. Provider Interface Improvements (Medium Priority)
- ✅ Standardize error handling across providers
- Implement clearer parameter validation
  - Add validation for common parameters
  - Improve error messages for invalid inputs
- Create unified approach to provider-specific features
  - Document provider capabilities
  - Implement feature detection
- Improve configuration validation
  - Add schema validation for config
  - Enhance error reporting for config issues

## Agent Assignments

### Agent 1: Provider Testing
- **Focus**: Complete OpenAI and Anthropic provider testing
- **Current task**: Implementing tests for OpenAI Provider (65% → 75%+) and Anthropic Provider (38% → 75%+)
- **Progress on OpenAI Provider**:
  - 196 passing tests implemented (up from 142)
  - Current coverage ~65% (up from 40%)
  - Target: 75%+ coverage
- **Next steps for OpenAI Provider**:
  - Complete implementation of additional embedding functionality tests
  - Improve authentication tests
  - Continue improving error handling tests for edge cases

### Agent 2: Package Structure and CI/CD
- **Focus**: Implement proper packaging, CI/CD pipeline, and documentation
- **Current task**: Update pyproject.toml, create GitHub Actions workflow
- **Next steps**:
  - Complete pyproject.toml with all necessary configuration
  - Implement GitHub Actions workflow with testing, quality checks, and publishing
  - Set up documentation generation with Sphinx
  - Create proper versioning mechanism

### Agent 3: Provider Template and Quality Tooling
- **Focus**: Complete Provider Template tests, implement quality tools
- **Current task**: Improving Provider Template coverage (47% → 75%+)
- **Next steps**:
  - Fix token counting tests in Provider Template
  - Complete embedding and reranking tests
  - Set up linting, formatting, and type checking
  - Implement pre-commit hooks


## Future Improvements
- **LOGGING**: Add option for streaming logs to remote data store
- **GUARDRAILS**: Add ability to add global and/or model specific guardrails
- **VOICE**: Add voice support using whisper or openai voice.
