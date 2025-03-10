# LLuminary Project - Development Roadmap and Issue Resolution Guide

This document provides a structured guide for addressing the open issues in the LLuminary project, ensuring GitHub Actions pass successfully, and maintaining code quality.

## Project Overview

LLuminary is a platform for interfacing with various LLM providers (OpenAI, Anthropic, Google, AWS Bedrock, Cohere) with a unified API. The project aims to expand support for additional providers and features.

## Testing and Quality Checks

Essential testing commands for LLuminary:

```bash
# Basic testing
python -m pytest tests/unit/ -v           # Unit tests
python -m pytest tests/integration/ -v    # Integration tests

# Docker testing (safer, more isolated)
make test-docker-unit                     # Run unit tests in Docker
make test-docker-integration              # Run integration tests in Docker

# Code quality checks
make lint                                 # Run linting (ruff)
make type-check                           # Run type checking (mypy)
make format                               # Format code (black)

# Combined checks
make check                                # Run lint, type-check, and tests
```

## GitHub Actions Workflow Issues

The `.github/workflows/matrix-docker-tests.yml` file has the following issues that need to be addressed to ensure CI passes:

1. **Dockerfile.matrix Handling**:
   - Fix the workflow to properly use the dynamically created Dockerfile.matrix

2. **Build Command Consistency**:
   - Ensure docker-build-matrix-cached target properly uses the Dockerfile.matrix

3. **Provider Tests Logic**:
   - Improve the conditional logic for provider-specific tests
   - Ensure the FILE parameter is properly passed to the test-docker-file command

4. **Secret Requirements**:
   - Ensure CODECOV_TOKEN is set in GitHub repository secrets for coverage reporting

## Open GitHub Issues and Resolution Plan

### High Priority

#### Issue #3: Implement unified type definitions across providers

- **Progress**: Initial implementation in `src/lluminary/models/types.py`
- **Next Steps**:
  1. Audit all provider files to use standard types:
     - [ ] Update `anthropic.py` to use unified types
     - [ ] Update `openai.py` to use unified types
     - [ ] Update `google.py` to use unified types
     - [ ] Update `bedrock.py` to use unified types
     - [ ] Update `cohere.py` to use unified types
  2. Add comprehensive type checking tests:
     - [ ] Create/update `tests/unit/test_types.py`
     - [ ] Add specific type validation tests for each provider
  3. Ensure mypy type checking works with --strict flag:
     - [ ] Fix any type errors identified with mypy

#### Issue #4: Enhance streaming support for tool/function calling

- **Next Steps**:
  1. Analyze current streaming implementation across providers:
     - [ ] Review OpenAI streaming implementation
     - [ ] Review Anthropic streaming implementation
     - [ ] Review Bedrock streaming implementation
  2. Design unified streaming tool calls interface:
     - [ ] Update types.py to include streaming tool call types
     - [ ] Add streaming tool call handling to base provider interface
  3. Implement for each provider:
     - [ ] Add streaming tool calls for OpenAI
     - [ ] Add streaming tool calls for Anthropic
     - [ ] Add streaming tool calls for Bedrock
  4. Create tests:
     - [ ] Add unit tests for streaming tool calls
     - [ ] Add integration tests for streaming tool calls

### Medium Priority

#### Issue #2: Add support for Mistral AI provider

- **Next Steps**:
  1. Create new provider file:
     - [ ] Create `src/lluminary/models/providers/mistral.py`
     - [ ] Implement core MistralLLM class based on BaseLLM
  2. Implement key features:
     - [ ] Add authentication via API key
     - [ ] Implement text generation and streaming
     - [ ] Implement token counting and cost estimation
     - [ ] Implement error handling and mapping
  3. Add tests:
     - [ ] Create unit tests (`tests/unit/test_mistral_*.py`)
     - [ ] Create integration tests if possible
  4. Add documentation and examples:
     - [ ] Add API documentation for Mistral provider
     - [ ] Create example script for Mistral usage

#### Issue #5: Add vector database integration support

- **Next Steps**:
  1. Design vector storage abstraction:
     - [ ] Create `src/lluminary/vector_db/` module
     - [ ] Design base interface for vector storage
  2. Implement adapters:
     - [ ] Add FAISS implementation
     - [ ] Add Pinecone implementation
  3. Add integration with embeddings:
     - [ ] Connect to existing embedding functionality
     - [ ] Add similarity search capabilities
  4. Create tests:
     - [ ] Add unit tests for vector storage
     - [ ] Add integration tests for full functionality

#### Issue #6: Implement robust caching mechanism

- **Next Steps**:
  1. Design caching interface:
     - [ ] Create `src/lluminary/cache/` module
     - [ ] Design base caching interface
  2. Implement providers:
     - [ ] Add in-memory cache implementation
     - [ ] Add disk-based cache implementation
     - [ ] Add Redis cache implementation
  3. Add semantic similarity caching:
     - [ ] Implement embedding-based cache lookup
  4. Create tests:
     - [ ] Add unit tests for each cache type
     - [ ] Add integration tests for caching behavior

### Long-Term Goals

#### Issue #7: Add support for local models via Ollama

- **Next Steps**:
  1. Create new provider:
     - [ ] Create `src/lluminary/models/providers/ollama.py`
     - [ ] Implement OllamaLLM class based on BaseLLM
  2. Implement key features:
     - [ ] Add offline text generation
     - [ ] Handle Ollama-specific configuration
  3. Add tests:
     - [ ] Create unit tests with mocked Ollama
     - [ ] Create integration tests for local deployment

#### Issue #8: Implement agent framework

- **Next Steps**:
  1. Design agent architecture:
     - [ ] Create `src/lluminary/agents/` module
     - [ ] Design base agent interfaces and abstractions
  2. Implement components:
     - [ ] Add memory systems
     - [ ] Implement planning capabilities
     - [ ] Create standard agent types
  3. Add tools integration:
     - [ ] Connect to existing tools module
  4. Create tests:
     - [ ] Add unit tests for agent components
     - [ ] Add integration tests for agent behaviors

#### Issue #9: Add advanced observability and monitoring

- **Next Steps**:
  1. Implement OpenTelemetry integration:
     - [ ] Create `src/lluminary/observability/` module
     - [ ] Add tracing support
     - [ ] Add metrics collection
  2. Add monitoring features:
     - [ ] Implement Prometheus export
     - [ ] Add dashboard templates
  3. Add cost tracking:
     - [ ] Implement budget alerts
     - [ ] Add cost reporting
  4. Create tests:
     - [ ] Add unit tests for observability components
     - [ ] Add integration tests for monitoring

## Github Actions Workflow Testing Procedure

Before creating a PR, run the following tests to ensure CI will pass:

1. Run local tests:
   ```bash
   make check
   ```

2. Run Docker tests:
   ```bash
   make docker-build
   make test-docker-unit
   make test-docker-integration
   ```

3. Check Docker matrix tests:
   ```bash
   # Create Dockerfile.matrix first
   make docker-build-matrix
   make test-docker DOCKER_TAG="lluminary-test:matrix"
   ```

4. Specific provider tests:
   ```bash
   # Example for OpenAI tests
   make test-docker-file FILE="tests/unit/test_openai_*.py"
   ```

## Adding New Providers Checklist

When implementing a new provider:

1. [ ] Create provider file in `src/lluminary/models/providers/`
2. [ ] Implement provider class extending BaseLLM
3. [ ] Support core functionalities:
   - [ ] Authentication
   - [ ] Text generation
   - [ ] Streaming
   - [ ] Error handling
   - [ ] Token counting
4. [ ] Add unit tests
5. [ ] Add integration tests
6. [ ] Update documentation
7. [ ] Add example usage script

## Type Checking Guidelines

All code should pass mypy with strict mode:

```bash
mypy --strict src/lluminary
```

For new types, follow these guidelines:
1. Add base types to `src/lluminary/models/types.py`
2. For provider-specific extensions, create a class that inherits from the base type
3. Use TypedDict for structured data
4. Use Literal types for enumeration values
5. Use Protocol classes for duck typing interfaces

## Session Notes

| Date         | Notes                                                          |
|--------------|----------------------------------------------------------------|
| 2025-03-10   | Identified Issues #3 and #4 as highest priorities; began work on types.py |
| [Next session] | [Notes for next session]                                      |

---

This guide will be updated as progress is made on the issues and new challenges are identified.
