# LLuminary Project - Development Guide

This document serves as the central reference for the LLuminary project development. It consolidates all development plans, tasks, and implementation notes to maintain a clean repository structure while preserving essential information.

## Project Overview

LLuminary is a platform for interfacing with various LLM providers (OpenAI, Anthropic, Google, AWS Bedrock, Cohere) with a unified API. The project aims to expand support for additional providers and features while maintaining a consistent interface.

## Development Status

Current version: v0.1.0 (Beta)

**Implemented Features:**
- ✅ Core abstraction layer for LLM providers
- ✅ Complete implementations for OpenAI, Anthropic, Google AI, Cohere, and AWS Bedrock
- ✅ Unified error handling with standardized exception types
- ✅ Type-safe interfaces with comprehensive error handling
- ✅ Support for text generation, streaming, and embeddings
- ✅ Function/tool calling capabilities
- ✅ Image understanding for multimodal models
- ✅ Token counting and cost estimation
- ✅ Classification functionality
- ✅ Document reranking
- ✅ Comprehensive test suite (>85% coverage)
- ✅ Docker-based testing and CI/CD pipeline

## Command Reference

### Testing Commands

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

### Docker Testing Commands

```bash
# Build Docker image
make docker-build                         # Build standard Docker image
make docker-build-matrix                  # Build matrix test Docker image

# Run tests in Docker
make test-docker DOCKER_TAG="lluminary-test:latest"  # Run all tests
make test-docker-file FILE="tests/unit/test_openai_*.py"  # Run specific tests
```

## Priority Tasks

### Critical Issues (Fix First)

1. **Fix GitHub Actions Workflow Issues**
   - [ ] Fix Dockerfile.matrix handling in CI workflow
   - [ ] Fix conditional execution logic for provider-specific tests
   - [ ] Ensure docker-build-matrix-cached target properly uses Dockerfile.matrix
   - [ ] Configure CODECOV_TOKEN in repository secrets for coverage reporting

2. **Issue #3: Implement unified type definitions across providers**
   - [ ] Finish implementation in `src/lluminary/models/types.py`
   - [ ] Update provider files to use standard types:
     - [ ] `anthropic.py`
     - [ ] `openai.py`
     - [ ] `google.py`
     - [ ] `bedrock.py`
     - [ ] `cohere.py`
   - [ ] Add comprehensive type checking tests
   - [ ] Ensure mypy type checking passes with --strict flag

### High Priority Features

1. **Issue #4: Enhance streaming support for tool/function calling**
   - [ ] Analyze current streaming implementation across providers
   - [ ] Design unified streaming tool calls interface
   - [ ] Update types.py to include streaming tool call types
   - [ ] Add streaming tool call handling to base provider interface
   - [ ] Implement for each provider (OpenAI, Anthropic, Bedrock)
   - [ ] Create tests for streaming tool calls

2. **Issue #2: Add support for Mistral AI provider**
   - [ ] Create new provider file `src/lluminary/models/providers/mistral.py`
   - [ ] Implement core MistralLLM class based on BaseLLM
   - [ ] Add authentication via API key
   - [ ] Implement text generation and streaming
   - [ ] Implement token counting and cost estimation
   - [ ] Implement error handling and mapping
   - [ ] Create unit and integration tests
   - [ ] Add API documentation and example script

### Medium Priority Features

1. **Issue #5: Add vector database integration support**
   - [ ] Create `src/lluminary/vector_db/` module
   - [ ] Design base interface for vector storage
   - [ ] Add FAISS implementation
   - [ ] Add Pinecone implementation
   - [ ] Connect to existing embedding functionality
   - [ ] Add similarity search capabilities
   - [ ] Create unit and integration tests

2. **Issue #6: Implement robust caching mechanism**
   - [ ] Create `src/lluminary/cache/` module
   - [ ] Design base caching interface
   - [ ] Add in-memory cache implementation
   - [ ] Add disk-based cache implementation
   - [ ] Add Redis cache implementation
   - [ ] Implement embedding-based cache lookup
   - [ ] Create unit and integration tests

### Long-Term Goals

1. **Issue #7: Add support for local models via Ollama**
   - [ ] Create `src/lluminary/models/providers/ollama.py`
   - [ ] Implement OllamaLLM class based on BaseLLM
   - [ ] Add offline text generation
   - [ ] Handle Ollama-specific configuration
   - [ ] Create unit and integration tests

2. **Issue #8: Implement agent framework**
   - [ ] Create `src/lluminary/agents/` module
   - [ ] Design base agent interfaces and abstractions
   - [ ] Add memory systems
   - [ ] Implement planning capabilities
   - [ ] Create standard agent types
   - [ ] Connect to existing tools module
   - [ ] Create unit and integration tests

3. **Issue #9: Add advanced observability and monitoring**
   - [ ] Create `src/lluminary/observability/` module
   - [ ] Add OpenTelemetry integration
   - [ ] Add tracing support
   - [ ] Add metrics collection
   - [ ] Implement Prometheus export
   - [ ] Add dashboard templates
   - [ ] Implement budget alerts and cost reporting
   - [ ] Create unit and integration tests

## Implementation Guidelines

### CI/CD Workflow Testing Procedure

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

### Adding New Providers Checklist

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

### Type Checking Guidelines

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

### Error Handling Best Practices

1. **Standard Exception Hierarchy**
   - Use exceptions defined in `src/lluminary/exceptions.py`
   - Map provider-specific errors to our standard exceptions
   - Include helpful error messages with context

2. **Consistent Error Mapping**
   - Categorize errors by type (Authentication, Rate Limiting, etc.)
   - Add debug information when appropriate
   - Preserve original error information when wrapping exceptions

3. **Retry Logic**
   - Implement appropriate backoff for retryable errors
   - Add clear logging for retry attempts
   - Use configurable retry policies

## Provider Implementation Status

### OpenAI
- ✅ Text generation
- ✅ Streaming
- ✅ Function calling
- ✅ Image understanding
- ✅ Embeddings
- ✅ Token counting
- ✅ Reranking
- ✅ Error handling

### Anthropic
- ✅ Text generation
- ✅ Streaming
- ✅ Function calling (tool use)
- ✅ Image understanding
- ✅ Embeddings
- ✅ Token counting
- ✅ Error handling

### Google
- ✅ Text generation
- ✅ Streaming
- ✅ Function calling
- ✅ Image understanding
- ✅ Embeddings
- ✅ Token counting
- ✅ Error handling

### AWS Bedrock
- ✅ Text generation
- ✅ Streaming
- ✅ Function calling (partial)
- ✅ Image understanding
- ✅ Embeddings (selected models)
- ✅ Token counting
- ✅ Error handling

### Cohere
- ✅ Text generation
- ✅ Streaming
- ✅ Embeddings
- ✅ Reranking
- ✅ Token counting
- ✅ Error handling

## Session Notes

| Date         | Notes                                                          |
|--------------|----------------------------------------------------------------|
| 2025-03-10   | Identified Issues #3 and #4 as highest priorities; began work on types.py |
| 2025-03-17   | Consolidated development notes into CLAUDE.md; fixed CI workflow issues in progress |

---

This guide will be updated as progress is made on the issues and new challenges are identified.
