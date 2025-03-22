# LLuMinary Development Guide

## Project Overview

LLuMinary is a platform for interfacing with various LLM providers (OpenAI, Anthropic, Google, AWS Bedrock, Cohere) with a unified API.

## Command Reference

```bash
# Testing
python -m pytest tests/unit/ -v           # Unit tests
python -m pytest tests/integration/ -v    # Integration tests
make test-docker-unit                     # Run unit tests in Docker
make test-docker-integration              # Run integration tests in Docker

# Code quality
make lint                                 # Run linting (ruff)
make type-check                           # Run type checking (mypy)
make format                               # Format code (black)
make check                                # Run lint, type-check, and tests

# Docker
make docker-build                         # Build standard Docker image
make docker-build-matrix                  # Build matrix test Docker image
make test-docker DOCKER_TAG="lluminary-test:latest"  # Run all tests
make test-docker-file FILE="tests/unit/test_openai_*.py"  # Run specific tests
```

## Current Tasks

### Critical Priority

1. **Issue #3: Implement unified type definitions across providers**
   - [ ] Finish implementation in `src/lluminary/models/types.py`
   - [ ] Update provider files to use standard types:
     - [ ] `google.py`
     - [ ] `bedrock.py`
     - [ ] `cohere.py`
   - [ ] Add comprehensive type checking tests
   - [ ] Ensure mypy type checking passes with --strict flag

### High Priority

1. **Issue #4: Enhance streaming support for tool/function calling**
   - [ ] Analyze current streaming implementation across providers
   - [ ] Design unified streaming tool calls interface
   - [ ] Update types.py to include streaming tool call types
   - [ ] Add streaming tool call handling to base provider interface
   - [ ] Implement for each provider (OpenAI, Anthropic, Bedrock)
   - [ ] Create tests for streaming tool calls

2. **Issue #173: Implement Mistral AI provider support**
   - [ ] Research Mistral AI API documentation
   - [ ] Design integration with existing provider interfaces
   - [ ] Implement basic client for Mistral AI
   - [ ] Add streaming support
   - [ ] Add tool/function calling support
   - [ ] Implement token counting and cost estimation
   - [ ] Implement error handling and mapping
   - [ ] Write tests for implementation
   - [ ] Add API documentation and example script

3. **Issue #198: JFKReveal Integration - Phase 1: Core Compatibility**
   - [ ] Issue #199: Validate Embedding Compatibility with Qdrant
   - [ ] Issue #200: Enhance Function Calling 
   - [ ] Issue #201: Enhance Error Handling
     - [ ] Standardize error types across providers
     - [ ] Implement consistent retry mechanisms
     - [ ] Add detailed error messages

### Medium Priority

1. **Issue #5: Add vector database integration support**
   - [ ] Create `src/lluminary/vector_db/` module
   - [ ] Design base interface for vector storage
   - [ ] Add FAISS implementation
   - [ ] Add Pinecone implementation
   - [ ] Connect to existing embedding functionality
   - [ ] Add similarity search capabilities
   - [ ] Add example for RAG (Retrieval Augmented Generation)
   - [ ] Create unit and integration tests

2. **Issue #6: Implement robust caching mechanism**
   - [ ] Create `src/lluminary/cache/` module
   - [ ] Design base caching interface
   - [ ] Add in-memory cache implementation
   - [ ] Add disk-based cache implementation
   - [ ] Add Redis cache implementation
   - [ ] Implement embedding-based cache lookup
   - [ ] Create unit and integration tests

3. **Issue #202: JFKReveal Integration - Phase 2: Documentation and Examples**
   - [ ] Issue #203: Create Integration Guide
   - [ ] Issue #204: Develop JFKReveal Example

### Long-Term Goals

1. **Issue #181: Add support for local models via Ollama**
   - [ ] Research Ollama API and capabilities
   - [ ] Design Ollama integration with existing provider interfaces
   - [ ] Implement basic client for Ollama
   - [ ] Add offline text generation
   - [ ] Handle Ollama-specific configuration
   - [ ] Add streaming and tool/function calling support if available
   - [ ] Write tests and documentation

2. **Issue #189: Implement agent framework**
   - [ ] Create `src/lluminary/agents/` module
   - [ ] Design agent architecture and interfaces
   - [ ] Implement AgentExecutor for orchestrating agent steps
   - [ ] Create memory systems for conversation history
   - [ ] Implement planning capabilities for multi-step reasoning
   - [ ] Add tool registry for agent tool use
   - [ ] Create standard agent types
   - [ ] Connect to existing tools module
   - [ ] Create example applications and documentation

3. **Issue #205: JFKReveal Integration - Phase 3: Testing and Validation**
   - [ ] Issue #206: Integration Testing
   - [ ] Issue #207: Performance Testing
   - [ ] Issue #208: JFKReveal Integration with LLuMinary

4. **Issue #9: Add advanced observability and monitoring**
   - [ ] Create `src/lluminary/observability/` module
   - [ ] Add OpenTelemetry integration
   - [ ] Add tracing support
   - [ ] Add metrics collection
   - [ ] Implement Prometheus export
   - [ ] Add dashboard templates
   - [ ] Implement budget alerts and cost reporting
   - [ ] Create unit and integration tests

## Release Planning

### v1.1.0 (Next Release)
- Fix GitHub Actions workflow issues
- Implement unified type definitions
- Enhance streaming support for tool/function calls
- Add Mistral AI provider support

### v1.2.0
- Add vector database integration
- Implement caching mechanism
- Enhance documentation and examples
- Improve error handling across providers

### v2.0.0 (Major Release)
- Add Ollama support for local models
- Implement agent framework
- Add advanced observability and monitoring
- Comprehensive documentation update

## Implementation Guidelines

### Type Checking Guidelines

All code should pass mypy with strict mode: `mypy --strict src/lluminary`

For new types:
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

## Implementation Notes for Remaining Providers

When implementing the unified type system for the remaining providers (Google, Bedrock, Cohere):

1. **Import pattern**: Import all relevant types from `types.py`
2. **Provider-specific types**: Define provider-specific types that match API requirements while maintaining compatibility with base types
3. **Fixing inheritance issues**: Avoid TypedDict inheritance; create standalone TypedDict definitions
4. **Provider identification**: Set the provider attribute using the Provider enum
5. **Usage statistics**: Ensure all usage statistics include standardized fields
6. **Configuration validation**: Implement the `_validate_provider_config` method
7. **Testing**: Run mypy with strict checking to ensure type safety

## Session Notes

| Date         | Notes                                                          |
|--------------|----------------------------------------------------------------|
| 2025-03-10   | Identified Issues #3 and #4 as highest priorities              |
| 2025-03-17   | Consolidated development notes into CLAUDE.md                  |
| 2025-03-22   | Implemented unified types for Anthropic and OpenAI providers   |