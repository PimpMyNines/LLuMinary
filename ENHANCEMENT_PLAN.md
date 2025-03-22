# LLuMinary Enhancement Plan

This document outlines the enhancement plan for the LLuMinary project, focusing on the next major development phases and feature additions based on current priorities.

## Priority Tasks

### Critical Issues (Fix First)

1. **Fix GitHub Actions Workflow Issues**
   - [x] Fix Dockerfile.matrix handling in CI workflow
   - [x] Fix conditional execution logic for provider-specific tests
   - [x] Ensure docker-build-matrix-cached target properly uses Dockerfile.matrix
   - [ ] Configure CODECOV_TOKEN in repository secrets for coverage reporting
   - [ ] Add custom workflow to handle direct-to-main PRs for collaborators

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

## Contribution Guidelines

When contributing to these enhancements:

1. Create a feature branch from `develop` for your work
2. Ensure all tests pass before submitting PR
3. Add appropriate documentation for new features
4. Follow the existing code style and patterns
5. Include both unit and integration tests

## Implementation Notes

The priority of these enhancements may shift based on community feedback and project needs. Regular updates to this plan will be made as development progresses.