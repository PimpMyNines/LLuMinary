# LLuMinary Development Guide

This document serves as the central reference for LLuMinary project development, combining the enhancement plan, implementation notes, and development lessons to maintain a clean repository structure while preserving essential information for future agents.

## Project Overview

LLuMinary is a platform for interfacing with various LLM providers (OpenAI, Anthropic, Google, AWS Bedrock, Cohere) with a unified API. The project aims to expand support for additional providers and features while maintaining a consistent interface.

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

## Project Consolidation Summary

The project was originally developed with two parallel package structures:
- `src/llmhandler/` - Original package name
- `src/lluminary/` - New package name

This caused inconsistency issues in imports, testing, and configuration. The consolidation effort fixed these issues by standardizing on `lluminary` as the package name.

### Changes Made

1. **Version Update**
   - Updated version number to `1.0.0` (from 0.0.1/0.1.0)
   - Ensured email domain is consistently set to `@pimpmynines.com`

2. **Package Structure**
   - Removed the duplicate `src/llmhandler/` directory and its egg-info
   - Kept only the `src/lluminary/` package structure

3. **Import Fixes**
   - Fixed 334 import statements across 116 files
   - Replaced `from src.lluminary` with `from lluminary` imports
   - Replaced `import src.lluminary` with `import lluminary` imports

4. **Path Manipulation Cleanup**
   - Removed sys.path modifications from 7 test files
   - Eliminated unnecessary path manipulation code

5. **Configuration Updates**
   - Updated mypy.ini configuration to use `src.lluminary` paths
   - Added comprehensive ignore patterns for third-party libraries

6. **API Consistency**
   - Fixed handler class naming inconsistency:
     - Updated `LLuMinary` to `LLMHandler` in handler.py
     - Added alias in `__init__.py` for backwards compatibility: `LLMHandler as LLuMinary`
   - Added `set_provider_config` and `get_provider_config` functions to models/__init__.py

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
   - [x] Fix Dockerfile.matrix handling in CI workflow
   - [x] Fix conditional execution logic for provider-specific tests
   - [x] Ensure docker-build-matrix-cached target properly uses Dockerfile.matrix
   - [x] Add detailed documentation for CODECOV_TOKEN setup
   - [x] Implement error detection and reporting for missing CODECOV_TOKEN
   - [x] Add custom workflow to handle direct-to-main PRs for collaborators

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

### Recent Progress

1. **GitHub Actions and Documentation Improvements**
   - Fixed workflow configuration for proper CI/CD integration
   - Added comprehensive documentation for GitHub Actions
   - Improved Codecov integration for test coverage reporting
   - Enhanced error handling in CI workflow scripts
   - Added direct-to-main PR workflow for streamlined contributions

2. **Unified Type System Implementation**
   - Enhanced types.py with comprehensive type definitions
   - Updated OpenAI provider to use standardized types
   - Added structured type system for all providers
   - Improved type safety throughout the codebase
   - Documented type system architecture

## Implementation Guidelines

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
- ✅ AWS Profile support for authentication
- ✅ Region support for us-east-1
- ✅ Thinking budget support (Claude 3.7)

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
| 2025-03-22   | Added direct-to-main PR workflow; improved type system implementation |

---

This guide will be updated as progress is made on the issues and new challenges are identified.