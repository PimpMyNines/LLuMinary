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

## Priority Tasks

### Critical Issues (Fix First)

1. **Fix GitHub Actions Workflow Issues**
   - [x] Fix Dockerfile.matrix handling in CI workflow
   - [x] Fix conditional execution logic for provider-specific tests
   - [x] Ensure docker-build-matrix-cached target properly uses Dockerfile.matrix
   - [ ] Configure CODECOV_TOKEN in repository secrets for coverage reporting

2. **Issue #3: Implement unified type definitions across providers**
   - [x] Begin implementation in `src/lluminary/models/types.py`
   - [ ] Update provider files to use standard types:
     - [x] `anthropic.py` (partial)
     - [x] `openai.py` (partial)
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

## Session Notes

| Date         | Notes                                                          |
|--------------|----------------------------------------------------------------|
| 2025-03-17   | Consolidated development notes into CLAUDE.md; fixed CI workflow issues |
| 2025-03-22   | Added direct-to-main PR workflow; improved type system implementation |

---

This guide will be updated as progress is made on the issues and new challenges are identified.