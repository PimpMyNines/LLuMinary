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

## Implementation Plan and Priorities

The following implementation plan is organized by priority and dependencies. Each task is broken down for independent agent work.

### Priority 1: Foundation and Infrastructure Tasks

#### 1.1. Fix GitHub Actions Workflow Issues - ✅ COMPLETED
**Impact**: Ensures reliable CI/CD pipeline for all future development
**Dependencies**: None - can be addressed immediately

##### Agent Tasks:
1. **CI Workflow Analysis Agent**
   - ✅ Review existing GitHub workflow files in `.github/workflows/`
   - ✅ Identify issues in matrix test configuration
   - ✅ Analyze conditional execution logic failures
   
2. **Dockerfile Matrix Agent**
   - ✅ Fix Dockerfile.matrix configuration 
   - ✅ Ensure proper layer caching and dependencies
   - ✅ Add proper test configuration for matrix testing
   
3. **Coverage Reporting Agent**
   - ✅ Configure CODECOV_TOKEN in repository secrets
   - ✅ Add coverage reporting steps to workflow
   - ✅ Verify coverage reports are being generated correctly

##### Improvements Made:
1. **Fixed Dockerfile.matrix Handling**
   - ✅ Updated matrix-docker-tests.yml to use existing Dockerfile.matrix instead of creating a temporary one
   - ✅ Clarified docker-build-matrix-cached Makefile target documentation
   - ✅ Ensured proper handling of PYTHON_VERSION build argument

2. **Improved Provider Test Conditional Logic**
   - ✅ Enhanced conditional execution logic for provider-specific tests
   - ✅ Added support for test-related file changes (not just provider implementation)
   - ✅ Improved detection of relevant commits

3. **Enhanced CODECOV_TOKEN Integration**
   - ✅ Updated README.md with detailed instructions for setting up CODECOV_TOKEN
   - ✅ Added warnings about required repository secrets
   - ✅ Created verification scripts for testing workflow and codecov setup

#### 1.2. Implement Unified Type Definitions (Issue #3) - IN PROGRESS
**Impact**: Ensures type consistency across providers, improves developer experience
**Dependencies**: None - can be addressed immediately

##### Agent Tasks:
1. **Type System Design Agent**
   - ✅ Complete implementation in `src/lluminary/models/types.py`
   - ✅ Design comprehensive type hierarchy for messages, completions, embeddings
   - ✅ Ensure compatibility with all providers' type systems
   
2. **Provider Type Implementation Agent**
   - [ ] Update each provider file to use standard types:
     - [ ] `anthropic.py`: Update to use shared message and completion types
     - [ ] `openai.py`: Implement standardized function call and response types
     - [ ] `google.py`: Convert to use common embedding and generation types
     - [ ] `bedrock.py`: Adapt model-specific types to common interfaces
     - [ ] `cohere.py`: Standardize reranking and embedding types
   
3. **Type Testing Agent**
   - [ ] Create comprehensive type checking tests
   - [ ] Implement test cases that validate correct type conversions
   - [ ] Ensure mypy type checking passes with --strict flag
   - [ ] Add type validation tests for edge cases

##### Progress Made:
1. **Enhanced Base Type System**
   - ✅ Comprehensive Provider enum class added for consistent provider identification
   - ✅ Standardized content part types (text, images, tool use, etc.)
   - ✅ Unified message types with proper role handling
   - ✅ Robust tool/function definition types with consistent interface
   - ✅ Complete API request/response type structures
   - ✅ Added streaming type definitions for all providers
   - ✅ Added embedding and reranking type definitions
   - ✅ Enhanced error handling type structures

2. **Type Conversion and Standardization**
   - ✅ Added standardized input/output message formats
   - ✅ Created protocol classes for function call processing
   - ✅ Added authentication config type definition
   - ✅ Enhanced usage statistics with latency and timing fields

##### Remaining Tasks:
- [ ] Update provider implementations to use the new type definitions
- [ ] Add conversion functions between provider-specific and standardized types
- [ ] Ensure type compatibility with streaming tool calls
- [ ] Create type tests to verify compatibility
- [ ] Update documentation with type examples

### Priority 2: Core Functionality Enhancements

#### 2.1. Enhance Streaming Support for Tool/Function Calling (Issue #4)
**Impact**: Enables more responsive applications using tool calls with streaming
**Dependencies**: Depends on unified type definitions (1.2)

##### Agent Tasks:
1. **Streaming Architecture Agent**
   - [ ] Analyze current streaming implementations across providers
   - [ ] Design unified streaming tool calls interface
   - [ ] Update types.py to include streaming tool call types
   - [ ] Create streaming tool call abstraction in base provider interface
   
2. **Provider Streaming Implementation Agent**
   - [ ] Implement OpenAI streaming tool calls
   - [ ] Implement Anthropic streaming tool calls
   - [ ] Implement Bedrock streaming tool calls with provider-specific adaptations
   
3. **Streaming Testing Agent**
   - [ ] Create streaming tool call test suite
   - [ ] Develop tests for multiple tool calls in a single stream
   - [ ] Test for edge cases and error handling during streaming
   - [ ] Ensure proper end-to-end streaming behavior

#### 2.2. Implement Retry Mechanism with Backoff
**Impact**: Improves reliability by handling transient failures automatically
**Dependencies**: None - can be implemented independently

##### Agent Tasks:
1. **Retry Mechanism Design Agent**
   - [ ] Implement `src/lluminary/utils/retry.py` module
   - [ ] Set up backoff library integration
   - [ ] Create configurable retry policies
   - [ ] Design logging for retry attempts

2. **Provider Retry Integration Agent**
   - [ ] Add retry support to BaseLLM class
   - [ ] Implement `with_retry` method that returns a retry-capable LLM instance
   - [ ] Apply retry decorators to key provider methods
   - [ ] Handle API-specific exceptions properly

3. **Retry Testing Agent**
   - [ ] Create tests that simulate transient failures
   - [ ] Verify correct backoff behavior
   - [ ] Test time-based and attempt-based retry policies
   - [ ] Validate retry logging and monitoring

### Priority 3: New Providers and Integrations

#### 3.1. Add Support for Mistral AI Provider (Issue #2)
**Impact**: Expands supported models with a high-performance open model provider
**Dependencies**: Unified type definitions (1.2) and retry mechanism (2.2)

##### Agent Tasks:
1. **Mistral API Integration Agent**
   - [ ] Create new provider file `src/lluminary/models/providers/mistral.py`
   - [ ] Implement core MistralLLM class extending BaseLLM
   - [ ] Add authentication via API key
   - [ ] Implement Mistral-specific parameter handling

2. **Mistral Functionality Agent**
   - [ ] Implement text generation and streaming
   - [ ] Implement token counting and cost estimation
   - [ ] Add support for Mistral's function calling
   - [ ] Handle Mistral's embedding API

3. **Mistral Testing and Documentation Agent**
   - [ ] Create unit and integration tests for Mistral
   - [ ] Add API documentation and example script
   - [ ] Create comprehensive test matrix for Mistral models
   - [ ] Test compatibility with unified type system

#### 3.2. Add Provider-Agnostic Prompt System
**Impact**: Simplifies multi-provider usage with unified prompt format
**Dependencies**: Unified type definitions (1.2)

##### Agent Tasks:
1. **Prompt System Design Agent**
   - [ ] Create `src/lluminary/prompts/agnostic.py` module
   - [ ] Implement provider-agnostic message representation
   - [ ] Design format conversion logic
   - [ ] Create unified prompt templating system

2. **Provider Format Conversion Agent**
   - [ ] Implement Anthropic format conversions
   - [ ] Implement OpenAI format conversions
   - [ ] Implement Google format conversions
   - [ ] Implement Cohere and other provider conversions

3. **Prompt Testing Agent**
   - [ ] Create tests for template formatting
   - [ ] Test bidirectional format conversions
   - [ ] Validate template variables and substitutions
   - [ ] Test integration with provider generate methods

### Priority 4: Vector Database Integration (Issue #5)

#### 4.1. Vector Database Base Implementation
**Impact**: Enables semantic search and retrieval capabilities
**Dependencies**: None - can be implemented independently

##### Agent Tasks:
1. **Vector DB Interface Agent**
   - [ ] Create `src/lluminary/vectorstores/base.py` module
   - [ ] Design abstract VectorStore class with core methods
   - [ ] Define standardized interfaces for add_texts, similarity_search
   - [ ] Create common metadata handling patterns

2. **Embedding Integration Agent**
   - [ ] Connect vector stores to existing embedding functionality
   - [ ] Create embedding function factory for vector stores
   - [ ] Design embedding cache to improve performance
   - [ ] Ensure compatibility with all provider embedding formats

3. **Factory Function Agent**
   - [ ] Implement unified factory function `get_vector_store`
   - [ ] Create store type auto-detection
   - [ ] Add embedding model configuration
   - [ ] Implement options validation

#### 4.2. Vector Database Implementations
**Impact**: Provides multiple storage options for different use cases
**Dependencies**: Vector database base implementation (4.1)

##### Agent Tasks:
1. **ChromaDB Implementation Agent**
   - [ ] Create `src/lluminary/vectorstores/chroma.py` module
   - [ ] Implement ChromaStore class
   - [ ] Add configuration options and settings class
   - [ ] Handle ChromaDB-specific functionality

2. **Qdrant Implementation Agent**
   - [ ] Create `src/lluminary/vectorstores/qdrant.py` module
   - [ ] Implement QdrantStore class
   - [ ] Set up Qdrant client management
   - [ ] Handle Qdrant-specific search features

3. **SQL Implementation Agent**
   - [ ] Create `src/lluminary/vectorstores/pgvector.py` module
   - [ ] Implement PostgreSQL with pgvector support
   - [ ] Set up connection pooling
   - [ ] Add index optimization for performance

4. **OpenSearch Implementation Agent**
   - [ ] Create `src/lluminary/vectorstores/opensearch.py` module
   - [ ] Implement OpenSearchStore class
   - [ ] Set up authentication and connection handling
   - [ ] Add support for OpenSearch vector features

#### 4.3. Vector DB Testing and Documentation
**Impact**: Ensures reliability and usability of vector database features
**Dependencies**: Vector database implementations (4.2)

##### Agent Tasks:
1. **Vector DB Unit Testing Agent**
   - [ ] Create comprehensive test suite for each vector store
   - [ ] Test add, search, and delete operations
   - [ ] Test metadata filtering and scoring
   - [ ] Test error handling and recovery

2. **Vector DB Integration Testing Agent**
   - [ ] Create end-to-end tests with real embedding models
   - [ ] Test performance with various data sizes
   - [ ] Test multi-user concurrent access patterns
   - [ ] Test persistence and recovery

3. **Vector DB Documentation Agent**
   - [ ] Create comprehensive API documentation
   - [ ] Add usage examples for each vector store
   - [ ] Create performance comparison guide
   - [ ] Document advanced configuration options

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

## Development Lessons

### Type System Development

#### Lesson 1: Design for Compatibility Across Providers

When designing a unified type system for multiple LLM providers:

1. **Start with the lowest common denominator**: Identify the core features that all providers support (messages, content types, basic parameters) and create a solid foundation.

2. **Use TypedDict with `total=False`**: This allows for optional fields, which is essential when dealing with provider-specific extensions.

3. **Create clear type hierarchies**: Message types should inherit from base types to maintain consistency while allowing specialization.

4. **Use Union types for flexibility**: Different providers handle content in different ways (strings vs arrays of content parts).

5. **Consider protocol classes**: For behaviors rather than structures, Protocol classes provide flexibility with type safety.

#### Lesson 2: Managing Provider-Specific Extensions

Providers like OpenAI and Anthropic have unique features that need special handling:

1. **Extend base types**: Create provider-specific extensions that inherit from base types while adding custom fields.

2. **Create conversion functions**: Implement utility functions to convert between standard types and provider-specific formats.

3. **Document type relationships**: Add clear comments showing the mapping between standard types and provider formats.

### GitHub Actions Workflow Optimization

#### Lesson 1: Matrix Testing Efficiency

When configuring matrix testing across multiple Python versions:

1. **Use shared Dockerfile with ARG**: Define a base Dockerfile.matrix with ARG PYTHON_VERSION to avoid duplication.

2. **Leverage buildx caching**: Properly configure cache paths to speed up builds across workflow runs.

3. **Use conditional execution wisely**: Provider-specific tests should only run when relevant files change.

#### Lesson 2: Fixing Conditional Logic

Common issues with GitHub Actions conditional logic:

1. **Careful syntax in conditionals**: Use proper parentheses and formatting for complex conditions.

2. **Test pattern matching thoroughly**: File pattern matching needs to account for both implementation and test files.

3. **Document secret requirements**: Be explicit about required secrets like CODECOV_TOKEN.

### Implementation Strategy

#### General Approach to Feature Development

1. **Implement in layers**:
   - First, create base types and interfaces
   - Then implement provider-specific adaptations
   - Finally, add tests to verify behavior

2. **Balance backward compatibility with improvements**:
   - Use optional fields and default values
   - Add deprecation warnings for changed APIs
   - Keep original method signatures intact when possible

3. **Documentation-driven development**:
   - Update docstrings with examples
   - Document changes in this file
   - Keep session notes to track progress

## Next Tasks

After completing each task:
1. Review this file to identify the next highest priority item
2. Complete the identified task
3. Update this file with the progress made
4. Add additional development lessons as they are discovered

### Next Tasks in Queue

1. **Complete Type System Implementation**
   - Update at least one provider file to use the new types (anthropic.py or openai.py)
   - Create a conversion utility to map between provider types and standard types
   - Add basic tests to verify type compatibility

2. **Begin Streaming Function Call Enhancement (Issue #4)**
   - Analyze existing streaming implementations
   - Design unified streaming tool calls interface
   - Focus on the most commonly used providers first (OpenAI, Anthropic)

## Dependency Map

```
[1.1 GitHub Actions] → [All other tasks benefit]
[1.2 Type Definitions] → [2.1 Streaming] → [3.1 Mistral AI]
                       → [3.2 Prompts]
                       → [4.x Vector DB] → [5.2 Doc Processing]
                                         → [6.2 Semantic Cache]
[2.2 Retry] → [3.1 Mistral AI]
           → [All provider interactions benefit]
[4.1 Vector Base] → [4.2 Vector Implementations] → [4.3 Vector Testing]
[5.1 Doc Loaders] → [5.2 Doc Processing]
[6.1 Cache Base] → [6.2 Semantic Cache]
```

## Session Notes

| Date         | Notes                                                          |
|--------------|----------------------------------------------------------------|
| 2025-03-10   | Identified Issues #3 and #4 as highest priorities; began work on types.py |
| 2025-03-17   | Consolidated development notes into CLAUDE.md; fixed CI workflow issues in progress |
| 2025-03-22   | Completed GitHub Actions workflow fixes (Priority 1.1 from plan):<br>- Fixed Dockerfile.matrix handling with build args<br>- Improved provider test conditional execution<br>- Enhanced coverage reporting with proper dependencies<br>- Added verification scripts and required secrets documentation<br>- Next priority: Implement unified type definitions (Priority 1.2) |
| 2025-03-22   | Made substantial progress on unified type definitions (Priority 1.2):<br>- Enhanced `types.py` with comprehensive type system<br>- Added Provider enum for consistent identification<br>- Created unified content, message, and tool types<br>- Added streaming, embedding, and reranking type definitions<br>- Implemented authentication type structures<br>- Next steps: Update provider implementations to use new types |
| 2025-03-22   | Combined LESSONS.md and ENHANCEMENT_PLAN.md into CLAUDE.md for a single comprehensive development guide for future agents. Reorganized plan with completed status updates. |

---

This guide will be updated as progress is made on the issues and new challenges are identified.