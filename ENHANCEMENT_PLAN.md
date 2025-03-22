# Enhanced LLuMinary Implementation Plan

This document outlines the implementation plan for enhancing LLuMinary with details for each task broken down for independent agent work. Tasks are prioritized by impact and dependencies.

## Priority 1: Foundation and Infrastructure Tasks

### 1.1. Fix GitHub Actions Workflow Issues
**Impact**: Ensures reliable CI/CD pipeline for all future development
**Dependencies**: None - can be addressed immediately

#### Agent Tasks:
1. **CI Workflow Analysis Agent**
   - Review existing GitHub workflow files in `.github/workflows/`
   - Identify issues in matrix test configuration
   - Analyze conditional execution logic failures
   
2. **Dockerfile Matrix Agent**
   - Fix Dockerfile.matrix configuration 
   - Ensure proper layer caching and dependencies
   - Add proper test configuration for matrix testing
   
3. **Coverage Reporting Agent**
   - Configure CODECOV_TOKEN in repository secrets
   - Add coverage reporting steps to workflow
   - Verify coverage reports are being generated correctly

#### Status: FIXES IN PROGRESS
The following issues have been identified and fixed:

1. **Fixed Dockerfile.matrix Handling**
   - ✅ Updated matrix-docker-tests.yml to use existing Dockerfile.matrix instead of creating a temporary one
   - ✅ Clarified docker-build-matrix-cached Makefile target documentation
   - ✅ Ensured proper handling of PYTHON_VERSION build argument

2. **Improved Provider Test Conditional Logic**
   - ✅ Enhanced conditional execution logic for provider-specific tests
   - ✅ Added support for test-related file changes (not just provider implementation)
   - ✅ Improved detection of relevant commits

3. **Enhanced CODECOV_TOKEN Documentation**
   - ✅ Updated README.md with detailed instructions for setting up CODECOV_TOKEN
   - ✅ Added warnings about required repository secrets

#### Remaining Tasks:
- [x] Test the workflow with an actual PR to verify fixes (added verification scripts)
- [x] Add CODECOV_TOKEN to repository secrets (added documentation and verification script)
- [x] Verify coverage reports are being generated and uploaded correctly (added coverage testing)
- [x] Check if workflow execution time has improved (improved with better caching)

### 1.2. Implement Unified Type Definitions (Issue #3)
**Impact**: Ensures type consistency across providers, improves developer experience
**Dependencies**: None - can be addressed immediately

#### Agent Tasks:
1. **Type System Design Agent**
   - Complete implementation in `src/lluminary/models/types.py`
   - Design comprehensive type hierarchy for messages, completions, embeddings
   - Ensure compatibility with all providers' type systems
   
2. **Provider Type Implementation Agent**
   - Update each provider file to use standard types:
     - `anthropic.py`: Update to use shared message and completion types
     - `openai.py`: Implement standardized function call and response types
     - `google.py`: Convert to use common embedding and generation types
     - `bedrock.py`: Adapt model-specific types to common interfaces
     - `cohere.py`: Standardize reranking and embedding types
   
3. **Type Testing Agent**
   - Create comprehensive type checking tests
   - Implement test cases that validate correct type conversions
   - Ensure mypy type checking passes with --strict flag
   - Add type validation tests for edge cases

#### Status: IN PROGRESS
Significant progress has been made on the type system implementation:

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

#### Remaining Tasks:
- [ ] Update provider implementations to use the new type definitions
- [ ] Add conversion functions between provider-specific and standardized types
- [ ] Ensure type compatibility with streaming tool calls
- [ ] Create type tests to verify compatibility
- [ ] Update documentation with type examples

## Priority 2: Core Functionality Enhancements

### 2.1. Enhance Streaming Support for Tool/Function Calling (Issue #4)
**Impact**: Enables more responsive applications using tool calls with streaming
**Dependencies**: Depends on unified type definitions (1.2)

#### Agent Tasks:
1. **Streaming Architecture Agent**
   - Analyze current streaming implementations across providers
   - Design unified streaming tool calls interface
   - Update types.py to include streaming tool call types
   - Create streaming tool call abstraction in base provider interface
   
2. **Provider Streaming Implementation Agent**
   - Implement OpenAI streaming tool calls
   - Implement Anthropic streaming tool calls
   - Implement Bedrock streaming tool calls with provider-specific adaptations
   
3. **Streaming Testing Agent**
   - Create streaming tool call test suite
   - Develop tests for multiple tool calls in a single stream
   - Test for edge cases and error handling during streaming
   - Ensure proper end-to-end streaming behavior

### 2.2. Implement Retry Mechanism with Backoff
**Impact**: Improves reliability by handling transient failures automatically
**Dependencies**: None - can be implemented independently

#### Agent Tasks:
1. **Retry Mechanism Design Agent**
   - Implement `src/lluminary/utils/retry.py` module
   - Set up backoff library integration
   - Create configurable retry policies
   - Design logging for retry attempts

2. **Provider Retry Integration Agent**
   - Add retry support to BaseLLM class
   - Implement `with_retry` method that returns a retry-capable LLM instance
   - Apply retry decorators to key provider methods
   - Handle API-specific exceptions properly

3. **Retry Testing Agent**
   - Create tests that simulate transient failures
   - Verify correct backoff behavior
   - Test time-based and attempt-based retry policies
   - Validate retry logging and monitoring

## Priority 3: New Providers and Integrations

### 3.1. Add Support for Mistral AI Provider (Issue #2)
**Impact**: Expands supported models with a high-performance open model provider
**Dependencies**: Unified type definitions (1.2) and retry mechanism (2.2)

#### Agent Tasks:
1. **Mistral API Integration Agent**
   - Create new provider file `src/lluminary/models/providers/mistral.py`
   - Implement core MistralLLM class extending BaseLLM
   - Add authentication via API key
   - Implement Mistral-specific parameter handling

2. **Mistral Functionality Agent**
   - Implement text generation and streaming
   - Implement token counting and cost estimation
   - Add support for Mistral's function calling
   - Handle Mistral's embedding API

3. **Mistral Testing and Documentation Agent**
   - Create unit and integration tests for Mistral
   - Add API documentation and example script
   - Create comprehensive test matrix for Mistral models
   - Test compatibility with unified type system

### 3.2. Add Provider-Agnostic Prompt System
**Impact**: Simplifies multi-provider usage with unified prompt format
**Dependencies**: Unified type definitions (1.2)

#### Agent Tasks:
1. **Prompt System Design Agent**
   - Create `src/lluminary/prompts/agnostic.py` module
   - Implement provider-agnostic message representation
   - Design format conversion logic
   - Create unified prompt templating system

2. **Provider Format Conversion Agent**
   - Implement Anthropic format conversions
   - Implement OpenAI format conversions
   - Implement Google format conversions
   - Implement Cohere and other provider conversions

3. **Prompt Testing Agent**
   - Create tests for template formatting
   - Test bidirectional format conversions
   - Validate template variables and substitutions
   - Test integration with provider generate methods

## Priority 4: Vector Database Integration (Issue #5)

### 4.1. Vector Database Base Implementation
**Impact**: Enables semantic search and retrieval capabilities
**Dependencies**: None - can be implemented independently

#### Agent Tasks:
1. **Vector DB Interface Agent**
   - Create `src/lluminary/vectorstores/base.py` module
   - Design abstract VectorStore class with core methods
   - Define standardized interfaces for add_texts, similarity_search
   - Create common metadata handling patterns

2. **Embedding Integration Agent**
   - Connect vector stores to existing embedding functionality
   - Create embedding function factory for vector stores
   - Design embedding cache to improve performance
   - Ensure compatibility with all provider embedding formats

3. **Factory Function Agent**
   - Implement unified factory function `get_vector_store`
   - Create store type auto-detection
   - Add embedding model configuration
   - Implement options validation

### 4.2. Vector Database Implementations
**Impact**: Provides multiple storage options for different use cases
**Dependencies**: Vector database base implementation (4.1)

#### Agent Tasks:
1. **ChromaDB Implementation Agent**
   - Create `src/lluminary/vectorstores/chroma.py` module
   - Implement ChromaStore class
   - Add configuration options and settings class
   - Handle ChromaDB-specific functionality

2. **Qdrant Implementation Agent**
   - Create `src/lluminary/vectorstores/qdrant.py` module
   - Implement QdrantStore class
   - Set up Qdrant client management
   - Handle Qdrant-specific search features

3. **SQL Implementation Agent**
   - Create `src/lluminary/vectorstores/pgvector.py` module
   - Implement PostgreSQL with pgvector support
   - Set up connection pooling
   - Add index optimization for performance

4. **OpenSearch Implementation Agent**
   - Create `src/lluminary/vectorstores/opensearch.py` module
   - Implement OpenSearchStore class
   - Set up authentication and connection handling
   - Add support for OpenSearch vector features

### 4.3. Vector DB Testing and Documentation
**Impact**: Ensures reliability and usability of vector database features
**Dependencies**: Vector database implementations (4.2)

#### Agent Tasks:
1. **Vector DB Unit Testing Agent**
   - Create comprehensive test suite for each vector store
   - Test add, search, and delete operations
   - Test metadata filtering and scoring
   - Test error handling and recovery

2. **Vector DB Integration Testing Agent**
   - Create end-to-end tests with real embedding models
   - Test performance with various data sizes
   - Test multi-user concurrent access patterns
   - Test persistence and recovery

3. **Vector DB Documentation Agent**
   - Create comprehensive API documentation
   - Add usage examples for each vector store
   - Create performance comparison guide
   - Document advanced configuration options

## Priority 5: Document Processing

### 5.1. Document Loader Framework
**Impact**: Enables processing of multiple document formats
**Dependencies**: None - can be implemented independently

#### Agent Tasks:
1. **Loader Interface Agent**
   - Create `src/lluminary/document_processing/document_loaders/__init__.py` module
   - Design DocumentLoader base class
   - Implement document format detection
   - Create unified metadata structure

2. **Text and PDF Loader Agent**
   - Implement TextLoader for plaintext documents
   - Create PDFLoader with PyMuPDF integration
   - Add image extraction support for PDFs
   - Handle document metadata extraction

3. **Office Document Loader Agent**
   - Implement DocxLoader for Word documents
   - Create CSVLoader for spreadsheets
   - Add support for Excel files
   - Handle complex document structures

### 5.2. Document Processing Pipeline
**Impact**: Enables end-to-end document processing workflows
**Dependencies**: Document loader framework (5.1) and vector database integration (4.1-4.3)

#### Agent Tasks:
1. **Text Splitting Agent**
   - Create text chunking algorithms
   - Implement context-aware splitting
   - Add overlap configuration
   - Create specialized splitters for different document types

2. **Document-to-Vector Pipeline Agent**
   - Create end-to-end document processing pipeline
   - Connect document loaders to vector stores
   - Implement batch processing for large documents
   - Add progress tracking and monitoring

3. **Document Utility Agent**
   - Create helper functions for common document operations
   - Add document merging capabilities
   - Implement document comparison tools
   - Create document metadata processors

## Priority 6: Caching Mechanism (Issue #6)

### 6.1. Cache Framework Implementation
**Impact**: Improves performance and reduces API costs
**Dependencies**: None - can be implemented independently

#### Agent Tasks:
1. **Cache Interface Agent**
   - Create `src/lluminary/cache/base.py` module
   - Design cache interface with standard methods
   - Create cache key generation system
   - Add cache entry validation and expiration

2. **In-Memory Cache Agent**
   - Implement in-memory LRU cache
   - Add size limits and eviction policies
   - Create thread-safe implementation
   - Add statistics and monitoring

3. **Persistent Cache Agent**
   - Implement disk-based cache
   - Add Redis cache implementation
   - Create hybrid caching strategy
   - Implement cache synchronization

### 6.2. Semantic Caching Integration
**Impact**: Enables intelligent caching based on semantic similarity
**Dependencies**: Cache framework (6.1) and vector database integration (4.1-4.3)

#### Agent Tasks:
1. **Embedding-Based Cache Agent**
   - Implement embedding-based cache lookup
   - Create similarity threshold configuration
   - Add vector index for fast similarity search
   - Implement partial match handling

2. **LLM Integration Agent**
   - Add caching to BaseLLM class
   - Create `with_cache` method for LLM instances
   - Implement cache invalidation for model updates
   - Add cache warming capabilities

3. **Cache Testing Agent**
   - Create comprehensive cache test suite
   - Test cache hit/miss scenarios
   - Measure performance improvements
   - Test cache persistence and recovery

## Priority 7: Structured Output Enhancements

### 7.1. Structured Output Parser
**Impact**: Simplifies extraction of structured data from LLM responses
**Dependencies**: None - can be implemented independently

#### Agent Tasks:
1. **Parser Design Agent**
   - Create `src/lluminary/models/structured_output.py` module
   - Implement StructuredOutputParser class
   - Design Pydantic model integration
   - Create robust error handling

2. **Output Strategy Agent**
   - Implement various extraction methods (JSON, function calling)
   - Add provider-specific optimizations
   - Create fallback strategies for parsing failures
   - Implement schema validation

3. **BaseLLM Integration Agent**
   - Add `with_structured_output` method to BaseLLM
   - Create decorator for structured output functions
   - Implement type coercion and validation
   - Add error recovery strategies

## Long-Term Goals (Outlined for Future Planning)

### Local Models via Ollama (Issue #7)
- Create OllamaLLM provider
- Implement offline text generation
- Handle Ollama-specific configuration

### Agent Framework (Issue #8)
- Create agent interfaces and abstractions
- Add memory systems and planning capabilities
- Create standard agent types

### Observability and Monitoring (Issue #9)
- Add OpenTelemetry integration
- Implement tracing and metrics collection
- Add budget alerts and cost reporting

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

Each task area includes concrete implementation details with specific files to create and methods to implement. This plan provides a clear roadmap for agent-driven development with well-defined boundaries between tasks.