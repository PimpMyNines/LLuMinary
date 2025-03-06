# Test Coverage Reference

This document provides an overview of the current test coverage for the LLM Handler library and identifies areas where additional testing is needed.

## Table of Contents

- [Current Coverage Summary](#current-coverage-summary)
- [Component Coverage](#component-coverage)
- [Newly Implemented Tests](#newly-implemented-tests)
- [Areas Needing Additional Tests](#areas-needing-additional-tests)
- [Test Implementation Guides](#test-implementation-guides)

## Current Coverage Summary

As of the latest update, the library has:

- 84 unit tests (including 9 CLI tests and 17 enhanced validator tests)
- Overall coverage: ~50%
- Target coverage: 90%+

| Module | Coverage | Priority | Status |
|--------|----------|----------|--------|
| Router | 93% | ✅ Done | Complete |
| Handler | 73% | ✅ Good | Complete |
| Base LLM | 76% | ✅ Good | Complete |
| AWS Utils | 88% | ✅ Good | Complete |
| Exceptions | 67% | ✅ Good | Complete |
| Tool Registry | 66% | ✅ Good | Complete |
| CLI | 85% | ✅ Good | Complete |
| Tools Validators | 80% | ✅ Good | Complete |
| Anthropic Provider | 38% | 🟡 Medium | In Progress |
| OpenAI Provider | 36% | 🟡 Medium | In Progress |
| Classification Components | 23-47% | 🔴 High | Needs Work |
| Google Provider | 14% | 🔴 High | In Progress |
| Bedrock Provider | 15% | 🔴 High | Needs Work |
| Cohere Provider | 0% | 🔴 High | Not Started |
| Provider Template | 0% | 🔴 High | Not Started |

## Component Coverage

### Core Components (Good Coverage)

- `Router`: Near complete coverage at 93%. Tests ensure proper model mapping and provider instantiation.
- `Handler`: Good coverage at 73%. Tests cover primary generation, error handling, and provider selection.
- `Base LLM`: Good coverage at 76%. Most key methods have tests, including generation, embedding, and reranking.

### Provider Implementations (Needs Improvement)

- `OpenAI Provider`: Partial coverage at 36%. Tests cover basic operations but miss streaming and complex scenarios.
- `Anthropic Provider`: Partial coverage at 38%. Additional tests needed for error handling and edge cases.
- `Google Provider`: Low coverage at 14%. Significant test gaps for most functionality.
- `Bedrock Provider`: Low coverage at 15%. Key functionality not thoroughly tested.
- `Cohere Provider`: No coverage (0%). No tests implemented yet.
- `Provider Template`: No coverage (0%). Template not tested.

### CLI & Tools (Newly Implemented)

- `CLI`: Good coverage with comprehensive tests for all classify commands.
- `Tools Validators`: Enhanced with good coverage for complex nested types, unions, and JSON serialization.

### Classification (Needs Improvement)

- `Classification Config`: Low coverage at 36%. More validation tests needed.
- `Classification Validators`: Very low coverage at 23%. Critical validation logic not fully tested.
- `Classifier`: Moderate coverage at 47%. Basic functionality tested, but edge cases not covered.

## Newly Implemented Tests

### CLI Tests

The classify CLI module now has comprehensive tests covering:

1. **List Configurations Command**
   - Tests with valid config directories
   - Tests with empty directories
   - Proper display of configuration details

2. **Validate Command**
   - Tests with valid configurations
   - Tests with invalid configurations
   - Error handling

3. **Test Command**
   - Basic classification tests
   - Custom model selection
   - System prompt customization
   - Error handling

4. **Create Command**
   - Interactive configuration creation
   - Validation of created configurations
   - Error handling for invalid inputs

### Enhanced Validator Tests

The enhanced type validation system now has detailed tests covering:

1. **Simple Type Validation**
   - Basic type checks (int, str, bool, float)
   - Container types (list, dict, set, tuple)

2. **Nested Type Validation**
   - Simple and complex nested dictionaries
   - Lists of complex objects
   - Multi-level nested structures

3. **Union and Optional Types**
   - Testing Union[Type1, Type2]
   - Testing Optional[Type] (equivalent to Union[Type, None])
   - Complex union combinations

4. **Container Type Validation**
   - Generic List[T], Dict[K, V], Set[T], Tuple checks
   - Validation with Any type
   - Variable-length tuple support (Tuple[T, ...])

5. **JSON Serialization**
   - Valid JSON structures
   - Detection of non-serializable objects
   - Nested serializable checks

## Areas Needing Additional Tests

### Provider Tests

1. **Cohere Provider**
   - Basic generation tests
   - Embedding tests
   - Reranking tests
   - Error handling tests

2. **Google Provider**
   - Comprehensive generation tests
   - Streaming tests
   - Image input tests
   - Error handling and retry logic

3. **Bedrock Provider**
   - Model-specific tests
   - AWS authentication tests
   - Token counting accuracy tests
   - Error handling for AWS-specific errors

### Classification Tests

1. **Config Validation**
   - Category validation
   - Example validation
   - Schema validation

2. **Classifier Logic**
   - Model selection tests
   - Classification prompt construction
   - Response parsing
   - Error handling

### Integration Tests

1. **End-to-End Workflows**
   - Complete user flows
   - Multi-provider comparisons
   - Performance testing

2. **Error Recovery**
   - Rate limit handling
   - API failure recovery
   - Retry logic verification

## Test Implementation Guides

### Provider Test Template

When implementing provider tests, follow this structure:

```python
def test_provider_basic_generation():
    """Test basic text generation with the provider."""
    # Arrange - Setup the provider with mock responses

    # Act - Call the generate method

    # Assert - Verify correct response and token counting

def test_provider_streaming():
    """Test streaming generation with the provider."""
    # Test implementation

def test_provider_embeddings():
    """Test embedding generation with the provider."""
    # Test implementation

def test_provider_reranking():
    """Test document reranking with the provider."""
    # Test implementation

def test_provider_error_handling():
    """Test error handling for various API failures."""
    # Test implementation
```

### Mock Framework

Use the shared mock framework defined in the AGENT_NOTES.md file:

```python
class MockLLM:
    """Base mock LLM implementation for testing."""

    SUPPORTED_MODELS = ["mock-model"]
    THINKING_MODELS = ["mock-model"]
    EMBEDDING_MODELS = ["mock-model"]
    RERANKING_MODELS = []

    CONTEXT_WINDOW = {
        "mock-model": 4096
    }

    COST_PER_MODEL = {
        "mock-model": {
            "read_token": 0.001,
            "write_token": 0.002,
            "image_cost": 0.01
        }
    }

    def __init__(self, model_name="mock-model", **kwargs):
        self.model_name = model_name
        self.config = kwargs

    # Additional mock methods...
```

### Classification Test Approach

For classification tests, use this approach:

```python
def test_classify_from_config():
    """Test classification from config object."""
    # Setup mock classifier and config
    config = ClassificationConfig(...)

    # Setup mock LLM with predefined response
    llm = MockLLM()

    # Call classify method
    categories, usage = llm.classify(config=config, messages=[...])

    # Verify correct categories and usage calculation
    assert categories == [...]
    assert usage["total_tokens"] == ...
    assert usage["total_cost"] == ...
```
