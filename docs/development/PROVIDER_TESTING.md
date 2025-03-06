# PROVIDER TESTING GUIDE

## Overview

This document provides comprehensive guidance on testing provider implementations in the LLuMinary library, using the Provider Template tests as a model for consistency and thorough test coverage.

## Table of Contents

- [Overview](#overview)
- [Test Structure for Providers](#test-structure-for-providers)
  - [Core Provider Tests](#1-core-provider-tests)
  - [Utility Tests](#2-utility-tests)
  - [Streaming Tests](#3-streaming-tests)
  - [Provider-Specific Feature Tests](#4-provider-specific-feature-tests)
- [Test Fixtures](#test-fixtures)
- [Key Testing Patterns](#key-testing-patterns)
- [Testing Challenges and Solutions](#testing-challenges-and-solutions)
- [Testing Coverage Requirements](#testing-coverage-requirements)
- [Running Provider Tests](#running-provider-tests)
- [Related Documentation](#related-documentation)

## Test Structure for Providers

Each provider implementation should have a comprehensive test suite organized into these categories:

### 1. Core Provider Tests

**File naming**: `test_[provider_name]_provider.py`

**Focus areas**:
- Provider static attributes (models, context windows, costs)
- Initialization with different parameters
- Authentication flows
- Message formatting
- Text generation (both successful and error cases)
- Helper methods (context window calculations, token counting)

**Example test class organization**:
```python
class Test[Provider]LLMAttributes:
    """Test static attributes of the provider class."""
    # Tests for model lists, context windows, costs

class Test[Provider]LLMInitialization:
    """Test initialization of provider instances."""
    # Tests for object creation, config parameters, error cases

class Test[Provider]Authentication:
    """Test authentication mechanisms."""
    # Tests for API key handling, AWS Secrets, etc.

class Test[Provider]MessageFormatting:
    """Test message formatting logic."""
    # Tests for converting standard messages to provider-specific format

class Test[Provider]Generation:
    """Test core text generation."""
    # Tests for the _raw_generate method
```

### 2. Utility Tests

**File naming**: `test_[provider_name]_utils.py`

**Focus areas**:
- Image processing methods
- Tool/function handling
- Special formatting requirements
- Provider-specific utility methods

### 3. Streaming Tests

**File naming**: `test_[provider_name]_streaming.py`

**Focus areas**:
- Streaming implementation
- Streaming error handling
- Streaming with tools/functions
- Token counting and usage stats in streaming mode

### 4. Provider-Specific Feature Tests

**File naming**: `test_[provider_name]_features.py`

**Focus areas**:
- Provider-specific capabilities not covered in the standard tests
- Special endpoint handling
- Custom parameter support
- Advanced features unique to this provider

## Test Fixtures

Use pytest fixtures to create reusable provider instances and minimize code duplication:

```python
@pytest.fixture
def provider_instance():
    """Create a basic provider instance for testing."""
    # Mock the auth method to avoid actual authentication
    with patch.object(ProviderLLM, "auth", return_value=None):
        provider = ProviderLLM(
            model_name="provider-model-1",
            timeout=30
        )
    return provider

@pytest.fixture
def mock_provider_env():
    """Set up mock environment variables for testing."""
    original_env = os.environ.copy()
    os.environ["PROVIDER_API_KEY"] = "test-api-key"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)
```

## Key Testing Patterns

### 1. Authentication Testing

Test authentication from different sources:

```python
def test_auth_from_env_var(self, mock_provider_env):
    """Test authentication using environment variables."""
    # Initialize with auth mocked to prevent actual auth during creation
    with patch.object(ProviderLLM, "auth", return_value=None):
        provider = ProviderLLM(model_name="provider-model-1")

    # Then call auth directly to test it
    # Restore the original method first
    provider.auth = ProviderLLM.auth.__get__(provider)
    provider.auth()

    assert provider.api_key == "test-api-key"
```

### 2. API Call Mocking

Mock external API calls and simulate responses:

```python
@patch("provider_sdk.Client")
def test_raw_generate(self, mock_client, provider_instance):
    # Set up mock client response
    mock_response = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }
    mock_client.return_value.generate.return_value = mock_response

    # Call the method under test
    result, usage = provider_instance._raw_generate(
        event_id="test-event",
        system_prompt="Test prompt",
        messages=[{"message_type": "human", "message": "Hello"}]
    )

    # Verify result and usage stats
    assert result == "Test response"
    assert usage["read_tokens"] == 10
    assert usage["write_tokens"] == 5
```

### 3. Error Handling Testing

Test how the provider handles API errors:

```python
@patch("provider_sdk.Client")
def test_raw_generate_error(self, mock_client, provider_instance):
    # Set up mock to raise an exception
    mock_client.return_value.generate.side_effect = Exception("API error")

    # Call the method and expect a LLMMistake exception
    with pytest.raises(LLMMistake) as excinfo:
        provider_instance._raw_generate(
            event_id="test-event",
            system_prompt="Test prompt",
            messages=[{"message_type": "human", "message": "Hello"}]
        )

    # Verify exception details
    assert "Error generating" in str(excinfo.value)
    assert excinfo.value.error_type == "api_error"
    assert excinfo.value.provider == "ProviderLLM"
```

### 4. Message Formatting Testing

Test the message format conversion:

```python
def test_message_formatting_with_images(self, provider_instance):
    messages = [
        {
            "message_type": "human",
            "message": "What's in this image?",
            "image_paths": ["/path/to/image.jpg"]
        }
    ]

    # Mock image processing
    with patch.object(provider_instance, "_process_image_file") as mock_process:
        mock_process.return_value = "encoded-image"

        # Format messages
        formatted = provider_instance._format_messages_for_model(messages)

        # Verify results
        assert formatted[0]["content"] == "What's in this image?"
        assert "images" in formatted[0]
        assert formatted[0]["images"][0] == "encoded-image"
```

### 5. Stream Testing

Test streaming implementations with generator functions:

```python
def test_stream_generate(self, provider_instance):
    # Set up mock streaming client
    mock_stream = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [{"delta": {"content": "!"}}]}
    ]

    with patch("provider_sdk.Client") as mock_client:
        # Configure the mock to return a generator
        mock_client.return_value.generate_stream.return_value = iter(mock_stream)

        # Call stream_generate and collect results
        chunks = []
        for chunk, usage in provider_instance._stream_generate(
            event_id="test-event",
            system_prompt="Test prompt",
            messages=[{"message_type": "human", "message": "Hi"}]
        ):
            chunks.append(chunk)

        # Verify results
        assert "".join(chunks) == "Hello world!"
```

## Testing Challenges and Solutions

### 1. Testing Without Real Credentials

**Challenge**: Testing provider API calls without exposing real credentials.

**Solution**:
- Mock the `auth` method during initialization
- Use environment variable fixtures for auth testing
- Mock the actual API client and its responses

### 2. Testing Streaming Responses

**Challenge**: Testing streaming implementations that use generators.

**Solution**:
- Mock the provider's streaming client to yield predetermined chunks
- Collect chunks in a list for verification
- Verify both content and usage statistics

### 3. Testing Rate Limiting and Retries

**Challenge**: Testing provider rate limiting and retry behavior.

**Solution**:
- Mock API responses to simulate 429 errors
- Test that the retry mechanism works correctly
- Verify exponential backoff implementation

### 4. Testing Image Processing

**Challenge**: Testing multimodal inputs without actual image files.

**Solution**:
- Mock file operations (open, read)
- Mock image processing libraries
- Use predetermined binary data and encoded results

## Testing Coverage Requirements

For a provider implementation to be considered production-ready, it should have:

1. **Core coverage**: 90%+ coverage of all methods
2. **Edge cases**: Tests for all error paths and edge cases
3. **Feature parity**: Tests for all features claimed in the provider documentation
4. **Integration tests**: At least basic integration tests with real API calls (skipped in CI)

## Running Provider Tests

```bash
# Run all tests for a specific provider
python -m pytest tests/unit/test_[provider]_*.py -v

# Run a specific category of tests
python -m pytest tests/unit/test_[provider]_streaming.py -v

# Run with coverage
python -m pytest tests/unit/test_[provider]_*.py --cov=src/lluminary/models/providers/[provider].py
```

## Related Documentation

- [API_REFERENCE](../API_REFERENCE.md) - Complete API reference for all components
- [MODELS](./MODELS.md) - Detailed information about model implementations
- [ERROR_HANDLING](./ERROR_HANDLING.md) - Error handling guidelines and implementation
- [ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION](./ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION.md) - Anthropic-specific error handling
- [OPENAI_ERROR_HANDLING_IMPLEMENTATION](./OPENAI_ERROR_HANDLING_IMPLEMENTATION.md) - OpenAI-specific error handling
- [TEST_COVERAGE](../TEST_COVERAGE.md) - Current test coverage status and goals
