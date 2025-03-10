"""
Tests for OpenAI provider error handling functionality.

This module focuses specifically on testing the error handling
mechanisms of the OpenAI provider in isolation.
"""

from unittest.mock import MagicMock, patch

import pytest
from lluminary.exceptions import (
    AuthenticationError,
    ConnectionError,
    ContentPolicyViolationError,
    ContextLengthExceededError,
    InvalidRequestError,
    LLMHandlerError,
    RateLimitExceededError,
    ServiceUnavailableError,
    TimeoutError,
)
from lluminary.models.providers.openai import OpenAILLM
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
)
from openai import AuthenticationError as OpenAIAuthError


def test_authentication_error_mapping():
    """Test mapping of OpenAI authentication error to LLMHandler error."""
    provider = OpenAILLM("gpt-4o")

    # Mock methods to avoid actual API calls
    provider.client = MagicMock()

    # Create a mock OpenAI authentication error
    openai_error = OpenAIAuthError("Invalid authentication")

    # Test error mapping
    with pytest.raises(AuthenticationError) as exc_info:
        provider._map_openai_error(openai_error)

    # Verify mapped error contains original error details
    assert "Invalid authentication" in str(exc_info.value)
    assert isinstance(exc_info.value, AuthenticationError)


def test_rate_limit_error_mapping():
    """Test mapping of OpenAI rate limit error to LLMHandler error."""
    provider = OpenAILLM("gpt-4o")

    # Mock methods to avoid actual API calls
    provider.client = MagicMock()

    # Create a mock OpenAI rate limit error
    openai_error = RateLimitError("Rate limit exceeded")

    # Test error mapping
    with pytest.raises(RateLimitExceededError) as exc_info:
        provider._map_openai_error(openai_error)

    # Verify mapped error contains original error details
    assert "Rate limit exceeded" in str(exc_info.value)
    assert isinstance(exc_info.value, RateLimitExceededError)


def test_context_length_error_mapping():
    """Test mapping of OpenAI context length error to LLMHandler error."""
    provider = OpenAILLM("gpt-4o")

    # Mock methods to avoid actual API calls
    provider.client = MagicMock()

    # Create a mock OpenAI error with context length message
    openai_error = BadRequestError("Maximum context length exceeded")

    # Test error mapping
    with pytest.raises(ContextLengthExceededError) as exc_info:
        provider._map_openai_error(openai_error)

    # Verify mapped error contains original error details
    assert "Maximum context length exceeded" in str(exc_info.value)
    assert isinstance(exc_info.value, ContextLengthExceededError)


def test_content_policy_error_mapping():
    """Test mapping of OpenAI content policy error to LLMHandler error."""
    provider = OpenAILLM("gpt-4o")

    # Mock methods to avoid actual API calls
    provider.client = MagicMock()

    # Create a mock OpenAI error with content policy message
    openai_error = BadRequestError(
        "Your request was rejected as a result of our safety system"
    )

    # Test error mapping
    with pytest.raises(ContentPolicyViolationError) as exc_info:
        provider._map_openai_error(openai_error)

    # Verify mapped error contains original error details
    assert "rejected as a result of our safety system" in str(exc_info.value)
    assert isinstance(exc_info.value, ContentPolicyViolationError)


def test_server_error_mapping():
    """Test mapping of OpenAI server error to LLMHandler error."""
    provider = OpenAILLM("gpt-4o")

    # Mock methods to avoid actual API calls
    provider.client = MagicMock()

    # Create a mock OpenAI server error
    openai_error = InternalServerError("Internal server error")

    # Test error mapping
    with pytest.raises(ServiceUnavailableError) as exc_info:
        provider._map_openai_error(openai_error)

    # Verify mapped error contains original error details
    assert "Internal server error" in str(exc_info.value)
    assert isinstance(exc_info.value, ServiceUnavailableError)


def test_timeout_error_mapping():
    """Test mapping of OpenAI timeout error to LLMHandler error."""
    provider = OpenAILLM("gpt-4o")

    # Mock methods to avoid actual API calls
    provider.client = MagicMock()

    # Create a mock OpenAI timeout error
    openai_error = APITimeoutError("Request timed out")

    # Test error mapping
    with pytest.raises(TimeoutError) as exc_info:
        provider._map_openai_error(openai_error)

    # Verify mapped error contains original error details
    assert "Request timed out" in str(exc_info.value)
    assert isinstance(exc_info.value, TimeoutError)


def test_connection_error_mapping():
    """Test mapping of OpenAI connection error to LLMHandler error."""
    provider = OpenAILLM("gpt-4o")

    # Mock methods to avoid actual API calls
    provider.client = MagicMock()

    # Create a mock OpenAI connection error
    openai_error = APIConnectionError("Failed to connect to OpenAI API")

    # Test error mapping
    with pytest.raises(ConnectionError) as exc_info:
        provider._map_openai_error(openai_error)

    # Verify mapped error contains original error details
    assert "Failed to connect to OpenAI API" in str(exc_info.value)
    assert isinstance(exc_info.value, ConnectionError)


def test_unknown_error_mapping():
    """Test mapping of unknown OpenAI error to LLMHandler error."""
    provider = OpenAILLM("gpt-4o")

    # Mock methods to avoid actual API calls
    provider.client = MagicMock()

    # Create a generic OpenAI error
    openai_error = OpenAIError("Unknown error occurred")

    # Test error mapping
    with pytest.raises(LLMHandlerError) as exc_info:
        provider._map_openai_error(openai_error)

    # Verify mapped error contains original error details
    assert "Unknown error occurred" in str(exc_info.value)
    assert isinstance(exc_info.value, LLMHandlerError)


def test_bad_request_error_mapping():
    """Test mapping of OpenAI bad request error to LLMHandler error."""
    provider = OpenAILLM("gpt-4o")

    # Mock methods to avoid actual API calls
    provider.client = MagicMock()

    # Create a bad request error that doesn't match other patterns
    openai_error = BadRequestError("Invalid parameter: model")

    # Test error mapping
    with pytest.raises(InvalidRequestError) as exc_info:
        provider._map_openai_error(openai_error)

    # Verify mapped error contains original error details
    assert "Invalid parameter: model" in str(exc_info.value)
    assert isinstance(exc_info.value, InvalidRequestError)


def test_retry_on_rate_limit():
    """Test retry mechanism for rate limit errors."""
    provider = OpenAILLM("gpt-4o")

    # Mock the client and completions method
    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = [
        RateLimitError(
            "Rate limit exceeded. Please retry after 1s"
        ),  # First call fails
        {
            "choices": [{"message": {"content": "Response after retry"}}]
        },  # Second call succeeds
    ]

    # Set up provider state
    provider.authenticated = True

    # Configure provider with retries
    provider.config["retry_count"] = 1
    provider.config["initial_retry_delay"] = 0.01  # Small delay for testing

    # Call generate which should trigger the retry mechanism
    with patch("time.sleep") as mock_sleep:  # Mock sleep to speed up test
        response = provider.generate(
            messages=[{"role": "user", "content": "Test message"}]
        )

    # Verify result and that retry happened
    assert "Response after retry" in response
    assert provider.client.chat.completions.create.call_count == 2
    assert mock_sleep.call_count == 1  # Sleep should be called for backoff


def test_retry_with_retry_after_header():
    """Test retry mechanism using retry-after header information."""
    provider = OpenAILLM("gpt-4o")

    # Create a rate limit error with headers
    error_with_headers = RateLimitError("Rate limit exceeded")
    error_with_headers.headers = {"retry-after": "2"}  # 2 seconds retry-after

    # Mock client
    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = [
        error_with_headers,  # First call fails with header
        {
            "choices": [{"message": {"content": "Response after header-guided retry"}}]
        },  # Second succeeds
    ]

    # Set up provider state
    provider.authenticated = True

    # Configure provider with retries
    provider.config["retry_count"] = 1
    provider.config["initial_retry_delay"] = 1  # Should be overridden by header

    # Call generate which should trigger the retry mechanism
    with patch("time.sleep") as mock_sleep:  # Mock sleep to speed up test
        response = provider.generate(
            messages=[{"role": "user", "content": "Test message"}]
        )

    # Verify sleep was called with the right duration from header
    mock_sleep.assert_called_once_with(2)
    assert "Response after header-guided retry" in response


def test_retry_not_applied_for_non_retryable_errors():
    """Test that retry is not applied for non-retryable errors."""
    provider = OpenAILLM("gpt-4o")

    # Mock the client with authentication error (non-retryable)
    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = OpenAIAuthError(
        "Invalid API key"
    )

    # Set up provider state
    provider.authenticated = True

    # Configure provider with retries
    provider.config["retry_count"] = 3

    # Call generate which should not retry for auth errors
    with patch("time.sleep") as mock_sleep:
        with pytest.raises(AuthenticationError):
            provider.generate(messages=[{"role": "user", "content": "Test message"}])

    # Verify no retry was attempted
    assert provider.client.chat.completions.create.call_count == 1
    assert mock_sleep.call_count == 0


def test_exponential_backoff():
    """Test exponential backoff for retries."""
    provider = OpenAILLM("gpt-4o")

    # Mock client with multiple rate limit errors
    provider.client = MagicMock()
    provider.client.chat.completions.create.side_effect = [
        RateLimitError("Rate limit 1"),
        RateLimitError("Rate limit 2"),
        RateLimitError("Rate limit 3"),
        {"choices": [{"message": {"content": "Success after multiple retries"}}]},
    ]

    # Set up provider state
    provider.authenticated = True

    # Configure provider with retries
    provider.config["retry_count"] = 3
    provider.config["initial_retry_delay"] = 0.1
    provider.config["exponential_backoff_multiplier"] = 2

    # Call generate which should trigger exponential backoff
    with patch("time.sleep") as mock_sleep, patch(
        "random.uniform", return_value=0
    ):  # Remove jitter for predictable testing
        response = provider.generate(
            messages=[{"role": "user", "content": "Test message"}]
        )

    # Verify sleep was called with exponentially increasing durations
    assert mock_sleep.call_count == 3
    assert mock_sleep.call_args_list[0][0][0] == 0.1  # First retry: 0.1s
    assert mock_sleep.call_args_list[1][0][0] == 0.2  # Second retry: 0.1 * 2 = 0.2s
    assert mock_sleep.call_args_list[2][0][0] == 0.4  # Third retry: 0.2 * 2 = 0.4s

    # Verify final response
    assert "Success after multiple retries" in response
