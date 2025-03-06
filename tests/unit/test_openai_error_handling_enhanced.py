"""
Enhanced tests for OpenAI provider error handling capabilities.

This module focuses on testing how the OpenAI provider handles various error scenarios,
including rate limits, timeout, invalid requests, and other API errors.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ContentError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
)
from lluminary.models.providers.openai import OpenAILLM


@pytest.fixture
def openai_llm():
    """Create a pre-configured OpenAI LLM instance."""
    # Mock the auth method to avoid actual API calls
    with patch.object(OpenAILLM, "auth"):
        # Create instance with test API key
        llm = OpenAILLM("gpt-4o", api_key="sk-test-key")

        # Set up mock client
        llm.client = MagicMock()

        yield llm


def test_rate_limit_error_mapping(openai_llm):
    """Test mapping of rate limit errors to the correct exception type."""
    # Create mock rate limit error
    rate_limit_error = MagicMock()
    rate_limit_error.__str__ = lambda _: "Rate limit exceeded"

    # Call the error mapping method
    mapped_error = openai_llm._map_openai_error(rate_limit_error)

    # Verify the mapping
    assert isinstance(mapped_error, RateLimitError)
    assert "rate limit" in str(mapped_error).lower()
    assert "openai" in str(mapped_error).lower()


def test_authentication_error_mapping(openai_llm):
    """Test mapping of authentication errors to the correct exception type."""
    # Create mock authentication error
    auth_error = MagicMock()
    auth_error.__str__ = lambda _: "Invalid API key"

    # Call the error mapping method
    mapped_error = openai_llm._map_openai_error(auth_error)

    # Verify the mapping
    assert isinstance(mapped_error, AuthenticationError)
    assert "api key" in str(mapped_error).lower()
    assert "openai" in str(mapped_error).lower()


def test_configuration_error_mapping(openai_llm):
    """Test mapping of configuration errors to the correct exception type."""
    # Create mock configuration error
    config_error = MagicMock()
    config_error.__str__ = lambda _: "Invalid parameter"

    # Call the error mapping method
    mapped_error = openai_llm._map_openai_error(config_error)

    # Verify the mapping
    assert isinstance(mapped_error, ConfigurationError)
    assert "parameter" in str(mapped_error).lower()
    assert "openai" in str(mapped_error).lower()


def test_content_error_mapping(openai_llm):
    """Test mapping of content errors to the correct exception type."""
    # Create mock content error
    content_error = MagicMock()
    content_error.__str__ = lambda _: "Content policy violation"

    # Call the error mapping method
    mapped_error = openai_llm._map_openai_error(content_error)

    # Verify the mapping
    assert isinstance(mapped_error, ContentError)
    assert "content" in str(mapped_error).lower()
    assert "openai" in str(mapped_error).lower()


def test_service_unavailable_error_mapping(openai_llm):
    """Test mapping of service unavailable errors to the correct exception type."""
    # Create mock service error
    service_error = MagicMock()
    service_error.__str__ = lambda _: "server_error: Service unavailable"

    # Call the error mapping method
    mapped_error = openai_llm._map_openai_error(service_error)

    # Verify the mapping
    assert isinstance(mapped_error, ServiceUnavailableError)
    assert "server" in str(mapped_error).lower()
    assert "openai" in str(mapped_error).lower()


def test_generic_error_mapping(openai_llm):
    """Test mapping of generic errors to the correct exception type."""
    # Create mock generic error
    generic_error = MagicMock()
    generic_error.__str__ = lambda _: "Unknown error occurred"

    # Call the error mapping method
    mapped_error = openai_llm._map_openai_error(generic_error)

    # Verify the mapping
    assert isinstance(mapped_error, ProviderError)
    assert "openai" in str(mapped_error).lower()


@patch("src.lluminary.models.providers.openai.time.sleep")
def test_call_with_retry_rate_limit(mock_sleep, openai_llm):
    """Test retry logic for rate limit errors."""
    # Create mock function that raises rate limit error on first call
    mock_func = MagicMock()
    mock_func.side_effect = [
        MagicMock(__str__=lambda _: "Rate limit exceeded"),  # First call fails
        "success",  # Second call succeeds
    ]

    # Override _map_openai_error to return a RateLimitError
    with patch.object(
        openai_llm,
        "_map_openai_error",
        return_value=RateLimitError(
            message="Rate limit exceeded", provider="openai", details={"retry_after": 1}
        ),
    ):
        # Call the function with retry
        result = openai_llm._call_with_retry(
            mock_func, max_retries=3, initial_backoff=0.1, retryable_errors=[Exception]
        )

        # Verify retry behavior
        assert result == "success"
        assert mock_func.call_count == 2
        assert mock_sleep.call_count == 1


@patch("src.lluminary.models.providers.openai.time.sleep")
def test_call_with_retry_max_retries(mock_sleep, openai_llm):
    """Test retry logic when max retries is exceeded."""
    # Create mock function that always raises an error
    mock_func = MagicMock()
    mock_func.side_effect = Exception("Persistent error")

    # Override _map_openai_error to return a ServiceUnavailableError
    with patch.object(
        openai_llm,
        "_map_openai_error",
        return_value=ServiceUnavailableError(
            message="Service unavailable", provider="openai", details={}
        ),
    ):
        # Call the function with retry - should raise after max retries
        with pytest.raises(ServiceUnavailableError) as excinfo:
            openai_llm._call_with_retry(
                mock_func,
                max_retries=2,
                initial_backoff=0.1,
                retryable_errors=[Exception],
            )

        # Verify retry behavior
        assert "Service unavailable" in str(excinfo.value)
        assert mock_func.call_count == 3  # Initial call + 2 retries
        assert mock_sleep.call_count == 2  # Sleep after first two failures


@patch("src.lluminary.models.providers.openai.time.sleep")
def test_call_with_retry_non_retryable_error(mock_sleep, openai_llm):
    """Test retry logic with non-retryable errors."""
    # Create mock function that raises a non-retryable error
    mock_func = MagicMock()
    mock_func.side_effect = ValueError("Non-retryable error")

    # Call the function with retry - should raise immediately for non-retryable error
    with pytest.raises(ValueError) as excinfo:
        openai_llm._call_with_retry(
            mock_func,
            max_retries=3,
            initial_backoff=0.1,
            retryable_errors=[RateLimitError],  # ValueError is not in this list
        )

    # Verify no retry was attempted
    assert "Non-retryable error" in str(excinfo.value)
    assert mock_func.call_count == 1
    assert mock_sleep.call_count == 0
