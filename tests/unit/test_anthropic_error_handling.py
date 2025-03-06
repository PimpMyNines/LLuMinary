"""
Unit tests for Anthropic error handling.

This module tests the error handling and error mapping functionality
in the Anthropic provider implementation.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import (
    AuthenticationError,
    ContentError,
    FormatError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
    ThinkingError,
    ToolError,
)
from lluminary.models.providers.anthropic import AnthropicLLM


@pytest.fixture
def anthropic_llm():
    """Fixture for Anthropic LLM instance."""
    with patch("anthropic.Anthropic") as mock_anthropic, patch(
        "requests.post"
    ) as mock_post:
        # Create the LLM instance with mock API key
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Ensure client is initialized
        llm.client = MagicMock()

        # Ensure config exists
        if not hasattr(llm, "config"):
            llm.config = {}

        # Add client to config as expected by implementation
        llm.config["client"] = llm.client
        llm.config["api_key"] = "test-key"

        yield llm


def test_map_anthropic_error_authentication(anthropic_llm):
    """Test mapping of authentication errors."""
    # Test API key error
    error = Exception("Invalid API key")
    response = MagicMock()
    response.status_code = 401

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error, response)

    # Verify mapping
    assert isinstance(mapped_error, AuthenticationError)
    assert "API key" in str(mapped_error)
    assert mapped_error.provider == "anthropic"

    # Test with just status code
    error = Exception("Generic error")
    response = MagicMock()
    response.status_code = 401

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error, response)

    # Verify mapping based on status code
    assert isinstance(mapped_error, AuthenticationError)


def test_map_anthropic_error_rate_limit(anthropic_llm):
    """Test mapping of rate limit errors."""
    # Test rate limit error with retry-after header
    error = Exception("Rate limit exceeded")
    response = MagicMock()
    response.status_code = 429
    response.headers = {"retry-after": "30"}

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error, response)

    # Verify mapping
    assert isinstance(mapped_error, RateLimitError)
    assert "rate limit" in str(mapped_error).lower()
    assert mapped_error.retry_after == 30

    # Test with keyword in error message
    error = Exception("Too many requests in short period")
    response = None

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error, response)

    # Verify mapping based on message
    assert isinstance(mapped_error, RateLimitError)


def test_map_anthropic_error_context_length(anthropic_llm):
    """Test mapping of context length errors."""
    # Test context length error
    error = Exception("Input is too long for model's context window")

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error)

    # Verify mapping
    assert isinstance(mapped_error, ProviderError)
    assert "context length" in str(mapped_error).lower()
    assert "context_window" in mapped_error.details


def test_map_anthropic_error_content_policy(anthropic_llm):
    """Test mapping of content policy violations."""
    # Test content policy error
    error = Exception("Content policy violation detected")

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error)

    # Verify mapping
    assert isinstance(mapped_error, ContentError)
    assert "content policy" in str(mapped_error).lower()


def test_map_anthropic_error_parameter_validation(anthropic_llm):
    """Test mapping of parameter validation errors."""
    # Test invalid parameter error
    error = Exception("Invalid parameter: temperature must be between 0 and 1")

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error)

    # Verify mapping
    assert isinstance(mapped_error, ProviderError)
    assert "parameter" in str(mapped_error).lower()


def test_map_anthropic_error_server_errors(anthropic_llm):
    """Test mapping of server errors."""
    # Test 5xx status code
    error = Exception("Server error")
    response = MagicMock()
    response.status_code = 503

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error, response)

    # Verify mapping
    assert isinstance(mapped_error, ServiceUnavailableError)
    assert "server error" in str(mapped_error).lower()
    assert mapped_error.details["status_code"] == 503


def test_map_anthropic_error_tool_errors(anthropic_llm):
    """Test mapping of tool/function related errors."""
    # Test tool error
    error = Exception("Error in function format: 'name' field is required")

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error)

    # Verify mapping
    assert isinstance(mapped_error, ToolError)
    assert "tool" in str(mapped_error).lower()


def test_map_anthropic_error_thinking_errors(anthropic_llm):
    """Test mapping of thinking related errors."""
    # Test thinking error
    error = Exception("Thinking budget exceeded")

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error)

    # Verify mapping
    assert isinstance(mapped_error, ThinkingError)
    assert "thinking" in str(mapped_error).lower()


def test_map_anthropic_error_default_fallback(anthropic_llm):
    """Test default error mapping fallback."""
    # Test generic error
    error = Exception("Some unexpected error")

    # Map the error
    mapped_error = anthropic_llm._map_anthropic_error(error)

    # Verify mapping to default
    assert isinstance(mapped_error, ProviderError)
    assert str(error) in str(mapped_error)


def test_error_propagation_in_generate(anthropic_llm):
    """Test error propagation in generate method."""
    with patch.object(anthropic_llm, "_raw_generate") as mock_raw_generate:
        # Configure mock to raise a specific error
        mock_raw_generate.side_effect = ContentError(
            message="Content policy violation", provider="anthropic"
        )

        # Call generate and expect the error to propagate
        with pytest.raises(ContentError) as excinfo:
            anthropic_llm.generate(
                event_id="test",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "Test message"}],
            )

        # Verify error details
        assert "Content policy violation" in str(excinfo.value)
        assert excinfo.value.provider == "anthropic"


def test_error_in_message_formatting(anthropic_llm):
    """Test error handling during message formatting."""
    # Create a message with invalid structure (missing message_type)
    invalid_message = [{"message": "Invalid message"}]

    # Call _format_messages_for_model and expect error
    with pytest.raises(FormatError) as excinfo:
        anthropic_llm._format_messages_for_model(invalid_message)

    # Verify error details
    assert "Missing required field" in str(excinfo.value)
    assert "message_type" in str(excinfo.value)
    assert excinfo.value.provider == "anthropic"


def test_error_in_tool_formatting(anthropic_llm):
    """Test error handling during tool formatting."""
    # Create invalid tool result message
    invalid_tool_message = [
        {"message_type": "tool", "tool_result": {"missing_required_field": "value"}}
    ]

    # Call _format_messages_for_model and expect error
    with pytest.raises(FormatError) as excinfo:
        anthropic_llm._format_messages_for_model(invalid_tool_message)

    # Verify error details
    assert "Invalid 'tool_result'" in str(excinfo.value)
    assert "tool_id" in str(excinfo.value)


def test_error_in_thinking_formatting(anthropic_llm):
    """Test error handling during thinking formatting."""
    # Create invalid thinking message
    invalid_thinking_message = [
        {
            "message_type": "human",
            "message": "Test",
            "thinking": "Not a proper thinking structure",  # Should be a dict
        }
    ]

    # Call _format_messages_for_model and expect error
    with pytest.raises(FormatError) as excinfo:
        anthropic_llm._format_messages_for_model(invalid_thinking_message)

    # Verify error details
    assert "Invalid 'thinking' structure" in str(excinfo.value)


def test_retry_logic_with_rate_limit(anthropic_llm):
    """Test retry logic when rate limit is encountered."""
    with patch("requests.request") as mock_request, patch("time.sleep") as mock_sleep:

        # Configure request to fail with rate limit on first call, then succeed
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {"retry-after": "2"}

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"success": True}

        # First call returns rate limit, second call succeeds
        mock_request.side_effect = [rate_limit_response, success_response]

        # Call _call_with_retry
        result = anthropic_llm._call_with_retry(
            method="POST", url="https://api.anthropic.com/v1/test", max_retries=3
        )

        # Verify retry behavior
        assert mock_request.call_count == 2
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == 2  # Should use retry-after value

        # Verify we got the successful response
        assert result == success_response


def test_retry_logic_with_server_error(anthropic_llm):
    """Test retry logic when server error is encountered."""
    with patch("requests.request") as mock_request, patch("time.sleep") as mock_sleep:

        # Configure request to fail with server error on first two calls, then succeed
        server_error_response = MagicMock()
        server_error_response.status_code = 503

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"success": True}

        # First two calls return server error, third call succeeds
        mock_request.side_effect = [
            server_error_response,
            server_error_response,
            success_response,
        ]

        # Call _call_with_retry
        result = anthropic_llm._call_with_retry(
            method="POST",
            url="https://api.anthropic.com/v1/test",
            max_retries=3,
            initial_backoff=0.1,  # Small value for testing
        )

        # Verify retry behavior
        assert mock_request.call_count == 3
        assert mock_sleep.call_count == 2

        # Verify we got the successful response
        assert result == success_response


def test_retry_logic_exhaustion(anthropic_llm):
    """Test retry logic when all retries are exhausted."""
    with patch("requests.request") as mock_request, patch("time.sleep") as mock_sleep:

        # Configure request to always fail with server error
        server_error_response = MagicMock()
        server_error_response.status_code = 503
        server_error_response.json.return_value = {
            "error": {"message": "Service unavailable"}
        }

        # All calls return server error
        mock_request.return_value = server_error_response

        # Call _call_with_retry and expect exception
        with pytest.raises(ServiceUnavailableError) as excinfo:
            anthropic_llm._call_with_retry(
                method="POST",
                url="https://api.anthropic.com/v1/test",
                max_retries=2,
                initial_backoff=0.1,  # Small value for testing
            )

        # Verify retry behavior
        assert mock_request.call_count == 3  # Initial + 2 retries
        assert mock_sleep.call_count == 2

        # Verify error message
        assert "API call failed after 2 retries" in str(excinfo.value)
        assert excinfo.value.provider == "anthropic"


def test_network_error_retry(anthropic_llm):
    """Test retry logic for network errors."""
    with patch("requests.request") as mock_request, patch("time.sleep") as mock_sleep:

        # Configure request to fail with network error, then succeed
        network_error = Exception("Connection timeout")

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"success": True}

        # First call raises network error, second call succeeds
        mock_request.side_effect = [network_error, success_response]

        # Call _call_with_retry
        result = anthropic_llm._call_with_retry(
            method="POST",
            url="https://api.anthropic.com/v1/test",
            max_retries=2,
            initial_backoff=0.1,
        )

        # Verify retry behavior
        assert mock_request.call_count == 2
        assert mock_sleep.call_count == 1

        # Verify we got the successful response
        assert result == success_response


def test_stream_error_handling(anthropic_llm):
    """Test error handling during streaming."""
    # Import anthropic here to avoid import error when anthropic package is not installed
    anthropic = pytest.importorskip("anthropic")

    with patch.object(anthropic_llm, "client") as mock_client:
        # Configure mock to raise API error during streaming
        api_error = anthropic.APIError(
            message="Rate limit exceeded", http_status=429, header={"retry-after": "30"}
        )
        api_error.status_code = 429
        mock_client.messages.create.side_effect = api_error

        # Call stream_generate and expect rate limit error
        with pytest.raises(RateLimitError) as excinfo:
            for _ in anthropic_llm.stream_generate(
                event_id="test",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "Test"}],
            ):
                pass

        # Verify error details
        assert "rate limit" in str(excinfo.value).lower()
        assert excinfo.value.provider == "anthropic"
