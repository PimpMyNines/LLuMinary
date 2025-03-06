"""
Unit tests for Anthropic provider timeout handling.

This module tests the timeout handling capabilities of the Anthropic provider,
including request timeouts, retries, and error propagation.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from lluminary.exceptions import ProviderError, ServiceUnavailableError
from lluminary.models.providers.anthropic import AnthropicLLM


@pytest.fixture
def anthropic_llm():
    """Fixture for Anthropic LLM instance."""
    with patch("anthropic.Anthropic"):
        # Create the LLM instance with mock API key
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Ensure config exists
        llm.config = {"api_key": "test-key"}

        yield llm


def test_request_timeout_handling(anthropic_llm):
    """Test handling of request timeouts."""
    with patch("requests.request") as mock_request:
        # Configure request to time out
        mock_request.side_effect = requests.exceptions.Timeout("Request timed out")

        # Call _call_with_retry and expect exception
        with pytest.raises(ProviderError) as excinfo:
            anthropic_llm._call_with_retry(
                method="POST",
                url="https://api.anthropic.com/v1/test",
                max_retries=1,
                initial_backoff=0.1,
                timeout=5,  # 5 seconds timeout
            )

        # Verify timeout details in error
        assert "timed out" in str(excinfo.value).lower()
        assert excinfo.value.provider == "anthropic"

        # Verify request was called with timeout parameter
        call_args = mock_request.call_args[1]
        assert call_args.get("timeout") == 5


def test_timeout_with_retries(anthropic_llm):
    """Test timeout handling with retry logic."""
    with patch("requests.request") as mock_request, patch("time.sleep") as mock_sleep:

        # Configure first request to time out, second to succeed
        timeout_error = requests.exceptions.Timeout("Request timed out")

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"success": True}

        # First call times out, second succeeds
        mock_request.side_effect = [timeout_error, success_response]

        # Call _call_with_retry with timeout parameter
        result = anthropic_llm._call_with_retry(
            method="POST",
            url="https://api.anthropic.com/v1/test",
            max_retries=2,
            initial_backoff=0.1,
            timeout=5,
        )

        # Verify retry behavior
        assert mock_request.call_count == 2
        assert mock_sleep.call_count == 1

        # Verify each request included timeout parameter
        for call in mock_request.call_args_list:
            assert call[1].get("timeout") == 5

        # Verify we got the successful response
        assert result == success_response


def test_generate_with_timeout(anthropic_llm):
    """Test timeout handling in generate method."""
    with patch.object(anthropic_llm, "_raw_generate") as mock_raw_generate:
        # Configure mock to raise a timeout error
        mock_raw_generate.side_effect = ServiceUnavailableError(
            message="Request timed out after 60 seconds",
            provider="anthropic",
            details={"timeout": 60},
        )

        # Call generate with timeout parameter
        with pytest.raises(ServiceUnavailableError) as excinfo:
            anthropic_llm.generate(
                event_id="test",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "Test message"}],
                timeout=60,
            )

        # Verify error propagation
        assert "timed out" in str(excinfo.value)
        assert excinfo.value.provider == "anthropic"
        assert excinfo.value.details.get("timeout") == 60

        # Verify _raw_generate was called with timeout parameter
        call_kwargs = mock_raw_generate.call_args[1]
        assert call_kwargs.get("timeout") == 60


def test_stream_with_timeout(anthropic_llm):
    """Test timeout handling in stream_generate method."""
    # Import anthropic to create API error
    anthropic = pytest.importorskip("anthropic")

    with patch.object(anthropic_llm, "client") as mock_client:
        # Configure mock to raise Timeout error
        timeout_error = requests.exceptions.Timeout("Request timed out after 30s")
        mock_client.messages.create.side_effect = timeout_error

        # Call stream_generate with timeout and expect ProviderError
        with pytest.raises(ProviderError) as excinfo:
            for _ in anthropic_llm.stream_generate(
                event_id="test",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "Test message"}],
                timeout=30,
            ):
                pass

        # Verify error details
        assert "timed out" in str(excinfo.value).lower()
        assert "30s" in str(excinfo.value)
        assert excinfo.value.provider == "anthropic"

        # Verify client.messages.create was called with timeout parameter
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs.get("timeout") == 30


def test_request_connection_error(anthropic_llm):
    """Test handling of request connection errors."""
    with patch("requests.request") as mock_request:
        # Configure request to raise connection error
        mock_request.side_effect = requests.exceptions.ConnectionError(
            "Failed to establish a connection"
        )

        # Call _call_with_retry and expect exception
        with pytest.raises(ProviderError) as excinfo:
            anthropic_llm._call_with_retry(
                method="POST",
                url="https://api.anthropic.com/v1/test",
                max_retries=1,
                initial_backoff=0.1,
            )

        # Verify error details
        assert "connection" in str(excinfo.value).lower()
        assert excinfo.value.provider == "anthropic"


def test_timeout_parameter_propagation():
    """Test timeout parameter propagation from high-level methods to requests."""
    with patch("requests.post") as mock_post, patch("anthropic.Anthropic"):

        # Configure mock to return success response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "response"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create LLM instance
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Call generate with timeout parameter
        llm.generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "Test message"}],
            timeout=45,
        )

        # Verify request was made with timeout parameter
        assert mock_post.call_count == 1
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs.get("timeout") == 45


def test_default_timeouts():
    """Test default timeout values are used when not specified."""
    with patch("requests.post") as mock_post, patch("anthropic.Anthropic"):

        # Configure mock to return success response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "response"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Create LLM instance
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Call generate without timeout parameter
        llm.generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "Test message"}],
        )

        # Verify request was made with default timeout
        assert mock_post.call_count == 1
        call_kwargs = mock_post.call_args[1]
        assert (
            call_kwargs.get("timeout") is not None
        )  # Some default timeout should be set


def test_timeout_during_auth(anthropic_llm):
    """Test timeout handling during authentication."""
    with patch("requests.head") as mock_head:
        # Configure head request to time out
        mock_head.side_effect = requests.exceptions.Timeout("Auth request timed out")

        # Reset API key to force re-authentication
        anthropic_llm.config = {"api_key": "test-key"}

        # Call auth method and expect exception
        with pytest.raises(ProviderError) as excinfo:
            anthropic_llm.auth(timeout=10)

        # Verify timeout details in error
        assert "timed out" in str(excinfo.value).lower()
        assert excinfo.value.provider == "anthropic"

        # Verify request was called with timeout parameter
        call_kwargs = mock_head.call_args[1]
        assert call_kwargs.get("timeout") == 10


def test_exponential_backoff_increases(anthropic_llm):
    """Test that backoff times increase exponentially with each retry."""
    with patch("requests.request") as mock_request, patch("time.sleep") as mock_sleep:

        # Configure all requests to time out
        timeout_error = requests.exceptions.Timeout("Request timed out")
        mock_request.side_effect = [timeout_error, timeout_error, timeout_error]

        # Call _call_with_retry and let it exhaust retries
        with pytest.raises(ProviderError):
            anthropic_llm._call_with_retry(
                method="POST",
                url="https://api.anthropic.com/v1/test",
                max_retries=2,
                initial_backoff=1.0,
                backoff_factor=2.0,
            )

        # Verify sleep was called with increasing durations
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 1.0  # First retry: initial_backoff
        assert (
            mock_sleep.call_args_list[1][0][0] == 2.0
        )  # Second retry: initial_backoff * backoff_factor
