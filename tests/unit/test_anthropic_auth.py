"""
Unit tests for Anthropic authentication flow.

This module tests the authentication mechanisms in the Anthropic provider,
including API key retrieval from different sources and error handling.
"""

import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import LLMAuthenticationError
from lluminary.models.providers.anthropic import AnthropicLLM


class MockAnthropicLLM(AnthropicLLM):
    """Mock implementation of AnthropicLLM for testing."""

    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """Mock implementation of abstract method."""
        pass


def test_auth_with_direct_api_key():
    """Test authentication with API key provided directly in constructor."""
    with patch("requests.head") as mock_head:
        # Configure mock to simulate successful API key validation
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        # Create LLM with api_key parameter
        llm = MockAnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-api-key")

        # Call auth method
        llm.auth()

        # Verify the API key was set from constructor parameter
        assert llm.config["api_key"] == "test-api-key"

        # Verify validation request was made
        mock_head.assert_called_once()
        # Verify correct headers were used
        call_args = mock_head.call_args[1]
        assert call_args["headers"]["x-api-key"] == "test-api-key"


def test_auth_with_environment_variable():
    """Test authentication with API key from environment variable."""
    with patch("requests.head") as mock_head, patch.dict(
        os.environ, {"ANTHROPIC_API_KEY": "env-api-key"}, clear=True
    ), patch("lluminary.utils.get_secret") as mock_get_secret:

        # Configure mock to simulate successful API key validation
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        # Configure get_secret to fail (so we fall back to env var)
        mock_get_secret.side_effect = Exception("Secret not found")

        # Create LLM without api_key parameter
        llm = MockAnthropicLLM("claude-3-5-sonnet-20241022")

        # Call auth method
        llm.auth()

        # Verify the API key was set from environment
        assert llm.config["api_key"] == "env-api-key"

        # Verify validation request was made with env var key
        call_args = mock_head.call_args[1]
        assert call_args["headers"]["x-api-key"] == "env-api-key"


def test_auth_with_secrets_manager():
    """Test authentication with API key from AWS Secrets Manager."""
    with patch("requests.head") as mock_head, patch(
        "lluminary.utils.get_secret"
    ) as mock_get_secret:

        # Configure mock to simulate successful API key validation
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        # Configure get_secret to return a secret
        mock_get_secret.return_value = {"api_key": "secret-api-key"}

        # Create LLM without api_key parameter
        llm = MockAnthropicLLM("claude-3-5-sonnet-20241022")

        # Call auth method
        llm.auth()

        # Verify the API key was set from secrets manager
        assert llm.config["api_key"] == "secret-api-key"

        # Verify get_secret was called correctly
        mock_get_secret.assert_called_once_with(
            "anthropic_api_key", required_keys=["api_key"]
        )

        # Verify validation request was made with secret key
        call_args = mock_head.call_args[1]
        assert call_args["headers"]["x-api-key"] == "secret-api-key"


def test_auth_failure_invalid_api_key():
    """Test authentication failure with invalid API key."""
    with patch("requests.head") as mock_head:
        # Configure mock to simulate authentication failure
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_head.return_value = mock_response

        # Create LLM with invalid API key
        llm = MockAnthropicLLM("claude-3-5-sonnet-20241022", api_key="invalid-key")

        # Call auth method and expect exception
        with pytest.raises(LLMAuthenticationError) as excinfo:
            llm.auth()

        # Verify error message mentions invalid API key
        assert "Invalid Anthropic API key" in str(excinfo.value)

        # Verify provider is identified in error
        assert excinfo.value.provider == "anthropic"


def test_auth_failure_no_api_key():
    """Test authentication failure when no API key is available."""
    with patch.dict(os.environ, {}, clear=True), patch(
        "lluminary.utils.get_secret"
    ) as mock_get_secret:

        # Configure get_secret to fail
        mock_get_secret.side_effect = Exception("Secret not found")

        # Create LLM without api_key
        llm = MockAnthropicLLM("claude-3-5-sonnet-20241022")

        # Call auth method and expect exception
        with pytest.raises(LLMAuthenticationError) as excinfo:
            llm.auth()

        # Verify error message mentions API key
        assert "Failed to get API key for Anthropic" in str(excinfo.value)

        # Verify tried sources are mentioned in details
        assert "tried_sources" in excinfo.value.details
        assert "AWS Secrets Manager" in excinfo.value.details["tried_sources"]
        assert "environment variables" in excinfo.value.details["tried_sources"]


def test_auth_failure_service_unavailable():
    """Test authentication when service is unavailable during validation."""
    with patch("requests.head") as mock_head:
        # Configure mock to simulate service unavailable
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_head.return_value = mock_response

        # Create LLM with API key
        llm = MockAnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-api-key")

        # Call auth method and expect exception
        with pytest.raises(Exception) as excinfo:
            llm.auth()

        # Verify error message mentions service issue
        assert "API returned status code 503" in str(excinfo.value)


def test_auth_failure_network_error():
    """Test authentication when network error occurs during validation."""
    with patch("requests.head") as mock_head:
        # Configure mock to simulate network error
        mock_head.side_effect = Exception("Network connection error")

        # Create LLM with API key
        llm = MockAnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-api-key")

        # Call auth method and expect exception
        with pytest.raises(Exception) as excinfo:
            llm.auth()

        # Verify error message mentions network issue
        assert "Network" in str(excinfo.value)


def test_auth_cached_credentials():
    """Test that auth() doesn't re-authenticate if key is already set."""
    with patch("requests.head") as mock_head:
        # Create LLM with API key
        llm = MockAnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-api-key")

        # Set up the config manually
        llm.config["api_key"] = "pre-existing-key"

        # Call auth method
        llm.auth()

        # Verify no validation request was made (reusing existing key)
        mock_head.assert_not_called()

        # Verify the API key was not changed
        assert llm.config["api_key"] == "pre-existing-key"
