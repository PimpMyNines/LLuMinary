"""
Comprehensive tests for OpenAI provider authentication mechanisms.

This module contains comprehensive tests for the authentication flow in the OpenAI provider,
covering direct API key usage, AWS Secrets Manager integration, environment variables,
organization ID handling, and various error scenarios.
"""

import os
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest
from lluminary.exceptions import LLMAuthenticationError
from lluminary.models.providers.openai import OpenAILLM


class MockOpenAILLM(OpenAILLM):
    """Mock implementation of OpenAILLM for testing."""

    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """Mock implementation of abstract method."""
        pass


def test_auth_with_direct_api_key():
    """Test authentication using directly provided API key."""
    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize the LLM with direct API key
        llm = MockOpenAILLM("gpt-4o", api_key="test-api-key")

        # Call auth method
        llm.auth()

        # Verify OpenAI client was initialized with provided API key
        mock_openai.assert_called_once_with(api_key="test-api-key", base_url=None)

        # Verify client was set correctly
        assert llm.client is mock_client

        # Verify config was updated
        assert llm.config["api_key"] == "test-api-key"


@patch("lluminary.models.providers.openai.get_secret")
def test_auth_with_aws_secrets(mock_get_secret):
    """Test authentication using AWS Secrets Manager."""
    # Mock the secret retrieval
    mock_get_secret.return_value = {"api_key": "secret-api-key"}

    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize LLM without direct API key
        llm = MockOpenAILLM("gpt-4o")

        # Call auth method - should use AWS Secrets
        llm.auth()

        # Verify secret was retrieved with correct parameters
        mock_get_secret.assert_called_once_with(
            "openai_api_key", required_keys=["api_key"]
        )

        # Verify OpenAI client was initialized with secret API key
        mock_openai.assert_called_once_with(api_key="secret-api-key", base_url=None)

        # Verify config was updated with retrieved key
        assert llm.config["api_key"] == "secret-api-key"


@patch("lluminary.models.providers.openai.get_secret")
def test_auth_with_env_variable_fallback(mock_get_secret):
    """Test authentication fallback to environment variable."""
    # Mock get_secret to fail
    mock_get_secret.side_effect = Exception("AWS Secrets Manager error")

    # Mock environment variable
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}), patch(
        "lluminary.models.providers.openai.OpenAI"
    ) as mock_openai:

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize LLM without direct API key
        llm = MockOpenAILLM("gpt-4o")

        # Call auth method - should fall back to env var
        llm.auth()

        # Verify secret retrieval was attempted
        mock_get_secret.assert_called_once()

        # Verify OpenAI client was initialized with env var API key
        mock_openai.assert_called_once_with(api_key="env-api-key", base_url=None)

        # Verify config was updated with env var key
        assert llm.config["api_key"] == "env-api-key"


@patch("lluminary.models.providers.openai.get_secret")
def test_auth_failure_no_key_source(mock_get_secret):
    """Test authentication failure when no API key source is available."""
    # Mock get_secret to fail
    mock_get_secret.side_effect = Exception("AWS Secrets Manager error")

    # Set up environment without API key
    with patch.dict(os.environ, {}, clear=True), patch(
        "lluminary.models.providers.openai.OpenAI"
    ) as mock_openai:

        # Initialize LLM without direct API key
        llm = MockOpenAILLM("gpt-4o")

        # Call auth method - should raise AuthenticationError
        with pytest.raises(LLMAuthenticationError) as excinfo:
            llm.auth()

        # Verify error message is appropriate
        assert "OpenAI authentication failed" in str(excinfo.value)
        assert "AWS Secrets Manager error" in str(excinfo.value)

        # Verify OpenAI client was not initialized
        mock_openai.assert_not_called()


def test_auth_with_organization_id():
    """Test authentication with organization ID."""
    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize LLM with organization ID
        llm = MockOpenAILLM(
            "gpt-4o", api_key="test-api-key", organization_id="org-123456"
        )

        # Call auth method
        llm.auth()

        # Verify OpenAI client was initialized with both api key and organization
        mock_openai.assert_called_once_with(
            api_key="test-api-key", organization="org-123456", base_url=None
        )

        # Verify config contains organization_id
        assert llm.config["organization_id"] == "org-123456"


@patch("lluminary.models.providers.openai.get_secret")
def test_auth_with_organization_from_secrets(mock_get_secret):
    """Test retrieving organization ID from AWS Secrets Manager."""
    # Mock the secret retrieval with organization ID included
    mock_get_secret.return_value = {
        "api_key": "secret-api-key",
        "organization_id": "org-from-secrets",
    }

    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize LLM without direct parameters
        llm = MockOpenAILLM("gpt-4o")

        # Call auth method
        llm.auth()

        # Verify secret was retrieved
        mock_get_secret.assert_called_once_with(
            "openai_api_key", required_keys=["api_key"]
        )

        # Verify OpenAI client was initialized with both parameters from secrets
        mock_openai.assert_called_once_with(
            api_key="secret-api-key", organization="org-from-secrets", base_url=None
        )

        # Verify config was updated with both values
        assert llm.config["api_key"] == "secret-api-key"
        assert llm.config["organization_id"] == "org-from-secrets"


def test_auth_with_custom_base_url():
    """Test authentication with custom API base URL."""
    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Custom API base URL
        custom_base = "https://custom-openai-api.example.com/v1"

        # Initialize the LLM with custom base URL
        llm = MockOpenAILLM("gpt-4o", api_key="test-api-key", api_base=custom_base)

        # Call auth method
        llm.auth()

        # Verify OpenAI client was initialized with custom base URL
        mock_openai.assert_called_once_with(
            api_key="test-api-key", base_url=custom_base
        )

        # Verify config and instance attribute were updated
        assert llm.config["api_base"] == custom_base
        assert llm.api_base == custom_base


@patch("lluminary.models.providers.openai.get_secret")
def test_auth_with_api_key_and_base_from_env(mock_get_secret):
    """Test retrieving both API key and base URL from environment variables."""
    # Mock get_secret to fail
    mock_get_secret.side_effect = Exception("AWS Secrets Manager error")

    # Mock environment variables for both API key and base URL
    env_vars = {
        "OPENAI_API_KEY": "env-api-key",
        "OPENAI_API_BASE": "https://env-openai-api.example.com/v1",
    }

    with patch.dict(os.environ, env_vars), patch(
        "lluminary.models.providers.openai.OpenAI"
    ) as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize LLM without direct parameters
        llm = MockOpenAILLM("gpt-4o")

        # Call auth method
        llm.auth()

        # Verify OpenAI client was initialized with env var values
        mock_openai.assert_called_once_with(
            api_key="env-api-key", base_url="https://env-openai-api.example.com/v1"
        )

        # Verify config was updated with env var values
        assert llm.config["api_key"] == "env-api-key"
        assert llm.api_base == "https://env-openai-api.example.com/v1"


def test_auth_with_timeout_setting():
    """Test authentication with custom timeout setting."""
    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize LLM with custom timeout
        custom_timeout = 120
        llm = MockOpenAILLM("gpt-4o", api_key="test-api-key", timeout=custom_timeout)

        # Call auth method
        llm.auth()

        # Verify OpenAI client was initialized with correct parameters
        mock_openai.assert_called_once_with(api_key="test-api-key", base_url=None)

        # Verify timeout was set correctly
        assert llm.timeout == custom_timeout
        assert llm.config["timeout"] == custom_timeout


def test_auth_invalid_api_key_format():
    """Test error handling for invalid API key format."""
    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Mock client initialization to fail with an invalid API key error
        error_message = "Invalid API key format. Expected format: 'sk-...'"
        mock_openai.side_effect = Exception(error_message)

        # Initialize LLM with malformed API key
        llm = MockOpenAILLM("gpt-4o", api_key="invalid-format-key")

        # Auth method should raise an AuthenticationError
        with pytest.raises(LLMAuthenticationError) as excinfo:
            llm.auth()

        # Verify error message
        assert "Invalid API key format" in str(excinfo.value)
        assert "OpenAI authentication failed" in str(excinfo.value)


def test_auth_client_error_handling():
    """Test handling of general client initialization errors."""
    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Mock a general OpenAI client error
        error_message = "Connection refused"
        mock_openai.side_effect = Exception(error_message)

        # Initialize LLM
        llm = MockOpenAILLM("gpt-4o", api_key="test-api-key")

        # Auth method should raise an AuthenticationError
        with pytest.raises(LLMAuthenticationError) as excinfo:
            llm.auth()

        # Verify error message
        assert "Connection refused" in str(excinfo.value)
        assert "OpenAI authentication failed" in str(excinfo.value)


def test_auth_flow_with_multiple_attempts():
    """Test authentication flow with multiple auth attempts."""
    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_openai.side_effect = [mock_client1, mock_client2]

        # First auth with initial key
        llm = MockOpenAILLM("gpt-4o", api_key="initial-key")
        llm.auth()

        # Verify first auth
        assert llm.config["api_key"] == "initial-key"
        assert mock_openai.call_count == 1
        assert mock_openai.call_args == call(api_key="initial-key", base_url=None)

        # Update API key and authenticate again
        llm.config["api_key"] = "updated-key"
        llm.auth()

        # Verify second auth used updated key
        assert mock_openai.call_count == 2
        assert mock_openai.call_args == call(api_key="updated-key", base_url=None)

        # Verify client was updated
        assert llm.client is mock_client2
