"""
Tests for OpenAI provider authentication functionality.

This module focuses specifically on testing the authentication
mechanisms of the OpenAI provider in isolation.
"""

import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import LLMAuthenticationError
from lluminary.models.providers.openai import OpenAILLM


class MockOpenAILLM(OpenAILLM):
    """Mock implementation of OpenAILLM for testing."""

    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """Mock implementation of abstract method."""
        pass


def test_auth_with_aws_secrets():
    """Test authentication using AWS Secrets Manager."""
    # Create mock for get_secret function
    mock_secret = {"api_key": "test-secret-key"}

    with patch(
        "lluminary.models.providers.openai.get_secret", return_value=mock_secret
    ) as mock_get_secret, patch(
        "lluminary.models.providers.openai.OpenAI"
    ) as mock_openai_client:

        # Create instance and call auth
        openai_llm = MockOpenAILLM("gpt-4o")
        openai_llm.auth()

        # Verify get_secret was called with correct parameters
        mock_get_secret.assert_called_once_with(
            "openai_api_key", required_keys=["api_key"]
        )

        # Verify API key was properly stored
        assert openai_llm.config["api_key"] == "test-secret-key"

        # Verify OpenAI client was initialized with correct API key
        mock_openai_client.assert_called_once_with(
            api_key="test-secret-key", base_url=None
        )


def test_auth_with_environment_variables():
    """Test authentication using environment variables instead of AWS Secrets Manager."""
    # Mock environment variable and make get_secret raise an exception
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}), patch(
        "lluminary.models.providers.openai.get_secret",
        side_effect=Exception("Secret not found"),
    ), patch("lluminary.models.providers.openai.OpenAI") as mock_openai_client:

        # Create instance and call auth - should fall back to env var
        openai_llm = MockOpenAILLM("gpt-4o")
        openai_llm.auth()

        # Verify API key was properly stored from environment variable
        assert openai_llm.config["api_key"] == "env-api-key"

        # Verify OpenAI client was initialized with correct API key
        mock_openai_client.assert_called_once_with(api_key="env-api-key", base_url=None)


def test_auth_with_config_api_key():
    """Test authentication using API key provided in config."""
    with patch(
        "lluminary.models.providers.openai.get_secret"
    ) as mock_get_secret, patch(
        "lluminary.models.providers.openai.OpenAI"
    ) as mock_openai_client:

        # Create instance with API key in config
        openai_llm = MockOpenAILLM("gpt-4o", config={"api_key": "config-api-key"})
        openai_llm.auth()

        # Verify get_secret was not called - should use config directly
        mock_get_secret.assert_not_called()

        # Verify OpenAI client was initialized with config API key
        mock_openai_client.assert_called_once_with(
            api_key="config-api-key", base_url=None
        )


def test_auth_precedence_order():
    """Test authentication precedence: config > AWS secrets > environment."""
    # Set up all three authentication methods to verify precedence
    mock_secret = {"api_key": "secret-api-key"}

    # Test 1: All methods available - should use config
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}), patch(
        "lluminary.models.providers.openai.get_secret", return_value=mock_secret
    ), patch("lluminary.models.providers.openai.OpenAI") as mock_openai_client:

        openai_llm = MockOpenAILLM("gpt-4o", config={"api_key": "config-api-key"})
        openai_llm.auth()

        # Should use config API key
        mock_openai_client.assert_called_once_with(
            api_key="config-api-key", base_url=None
        )
        mock_openai_client.reset_mock()

    # Test 2: No config, should use AWS secrets over env var
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}), patch(
        "lluminary.models.providers.openai.get_secret", return_value=mock_secret
    ), patch("lluminary.models.providers.openai.OpenAI") as mock_openai_client:

        openai_llm = MockOpenAILLM("gpt-4o")  # No config API key
        openai_llm.auth()

        # Should use AWS secrets API key
        mock_openai_client.assert_called_once_with(
            api_key="secret-api-key", base_url=None
        )
        mock_openai_client.reset_mock()

    # Test 3: No config, AWS fails, should use env var
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}), patch(
        "lluminary.models.providers.openai.get_secret",
        side_effect=Exception("Secret not found"),
    ), patch("lluminary.models.providers.openai.OpenAI") as mock_openai_client:

        openai_llm = MockOpenAILLM("gpt-4o")  # No config API key
        openai_llm.auth()

        # Should use environment API key
        mock_openai_client.assert_called_once_with(api_key="env-api-key", base_url=None)


def test_auth_with_organization():
    """Test authentication with organization ID."""
    mock_secret = {"api_key": "test-secret-key"}

    # Test with organization in config
    with patch(
        "lluminary.models.providers.openai.get_secret", return_value=mock_secret
    ), patch("lluminary.models.providers.openai.OpenAI") as mock_openai_client:

        openai_llm = MockOpenAILLM("gpt-4o", config={"organization": "test-org"})
        openai_llm.auth()

        # Verify OpenAI client was initialized with organization
        mock_openai_client.assert_called_once_with(
            api_key="test-secret-key", organization="test-org", base_url=None
        )


def test_auth_with_base_url():
    """Test authentication with custom base URL."""
    mock_secret = {"api_key": "test-secret-key"}

    # Test with base_url in config
    with patch(
        "lluminary.models.providers.openai.get_secret", return_value=mock_secret
    ), patch("lluminary.models.providers.openai.OpenAI") as mock_openai_client:

        openai_llm = MockOpenAILLM(
            "gpt-4o", config={"base_url": "https://custom-openai-api.example.com"}
        )
        openai_llm.auth()

        # Verify OpenAI client was initialized with base_url
        mock_openai_client.assert_called_once_with(
            api_key="test-secret-key", base_url="https://custom-openai-api.example.com"
        )


def test_auth_failure_handling():
    """Test handling of authentication failures."""
    # Test different error scenarios
    error_messages = [
        "Secret not found",
        "Access denied",
        "Invalid parameters",
        "Network error",
    ]

    for error_msg in error_messages:
        # Mock get_secret to raise exception and ensure no env var fallback
        with patch(
            "lluminary.models.providers.openai.get_secret",
            side_effect=Exception(error_msg),
        ), patch.dict(os.environ, {}, clear=True), patch(
            "lluminary.models.providers.openai.OpenAI"
        ):

            # Create instance
            openai_llm = MockOpenAILLM("gpt-4o")

            # Call auth and expect exception
            with pytest.raises(Exception) as excinfo:
                openai_llm.auth()

            # Verify error message - check that it contains both the original error message
            # and the standard "OpenAI authentication failed" prefix
            assert error_msg in str(excinfo.value)
            assert "OpenAI authentication failed" in str(excinfo.value)


def test_auth_credential_verification():
    """Test that credentials are verified with a test API call."""
    with patch(
        "lluminary.models.providers.openai.get_secret",
        return_value={"api_key": "test-secret-key"},
    ), patch("lluminary.models.providers.openai.OpenAI") as mock_openai_client:

        # Set up mock client to return models list
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_models.list.return_value = {"data": [{"id": "gpt-4o"}]}
        mock_client.models = mock_models
        mock_openai_client.return_value = mock_client

        # Create instance and call auth
        openai_llm = MockOpenAILLM("gpt-4o")
        openai_llm.auth()

        # Verify models.list was called to test credentials
        mock_models.list.assert_called_once()


def test_auth_credential_verification_failure():
    """Test handling of credential verification failures."""
    with patch(
        "lluminary.models.providers.openai.get_secret",
        return_value={"api_key": "invalid-api-key"},
    ), patch("lluminary.models.providers.openai.OpenAI") as mock_openai_client:

        # Set up mock client to raise an exception during verification
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_models.list.side_effect = Exception("Invalid API key")
        mock_client.models = mock_models
        mock_openai_client.return_value = mock_client

        # Create instance
        openai_llm = MockOpenAILLM("gpt-4o")

        # Call auth and expect exception
        with pytest.raises(LLMAuthenticationError) as excinfo:
            openai_llm.auth()

        # Verify error message
        assert "Invalid API key" in str(excinfo.value)
        assert "OpenAI authentication failed" in str(excinfo.value)
