"""
Basic tests for OpenAI provider authentication functionality.

This module focuses on testing the authentication mechanisms of the OpenAI
provider, including API key handling, AWS Secrets Manager integration,
and error handling.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from lluminary.exceptions import AuthenticationError
from lluminary.models.providers.openai import OpenAILLM


def test_auth_with_direct_api_key():
    """Test authentication with directly provided API key."""
    # Mock the OpenAI client
    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Mock the client instance and its methods
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        client_instance.models = MagicMock()
        client_instance.models.list = MagicMock()

        # Create LLM instance with direct API key
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Call auth method
        llm.auth()

        # Verify OpenAI client was initialized with the provided key
        mock_openai.assert_called_once_with(api_key="test-key", base_url=None)

        # Verify client was set correctly
        assert llm.client is client_instance


def test_auth_with_environment_variable():
    """Test authentication using environment variable."""
    # Save original environment
    original_env = os.environ.copy()

    try:
        # Set environment variable
        os.environ["OPENAI_API_KEY"] = "env-api-key"

        # Mock OpenAI client and get_secret function
        with patch(
            "src.lluminary.models.providers.openai.OpenAI"
        ) as mock_openai, patch(
            "src.lluminary.models.providers.openai.get_secret",
            side_effect=Exception("AWS Secrets error"),
        ):

            # Mock client instance
            client_instance = MagicMock()
            mock_openai.return_value = client_instance
            client_instance.models = MagicMock()
            client_instance.models.list = MagicMock()

            # Create LLM instance without API key
            llm = OpenAILLM("gpt-4o")

            # Call auth method - should use environment variable
            llm.auth()

            # Verify key was obtained from environment
            assert llm.config["api_key"] == "env-api-key"

            # Verify OpenAI client was initialized with env key
            mock_openai.assert_called_once_with(api_key="env-api-key", base_url=None)
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


def test_auth_failure_no_key():
    """Test authentication failure when no API key is available."""
    # Save original environment
    original_env = os.environ.copy()

    try:
        # Remove relevant environment variables
        for key in ["OPENAI_API_KEY"]:
            if key in os.environ:
                del os.environ[key]

        # Mock get_secret to fail
        with patch(
            "src.lluminary.models.providers.openai.get_secret",
            side_effect=Exception("AWS Secrets error"),
        ):

            # Create LLM instance without API key
            llm = OpenAILLM("gpt-4o")

            # Call auth method - should raise AuthenticationError
            with pytest.raises(AuthenticationError) as excinfo:
                llm.auth()

            # Verify error contains appropriate context
            assert "OpenAI authentication failed" in str(excinfo.value)
            assert "AWS Secrets error" in str(excinfo.value)
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


def test_auth_with_custom_base_url():
    """Test authentication with custom API base URL."""
    # Mock OpenAI client
    with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Mock client instance
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        client_instance.models = MagicMock()
        client_instance.models.list = MagicMock()

        # Custom base URL
        custom_url = "https://custom-openai-api.example.com/v1"

        # Create LLM instance with API key and custom base URL
        llm = OpenAILLM("gpt-4o", api_key="test-key", api_base=custom_url)

        # Call auth method
        llm.auth()

        # Verify base URL was stored correctly
        assert llm.api_base == custom_url
        assert llm.config["api_base"] == custom_url

        # Verify OpenAI client was initialized with custom base URL
        mock_openai.assert_called_once_with(api_key="test-key", base_url=custom_url)
