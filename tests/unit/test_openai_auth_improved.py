"""
Improved tests for OpenAI provider authentication.

This module provides focused tests for the authentication mechanisms in the OpenAI provider,
with proper environment variable isolation between tests.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import AuthenticationError
from lluminary.models.providers.openai import OpenAILLM


@pytest.fixture(autouse=True)
def clean_env_vars():
    """Clean up environment variables before and after each test."""
    # Save original environment variables
    orig_env = os.environ.copy()

    # Clear relevant env vars for isolation
    for key in ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_ORGANIZATION"]:
        if key in os.environ:
            del os.environ[key]

    # Run the test
    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(orig_env)


def test_auth_with_direct_api_key():
    """Test authentication with a direct API key."""
    # Mock the OpenAI client class
    with patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Create a mock client instance
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock models.list to verify credentials
        mock_client.models = MagicMock()
        mock_client.models.list = MagicMock()

        # Create LLM with direct API key
        llm = OpenAILLM("gpt-4o", api_key="test-api-key")

        # Call auth method
        llm.auth()

        # Verify OpenAI client was initialized with correct parameters
        mock_openai.assert_called_once_with(api_key="test-api-key", base_url=None)

        # Verify API key was stored in config
        assert llm.config["api_key"] == "test-api-key"


def test_auth_with_aws_secrets():
    """Test authentication using AWS Secrets Manager."""
    # Create mock secret
    mock_secret = {"api_key": "secret-api-key"}

    # Mock get_secret and OpenAI client
    with patch(
        "src.lluminary.models.providers.openai.get_secret", return_value=mock_secret
    ), patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:

        # Create a mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock models.list to verify credentials
        mock_client.models = MagicMock()
        mock_client.models.list = MagicMock()

        # Create LLM without direct API key
        llm = OpenAILLM("gpt-4o")

        # Call auth method
        llm.auth()

        # Verify get_secret was called with correct parameters
        assert llm.config["api_key"] == "secret-api-key"

        # Verify OpenAI client was initialized with secret API key
        mock_openai.assert_called_once_with(api_key="secret-api-key", base_url=None)


def test_auth_with_environment_variables():
    """Test authentication using environment variables when AWS Secrets Manager fails."""
    # Set environment variable
    os.environ["OPENAI_API_KEY"] = "env-api-key"

    # Mock get_secret to fail and OpenAI client
    with patch(
        "src.lluminary.models.providers.openai.get_secret",
        side_effect=Exception("AWS Secrets Manager error"),
    ), patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:

        # Create a mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock models.list to verify credentials
        mock_client.models = MagicMock()
        mock_client.models.list = MagicMock()

        # Create LLM without direct API key
        llm = OpenAILLM("gpt-4o")

        # Call auth method
        llm.auth()

        # Verify API key from environment was used
        assert llm.config["api_key"] == "env-api-key"

        # Verify OpenAI client was initialized with env API key
        mock_openai.assert_called_once_with(api_key="env-api-key", base_url=None)


def test_auth_failure_no_credentials():
    """Test authentication failure when no credentials are available."""
    # Mock get_secret to fail
    with patch(
        "src.lluminary.models.providers.openai.get_secret",
        side_effect=Exception("AWS Secrets Manager error"),
    ), patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:

        # Create LLM without direct API key and no env vars
        llm = OpenAILLM("gpt-4o")

        # Call auth method, should raise exception
        with pytest.raises(AuthenticationError) as excinfo:
            llm.auth()

        # Verify appropriate error message
        assert "Failed to get API key for OpenAI" in str(excinfo.value)

        # Verify OpenAI client was not initialized
        mock_openai.assert_not_called()


def test_auth_with_organization_id():
    """Test authentication with organization ID."""
    # Mock OpenAI client
    with patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Create a mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock models.list to verify credentials
        mock_client.models = MagicMock()
        mock_client.models.list = MagicMock()

        # Create LLM with API key and organization ID
        llm = OpenAILLM("gpt-4o", api_key="test-api-key", organization_id="org-123456")

        # Call auth method
        llm.auth()

        # Verify OpenAI client was initialized with both parameters
        mock_openai.assert_called_once_with(
            api_key="test-api-key", organization="org-123456", base_url=None
        )

        # Verify both values were stored in config
        assert llm.config["api_key"] == "test-api-key"
        assert llm.config["organization_id"] == "org-123456"
