"""
Enhanced tests for OpenAI provider authentication functionality.

This module provides comprehensive testing of authentication mechanisms
for the OpenAI provider, focusing on proper environment variables use,
API key verification, and error handling.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import AuthenticationError
from lluminary.models.providers.openai import OpenAILLM


def test_auth_with_direct_api_key():
    """Test authentication with directly provided API key."""
    # Mock the OpenAI client
    with patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Mock the models.list method to verify auth
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        client_instance.models = MagicMock()
        client_instance.models.list = MagicMock()

        # Initialize with direct API key
        llm = OpenAILLM("gpt-4o", api_key="test-api-key")

        # Call auth method
        llm.auth()

        # Verify OpenAI client was initialized with provided API key
        mock_openai.assert_called_once_with(api_key="test-api-key", base_url=None)

        # Verify models.list was called to validate the key
        client_instance.models.list.assert_called_once_with(limit=1)

        # Verify client was set on the instance
        assert llm.client is client_instance


def test_auth_with_aws_secrets():
    """Test authentication using AWS Secrets Manager."""
    # Create mock for get_secret function
    mock_secret = {"api_key": "test-secret-key"}

    # Mock the OpenAI client and get_secret
    with patch(
        "src.lluminary.models.providers.openai.get_secret", return_value=mock_secret
    ), patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:

        # Mock the client instance
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        client_instance.models = MagicMock()
        client_instance.models.list = MagicMock()

        # Create instance and call auth
        llm = OpenAILLM("gpt-4o")
        llm.auth()

        # Verify get_secret was called with correct params
        from lluminary.models.providers.openai import get_secret

        get_secret.assert_called_once_with("openai_api_key", required_keys=["api_key"])

        # Verify API key was properly stored
        assert llm.config["api_key"] == "test-secret-key"

        # Verify OpenAI client was initialized with correct API key
        mock_openai.assert_called_once_with(api_key="test-secret-key", base_url=None)


def test_auth_with_environment_variable():
    """Test authentication using environment variables."""
    # Save original environment
    original_env = os.environ.copy()

    try:
        # Set test environment variable
        os.environ["OPENAI_API_KEY"] = "env-api-key"

        # Mock AWS Secrets to fail and OpenAI client
        with patch(
            "src.lluminary.models.providers.openai.get_secret",
            side_effect=Exception("AWS Secrets error"),
        ), patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:

            # Mock client instance
            client_instance = MagicMock()
            mock_openai.return_value = client_instance
            client_instance.models = MagicMock()
            client_instance.models.list = MagicMock()

            # Create instance and call auth
            llm = OpenAILLM("gpt-4o")
            llm.auth()

            # Verify API key from environment was used
            assert llm.config["api_key"] == "env-api-key"

            # Verify OpenAI client was initialized correctly
            mock_openai.assert_called_once_with(api_key="env-api-key", base_url=None)
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


def test_auth_with_org_id():
    """Test authentication with organization ID."""
    # Mock OpenAI client
    with patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Mock client instance
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        client_instance.models = MagicMock()
        client_instance.models.list = MagicMock()

        # Initialize with API key and organization ID
        llm = OpenAILLM("gpt-4o", api_key="test-key", organization_id="org-123456")
        llm.auth()

        # Verify OpenAI client initialized with both parameters
        mock_openai.assert_called_once_with(
            api_key="test-key", organization="org-123456", base_url=None
        )


def test_auth_error_no_key():
    """Test error when no API key is available."""
    # Save original environment
    original_env = os.environ.copy()

    try:
        # Clear environment variables
        os.environ.pop("OPENAI_API_KEY", None)

        # Mock AWS Secrets to fail
        with patch(
            "src.lluminary.models.providers.openai.get_secret",
            side_effect=Exception("AWS Secrets error"),
        ):

            # Create instance
            llm = OpenAILLM("gpt-4o")

            # Auth should fail with no available keys
            with pytest.raises(AuthenticationError) as excinfo:
                llm.auth()

            # Check error message
            assert "OpenAI authentication failed" in str(excinfo.value)
            assert "AWS Secrets error" in str(excinfo.value)
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


def test_auth_error_validation_fail():
    """Test error when API key validation fails."""
    # Mock OpenAI client initialization to fail during models.list
    with patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Create client that will fail validation
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        client_instance.models = MagicMock()
        client_instance.models.list = MagicMock(
            side_effect=Exception("Invalid API key format")
        )

        # Create instance with API key
        llm = OpenAILLM("gpt-4o", api_key="invalid-key")

        # Auth should fail with validation error
        with pytest.raises(AuthenticationError) as excinfo:
            llm.auth()

        # Check error message
        assert "Invalid API key format" in str(excinfo.value)


def test_auth_with_custom_base_url():
    """Test authentication with custom API base URL."""
    # Mock OpenAI client
    with patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Mock client instance
        client_instance = MagicMock()
        mock_openai.return_value = client_instance
        client_instance.models = MagicMock()
        client_instance.models.list = MagicMock()

        # Custom base URL
        custom_url = "https://custom-openai.example.com/v1"

        # Initialize with custom base URL
        llm = OpenAILLM("gpt-4o", api_key="test-key", api_base=custom_url)
        llm.auth()

        # Verify base URL was set
        assert llm.api_base == custom_url
        assert llm.config["api_base"] == custom_url

        # Verify client initialized with correct parameters
        mock_openai.assert_called_once_with(api_key="test-key", base_url=custom_url)


def test_auth_reuse_existing_client():
    """Test that auth doesn't recreate client if it already exists."""
    # Mock OpenAI client for first auth
    with patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Mock client instance
        first_client = MagicMock()
        mock_openai.return_value = first_client
        first_client.models = MagicMock()
        first_client.models.list = MagicMock()

        # Create instance and authenticate
        llm = OpenAILLM("gpt-4o", api_key="test-key")
        llm.auth()

        # Verify client was created once
        mock_openai.assert_called_once()

        # Save client reference
        original_client = llm.client

        # Reset mock and authenticate again
        mock_openai.reset_mock()
        llm.auth()

        # Verify client wasn't recreated
        mock_openai.assert_not_called()
        assert llm.client is original_client


def test_auth_multiple_attempts_with_different_keys():
    """Test authentication with multiple auth attempts using different keys."""
    # Mock OpenAI client for first auth
    with patch("src.lluminary.models.providers.openai.OpenAI") as mock_openai:
        # Mock client instance
        first_client = MagicMock()
        first_client.models = MagicMock()
        first_client.models.list = MagicMock()

        # Mock second client instance
        second_client = MagicMock()
        second_client.models = MagicMock()
        second_client.models.list = MagicMock()

        # Set up mock to return different clients on each call
        mock_openai.side_effect = [first_client, second_client]

        # Create instance and authenticate
        llm = OpenAILLM("gpt-4o", api_key="first-key")
        llm.auth()

        # Verify first client was created with correct key
        assert mock_openai.call_count == 1
        assert mock_openai.call_args.kwargs["api_key"] == "first-key"
        assert llm.client is first_client

        # Update key and authenticate again
        llm.config["api_key"] = "second-key"
        llm.auth()

        # Verify second client was created with new key
        assert mock_openai.call_count == 2
        assert mock_openai.call_args.kwargs["api_key"] == "second-key"
        assert llm.client is second_client
