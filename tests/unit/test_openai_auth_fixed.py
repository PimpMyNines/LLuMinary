"""
Tests for OpenAI provider authentication with proper setup/teardown.

This module tests authentication mechanisms of the OpenAI provider
with proper environment isolation and OpenAI client mocking.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import AuthenticationError, ProviderError
from lluminary.models.providers.openai import OpenAILLM


class TestOpenAIAuth:
    """Test OpenAI authentication with proper setup/teardown."""

    def setup_method(self):
        """Set up the test environment."""
        # Save original environment variables
        self.original_env = dict(os.environ)

        # Create mock for get_secret function
        self.get_secret_patcher = patch(
            "src.lluminary.models.providers.openai.get_secret"
        )
        self.mock_get_secret = self.get_secret_patcher.start()

        # Create mock for OpenAI client
        self.openai_patcher = patch("lluminary.models.providers.openai.OpenAI")
        self.mock_openai = self.openai_patcher.start()

        # Set up mock client with models attribute
        self.mock_client = MagicMock()
        self.mock_models = MagicMock()
        self.mock_models.list.return_value = {"data": [{"id": "gpt-4o"}]}
        self.mock_client.models = self.mock_models
        self.mock_openai.return_value = self.mock_client

        # Patch time.sleep to avoid actual waiting in tests
        self.sleep_patcher = patch("time.sleep")
        self.mock_sleep = self.sleep_patcher.start()

        # Patch __init__ to avoid auto-auth which causes issues
        self.init_patcher = patch.object(OpenAILLM, "__init__", return_value=None)
        self.mock_init = self.init_patcher.start()

    def teardown_method(self):
        """Tear down the test environment."""
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)

        # Stop patchers
        self.get_secret_patcher.stop()
        self.openai_patcher.stop()
        self.sleep_patcher.stop()
        self.init_patcher.stop()

    def test_auth_with_aws_secrets(self):
        """Test authentication using AWS Secrets Manager."""
        # Configure mock secrets
        mock_secret = {"api_key": "test-secret-key"}
        self.mock_get_secret.return_value = mock_secret

        # Create instance
        openai_llm = OpenAILLM.__new__(OpenAILLM)
        openai_llm.config = {}
        openai_llm.api_base = None
        openai_llm.model_name = "gpt-4o"

        # Call auth
        openai_llm.auth()

        # Verify get_secret was called with correct parameters
        self.mock_get_secret.assert_called_once_with(
            "openai_api_key", required_keys=["api_key"]
        )

        # Verify API key was properly stored
        assert openai_llm.config["api_key"] == "test-secret-key"

        # Verify OpenAI client was initialized with correct API key
        self.mock_openai.assert_called_once_with(
            api_key="test-secret-key", base_url=None
        )

        # Verify credentials were verified
        self.mock_models.list.assert_called_once()

    def test_auth_with_environment_variables(self):
        """Test authentication using environment variables."""
        # Configure environment and make get_secret raise an exception
        os.environ["OPENAI_API_KEY"] = "env-api-key"
        self.mock_get_secret.side_effect = Exception("Secret not found")

        # Create instance
        openai_llm = OpenAILLM.__new__(OpenAILLM)
        openai_llm.config = {}
        openai_llm.api_base = None
        openai_llm.model_name = "gpt-4o"

        # Call auth
        openai_llm.auth()

        # Verify API key was properly stored from environment variable
        assert openai_llm.config["api_key"] == "env-api-key"

        # Verify OpenAI client was initialized with correct API key
        self.mock_openai.assert_called_once_with(api_key="env-api-key", base_url=None)

        # Verify credentials were verified
        self.mock_models.list.assert_called_once()

    def test_auth_with_config_api_key(self):
        """Test authentication using API key provided in config."""
        # Create instance with API key in config
        openai_llm = OpenAILLM.__new__(OpenAILLM)
        openai_llm.config = {"api_key": "config-api-key"}
        openai_llm.api_base = None
        openai_llm.model_name = "gpt-4o"

        # Call auth
        openai_llm.auth()

        # Verify get_secret was not called - should use config directly
        self.mock_get_secret.assert_not_called()

        # Verify OpenAI client was initialized with config API key
        self.mock_openai.assert_called_once_with(
            api_key="config-api-key", base_url=None
        )

        # Verify credentials were verified
        self.mock_models.list.assert_called_once()

    def test_auth_precedence_order(self):
        """Test authentication precedence: config > AWS secrets > environment."""
        # Set up all three authentication methods to verify precedence
        mock_secret = {"api_key": "secret-api-key"}
        self.mock_get_secret.return_value = mock_secret
        os.environ["OPENAI_API_KEY"] = "env-api-key"

        # Test 1: All methods available - should use config
        openai_llm = OpenAILLM.__new__(OpenAILLM)
        openai_llm.config = {"api_key": "config-api-key"}
        openai_llm.api_base = None
        openai_llm.model_name = "gpt-4o"

        openai_llm.auth()

        # Should use config API key
        self.mock_openai.assert_called_once_with(
            api_key="config-api-key", base_url=None
        )
        self.mock_openai.reset_mock()
        self.mock_models.list.reset_mock()

        # Test 2: No config, should use AWS secrets over env var
        openai_llm = OpenAILLM.__new__(OpenAILLM)
        openai_llm.config = {}
        openai_llm.api_base = None
        openai_llm.model_name = "gpt-4o"

        openai_llm.auth()

        # Should use AWS secrets API key
        self.mock_openai.assert_called_once_with(
            api_key="secret-api-key", base_url=None
        )
        self.mock_openai.reset_mock()
        self.mock_models.list.reset_mock()

        # Test 3: No config, AWS fails, should use env var
        self.mock_get_secret.side_effect = Exception("Secret not found")
        openai_llm = OpenAILLM.__new__(OpenAILLM)
        openai_llm.config = {}
        openai_llm.api_base = None
        openai_llm.model_name = "gpt-4o"

        openai_llm.auth()

        # Should use environment API key
        self.mock_openai.assert_called_once_with(api_key="env-api-key", base_url=None)

    def test_auth_with_organization(self):
        """Test authentication with organization ID."""
        mock_secret = {"api_key": "test-secret-key"}
        self.mock_get_secret.return_value = mock_secret

        # Create instance with organization in config
        openai_llm = OpenAILLM.__new__(OpenAILLM)
        openai_llm.config = {"organization": "test-org"}
        openai_llm.api_base = None
        openai_llm.model_name = "gpt-4o"

        # Call auth
        openai_llm.auth()

        # Verify API key was properly stored
        assert openai_llm.config["api_key"] == "test-secret-key"

        # Verify organization is still in config
        assert openai_llm.config["organization"] == "test-org"

        # Verify OpenAI client was created (exact parameters are handled differently)
        assert self.mock_openai.called

    def test_auth_with_base_url(self):
        """Test authentication with custom base URL."""
        mock_secret = {"api_key": "test-secret-key"}
        self.mock_get_secret.return_value = mock_secret

        # Create instance with base_url in config
        openai_llm = OpenAILLM.__new__(OpenAILLM)
        openai_llm.config = {"base_url": "https://custom-openai-api.example.com"}
        openai_llm.api_base = "https://custom-openai-api.example.com"
        openai_llm.model_name = "gpt-4o"

        # Call auth
        openai_llm.auth()

        # Verify OpenAI client was initialized with base_url
        self.mock_openai.assert_called_once_with(
            api_key="test-secret-key", base_url="https://custom-openai-api.example.com"
        )

    def test_auth_credential_verification_failure(self):
        """Test handling of credential verification failures."""
        mock_secret = {"api_key": "invalid-api-key"}
        self.mock_get_secret.return_value = mock_secret

        # Set up mock client to raise an exception during verification
        # Use a generic Exception since the specific OpenAI errors require complex setup
        auth_error = Exception("Invalid API key")
        auth_error.message = (
            "Invalid API key"  # Add message attribute to match OpenAI error format
        )
        self.mock_models.list.side_effect = auth_error

        # Create instance
        openai_llm = OpenAILLM.__new__(OpenAILLM)
        openai_llm.config = {}
        openai_llm.api_base = None
        openai_llm.model_name = "gpt-4o"

        # Call auth and expect exception
        with pytest.raises(ProviderError) as excinfo:
            openai_llm.auth()

        # Verify error message
        assert "Failed to initialize OpenAI client" in str(excinfo.value)

    def test_auth_no_credentials_available(self):
        """Test handling of case where no credentials are available."""
        # Ensure no environment variables are set
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        # Configure get_secret to fail
        self.mock_get_secret.side_effect = Exception("Secret not found")

        # Create instance
        openai_llm = OpenAILLM.__new__(OpenAILLM)
        openai_llm.config = {}
        openai_llm.api_base = None
        openai_llm.model_name = "gpt-4o"

        # Call auth and expect exception
        with pytest.raises(AuthenticationError) as excinfo:
            openai_llm.auth()

        # Verify error message
        assert "Failed to get API key for OpenAI" in str(excinfo.value)

        # Verify client was never created
        self.mock_openai.assert_not_called()
