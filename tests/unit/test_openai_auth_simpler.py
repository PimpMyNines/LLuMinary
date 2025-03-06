"""
Simplified tests for OpenAI provider authentication.

This module uses a simpler approach to test authentication functionality in the OpenAI provider,
using dependency injection and clean mocking techniques.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.openai import OpenAILLM


# We'll inject our mocks directly in each test, rather than trying to patch at the module level
class MockClient:
    """Simple mock for OpenAI client"""

    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.organization = kwargs.get("organization")
        self.base_url = kwargs.get("base_url")

        # Add models attribute with list method
        self.models = MagicMock()
        self.models.list = MagicMock(return_value=[])


# Clean up environment variables before and after tests
@pytest.fixture(autouse=True)
def clean_env():
    """Save and restore environment variables"""
    # Store original environment
    original_env = os.environ.copy()

    # Clear relevant env vars for isolation
    for key in ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_ORGANIZATION"]:
        if key in os.environ:
            del os.environ[key]

    # Run the test
    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def test_direct_api_key_config():
    """Test configuration with directly provided API key"""
    # Patch the OpenAI class to return our mock
    with patch(
        "src.lluminary.models.providers.openai.OpenAI", return_value=MockClient()
    ) as mock_openai:
        # Create instance with direct API key
        api_key = "sk-test-direct-key"
        llm = OpenAILLM("gpt-4o", api_key=api_key)

        # Perform auth
        llm.auth()

        # Check initialization parameters
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == api_key

        # Check stored config
        assert llm.config["api_key"] == api_key


def test_aws_secrets_config():
    """Test configuration with AWS Secrets Manager"""
    # Create mock secret
    secret = {"api_key": "sk-test-secret-key", "organization_id": "org-test-123"}

    # Patch get_secret and OpenAI class
    with patch(
        "src.lluminary.models.providers.openai.get_secret", return_value=secret
    ), patch(
        "src.lluminary.models.providers.openai.OpenAI", return_value=MockClient()
    ) as mock_openai:

        # Create instance without direct parameters
        llm = OpenAILLM("gpt-4o")

        # Perform auth
        llm.auth()

        # Check initialization parameters
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == secret["api_key"]
        assert call_kwargs["organization"] == secret["organization_id"]

        # Check stored config
        assert llm.config["api_key"] == secret["api_key"]
        assert llm.config["organization_id"] == secret["organization_id"]


def test_env_vars_config():
    """Test configuration with environment variables"""
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = "sk-test-env-key"
    os.environ["OPENAI_API_BASE"] = "https://test-env.openai.com"

    # Patch get_secret to fail and OpenAI class
    with patch(
        "src.lluminary.models.providers.openai.get_secret",
        side_effect=Exception("AWS error"),
    ), patch(
        "src.lluminary.models.providers.openai.OpenAI", return_value=MockClient()
    ) as mock_openai:

        # Create instance without direct parameters
        llm = OpenAILLM("gpt-4o")

        # Perform auth
        llm.auth()

        # Check initialization parameters
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-test-env-key"
        assert call_kwargs["base_url"] == "https://test-env.openai.com"

        # Check stored config
        assert llm.config["api_key"] == "sk-test-env-key"
        assert llm.api_base == "https://test-env.openai.com"


def test_combined_config():
    """Test config with a combination of parameters"""
    # Set up test parameters
    direct_api_key = "sk-test-direct-key"
    direct_api_base = "https://direct-test.openai.com"
    direct_org_id = "org-direct-test"

    # Patch OpenAI class
    with patch(
        "src.lluminary.models.providers.openai.OpenAI", return_value=MockClient()
    ) as mock_openai:
        # Create instance with all parameters
        llm = OpenAILLM(
            "gpt-4o",
            api_key=direct_api_key,
            api_base=direct_api_base,
            organization_id=direct_org_id,
        )

        # Perform auth
        llm.auth()

        # Check initialization parameters
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == direct_api_key
        assert call_kwargs["base_url"] == direct_api_base
        assert call_kwargs["organization"] == direct_org_id

        # Check stored config
        assert llm.config["api_key"] == direct_api_key
        assert llm.api_base == direct_api_base
        assert llm.config["organization_id"] == direct_org_id
