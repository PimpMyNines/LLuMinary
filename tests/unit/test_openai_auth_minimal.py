"""
Minimal tests for OpenAI provider authentication mechanisms.

These tests focus on the core authentication behavior of the OpenAI provider,
with minimal dependencies and mocking.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.openai import OpenAILLM


# Create a mock module for openai
class MockOpenAI:
    def __init__(self, api_key=None, organization=None, base_url=None):
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url
        self.models = MagicMock()
        self.models.list = MagicMock(return_value=[])


# Mock the entire openai module
sys.modules["openai"] = MagicMock()
sys.modules["openai"].OpenAI = MockOpenAI


@pytest.fixture(autouse=True)
def clean_environment():
    """Setup and teardown for environment variables."""
    # Save original environment
    original_env = os.environ.copy()

    # Remove any existing OpenAI environment variables
    for key in list(os.environ.keys()):
        if key.startswith("OPENAI_"):
            del os.environ[key]

    # Run the test
    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def test_direct_api_key():
    """Test direct API key configuration."""
    # Direct API key test
    test_key = "sk-test-direct-key"

    # Create and auth the LLM
    llm = OpenAILLM("gpt-4o", api_key=test_key)
    llm.auth()

    # Verify key is stored in config
    assert llm.config["api_key"] == test_key

    # Verify client has correct key
    assert hasattr(llm, "client")
    assert llm.client.api_key == test_key


def test_environment_api_key():
    """Test environment variable API key fallback."""
    # Set environment variable
    test_key = "sk-test-env-key"
    os.environ["OPENAI_API_KEY"] = test_key

    # Mock get_secret to fail
    with patch(
        "src.lluminary.models.providers.openai.get_secret",
        side_effect=Exception("AWS Secret not found"),
    ):

        # Create and auth the LLM without direct key
        llm = OpenAILLM("gpt-4o")
        llm.auth()

        # Verify key is retrieved from environment
        assert llm.config["api_key"] == test_key

        # Verify client has correct key
        assert hasattr(llm, "client")
        assert llm.client.api_key == test_key


def test_custom_base_url():
    """Test custom base URL configuration."""
    # Custom base URL
    test_url = "https://custom-openai.example.com/v1"

    # Create and auth the LLM with custom URL
    llm = OpenAILLM("gpt-4o", api_key="sk-test-key", api_base=test_url)
    llm.auth()

    # Verify URL is stored
    assert llm.api_base == test_url
    assert llm.config["api_base"] == test_url

    # Verify client has correct URL
    assert hasattr(llm, "client")
    assert llm.client.base_url == test_url
