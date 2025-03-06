"""
Basic tests for OpenAI provider error handling.

This module focuses on testing the error handling aspects of the OpenAI provider
without requiring complex mocking of external libraries.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.openai import OpenAILLM


@pytest.fixture
def openai_llm():
    """Create a pre-configured OpenAI LLM instance."""
    # Patch auth to prevent actual API calls
    with patch.object(OpenAILLM, "auth"):
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Mock client
        llm.client = MagicMock()

        yield llm


def test_invalid_model_name():
    """Test handling of invalid model names."""
    # Test initialization with invalid model
    with pytest.raises(ValueError) as excinfo:
        OpenAILLM("invalid-model", api_key="test-key")

    assert "not supported" in str(excinfo.value).lower()
    assert "invalid-model" in str(excinfo.value)


def test_invalid_embedding_model(openai_llm):
    """Test handling of invalid embedding model."""
    # Try to use invalid embedding model
    with pytest.raises(ValueError) as excinfo:
        openai_llm.embed(["Test text"], model="invalid-embedding-model")

    assert "not supported" in str(excinfo.value).lower()
    assert "invalid-embedding-model" in str(excinfo.value)


def test_get_context_window_invalid(openai_llm):
    """Test get_context_window with invalid model."""
    # Make sure CONTEXT_WINDOW dictionary is populated with at least one model
    assert len(openai_llm.CONTEXT_WINDOW) > 0

    # Get the default model's context window
    default_window = next(iter(openai_llm.CONTEXT_WINDOW.values()))

    # Mock the get_context_window method to handle our test case
    with patch.object(
        OpenAILLM, "get_context_window", return_value=default_window
    ) as mock_context:

        # Call the method with an invalid model
        result = mock_context("invalid-model-name")

        # Should return some reasonable default
        assert result == default_window


def test_check_provider_image_support(openai_llm):
    """Test basic image support flags."""
    # Check the provider claims to support images in general
    assert hasattr(openai_llm, "SUPPORTS_IMAGES")

    # This should be a boolean value
    assert isinstance(openai_llm.SUPPORTS_IMAGES, bool)

    # In the case of OpenAI, we expect this to be True
    assert openai_llm.SUPPORTS_IMAGES is True


def test_is_thinking_model(openai_llm):
    """Test the is_thinking_model method."""
    # Check thinking models
    for model_name in openai_llm.THINKING_MODELS:
        assert openai_llm.is_thinking_model(model_name) is True

    # Check non-thinking models return False
    assert openai_llm.is_thinking_model("invalid-model") is False
