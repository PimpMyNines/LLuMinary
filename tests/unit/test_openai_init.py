"""
Basic tests for OpenAI provider initialization and authentication.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.openai import OpenAILLM


@pytest.fixture
def openai_llm():
    """Fixture for OpenAI LLM instance."""
    with patch.object(OpenAILLM, "auth") as mock_auth:
        # Mock authentication to avoid API errors
        mock_auth.return_value = None

        # Create the LLM instance with mock API key
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Initialize client attribute directly for tests
        llm.client = MagicMock()

        yield llm


def test_openai_initialization(openai_llm):
    """Test OpenAI provider initialization."""
    # Verify model-related settings in initialization
    assert openai_llm.model_name == "gpt-4o"
    assert openai_llm.config["api_key"] == "test-key"

    # Test validation of model name during initialization
    assert openai_llm.validate_model("gpt-4o") is True
    assert openai_llm.validate_model("invalid-model") is False
