"""Simple tests for OpenAI provider."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from lluminary.exceptions import ServiceUnavailableError
from lluminary.models.providers.openai import OpenAILLM


def test_openai_init():
    """Test basic initialization."""
    with patch.object(OpenAILLM, "auth"):
        llm = OpenAILLM("gpt-4o")
        assert llm.model_name == "gpt-4o"


def test_openai_timeout_handling():
    """Test handling of timeout errors."""
    with patch.object(OpenAILLM, "auth"):
        # Create LLM instance with mock
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Mock the client
        llm.client = MagicMock()

        # Mock a timeout error
        llm.client.chat.completions.create.side_effect = requests.exceptions.Timeout(
            "Request timed out"
        )

        # Try to generate - this should handle the timeout
        with pytest.raises(ServiceUnavailableError) as excinfo:
            llm._raw_generate(
                event_id="test",
                system_prompt="Test",
                messages=[{"message_type": "human", "message": "Hello"}],
            )

        # Check that error was properly wrapped
        assert "timed out" in str(excinfo.value).lower()
