"""
Base test configuration and fixtures for testing LLMHandler.
"""

import os
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from lluminary.exceptions import LLMMistake


# Sample test function for function calling tests
def get_weather(location: str) -> str:
    """Get the current weather in a given location

    Args:
        location: The city and state, e.g. San Francisco, CA

    Returns:
        The current weather in the given location
    """
    return "60 degrees"


# List of models to test with, one from each provider
@pytest.fixture
def test_models() -> List[str]:
    """
    Returns a list of models to test, one per provider.
    This helps run tests without making too many redundant API calls.
    """
    return [
        "gpt-4o-mini",  # OpenAI
        "claude-haiku-3.5",  # Anthropic
        "gemini-2.0-flash-lite",  # Google
        "bedrock-claude-haiku-3.5",  # AWS Bedrock
        "text-embedding-3-small",  # OpenAI embedding
        "rerank-english-v3.0",  # Cohere reranking
    ]


@pytest.fixture
def test_image_url() -> str:
    """Returns a publicly available test image URL."""
    return "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png"


@pytest.fixture
def test_categories() -> Dict[str, str]:
    """Returns sample categories for classification tests."""
    return {
        "question": "A query seeking information",
        "command": "A directive to perform an action",
        "statement": "A declarative sentence",
    }


@pytest.fixture
def test_examples() -> List[Dict[str, str]]:
    """Returns sample examples for classification tests."""
    return [
        {
            "user_input": "What is the weather like?",
            "doc_str": "This is a question seeking information about weather",
            "selection": "question",
        }
    ]


@pytest.fixture
def test_message() -> Dict[str, Any]:
    """Returns a sample message for testing."""
    return {
        "message_type": "human",
        "message": "How do I open this file?",
        "image_paths": [],
        "image_urls": [],
    }


@pytest.fixture
def test_image_message(test_image_url: str) -> Dict[str, Any]:
    """Returns a sample message with an image for testing."""
    return {
        "message_type": "human",
        "message": "What is in this image?",
        "image_paths": [],
        "image_urls": [test_image_url],
    }


@pytest.fixture
def number_validator():
    """Returns a function that validates numeric responses."""

    def process_function(response: str):
        """
        Process the response to extract a number between 6 and 12 from <num> tags.

        Args:
            response (str): The model's response

        Returns:
            int: The extracted number

        Raises:
            LLMMistake: If the response is not properly formatted or the number is outside the range
        """
        try:
            number_str = response.split("<num>")[1].split("</num>")[0]
            number = int(number_str)
        except:
            raise LLMMistake("The response must be wrapped in <num> tags.")

        if not (number < 12 and number > 6):
            raise LLMMistake("The given number must be between 6 and 12.")

        return number

    return process_function


@pytest.fixture
def mock_env_vars():
    """Set mock environment variables for testing."""
    # Save original environment
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ["OPENAI_API_KEY_API_KEY"] = "test-openai-key"
    os.environ["ANTHROPIC_API_KEY_API_KEY"] = "test-anthropic-key"
    os.environ["GOOGLE_API_KEY_API_KEY"] = "test-google-key"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_model_clients():
    """Patch model clients and auth to avoid actual API calls during unit tests."""
    # This is a more focused mock implementation
    with patch("src.lluminary.models.base.LLM.auth", return_value=None):
        yield {"mock_fixture": True}
