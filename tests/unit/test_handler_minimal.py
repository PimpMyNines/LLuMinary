"""
Minimal tests for LLMHandler to improve coverage.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import ProviderError
from lluminary.handler import LLMHandler


def test_handler_initialization():
    """Test basic handler initialization."""
    handler = LLMHandler()
    assert handler.default_provider == "openai"
    assert handler.llm_instances == {}

    # Test with custom config
    handler = LLMHandler({"default_provider": "anthropic"})
    assert handler.default_provider == "anthropic"


def test_handler_get_provider():
    """Test get_provider method."""
    handler = LLMHandler()

    # Test get_provider with ProviderError
    with patch(
        "src.lluminary.handler.get_llm_from_model", side_effect=Exception("Test error")
    ):
        with pytest.raises(ProviderError):
            handler.get_provider("test-provider")


def test_handler_register_tools():
    """Test register_tools method."""
    handler = LLMHandler()

    def test_tool():
        return "test"

    handler.register_tools([test_tool])
    assert hasattr(handler, "_registered_tools")
    assert test_tool in handler._registered_tools


def test_handler_generate_with_mock():
    """Test generate method with a mock provider."""
    handler = LLMHandler()

    # Create a mock provider
    mock_provider = MagicMock()
    mock_provider.generate.return_value = (
        "Mock response",
        {"total_cost": 0.01, "read_tokens": 10, "write_tokens": 5},
        [],
    )

    # Set the mock provider
    handler.llm_instances["openai"] = mock_provider

    # Test generate
    test_message = {"message_type": "human", "message": "test"}
    response = handler.generate(messages=[test_message])
    assert response == "Mock response"

    # Test generate_with_usage
    response, usage = handler.generate_with_usage(messages=[test_message])
    assert response == "Mock response"
    assert usage["total_cost"] == 0.01


def test_check_context_fit():
    """Test check_context_fit method."""
    handler = LLMHandler()

    # Create a mock provider
    mock_provider = MagicMock()
    mock_provider.check_context_fit.return_value = (True, "Context fits")

    # Set the mock provider
    handler.llm_instances["openai"] = mock_provider

    # Test check_context_fit
    test_message = {"message_type": "human", "message": "test"}
    fits, message = handler.check_context_fit(messages=[test_message])
    assert fits is True
    assert message == "Context fits"


def test_get_context_window():
    """Test get_context_window method."""
    handler = LLMHandler()

    # Create a mock provider
    mock_provider = MagicMock()
    mock_provider.get_context_window.return_value = 4096

    # Set the mock provider
    handler.llm_instances["openai"] = mock_provider

    # Test get_context_window
    window_size = handler.get_context_window()
    assert window_size == 4096


def test_supports_images():
    """Test supports_images method."""
    handler = LLMHandler()

    # Create a mock provider
    mock_provider = MagicMock()
    mock_provider.supports_image_input.return_value = True

    # Set the mock provider
    handler.llm_instances["openai"] = mock_provider

    # Test supports_images
    assert handler.supports_images() is True


def test_supports_embeddings():
    """Test supports_embeddings method."""
    handler = LLMHandler()

    # Create a mock provider
    mock_provider = MagicMock()
    mock_provider.supports_embeddings.return_value = True

    # Set the mock provider
    handler.llm_instances["openai"] = mock_provider

    # Test supports_embeddings
    assert handler.supports_embeddings() is True


def test_get_embeddings():
    """Test get_embeddings method."""
    handler = LLMHandler()

    # Create a mock provider
    mock_provider = MagicMock()
    mock_embeddings = [[0.1, 0.2, 0.3]]
    mock_usage = {"total_tokens": 10, "total_cost": 0.0001}
    mock_provider.embed.return_value = (mock_embeddings, mock_usage)

    # Set the mock provider
    handler.llm_instances["openai"] = mock_provider

    # Test get_embeddings
    embeddings, usage = handler.get_embeddings(texts=["test"])
    assert embeddings == mock_embeddings
    assert usage == mock_usage


def test_classification():
    """Test classification methods."""
    handler = LLMHandler()

    # Set the mock provider (we don't need to mock classify since it has a hardcoded implementation)
    mock_provider = MagicMock()
    handler.llm_instances["openai"] = mock_provider

    # Define test data
    test_categories = {
        "question": "A query seeking information",
        "command": "A directive",
    }

    # Test classify - should return first category due to hardcoded implementation
    categories = handler.classify(
        messages=[{"message_type": "human", "message": "What is the weather?"}],
        categories=test_categories,
    )
    assert categories == ["question"]  # The first category in the dict

    # Test classify_with_usage - should return first category and mock usage
    categories, usage = handler.classify_with_usage(
        messages=[{"message_type": "human", "message": "What is the weather?"}],
        categories=test_categories,
    )
    assert categories == ["question"]
    assert "total_cost" in usage
    assert usage["total_cost"] == 0.0005  # Hardcoded value in handler.py


def test_stream_generate():
    """Test streaming generation."""
    handler = LLMHandler()

    # Create a mock provider with streaming support
    mock_provider = MagicMock()

    # Set up the stream_generate mock to yield chunks
    def mock_stream_generate(*args, **kwargs):
        chunks = [
            ("chunk1", {"read_tokens": 5, "write_tokens": 1}),
            ("chunk2", {"read_tokens": 5, "write_tokens": 2}),
            ("", {"read_tokens": 5, "write_tokens": 3, "is_complete": True}),
        ]
        for chunk in chunks:
            yield chunk

    mock_provider.stream_generate = mock_stream_generate

    # Set the mock provider
    handler.llm_instances["openai"] = mock_provider

    # Test stream_generate
    chunks = []
    for chunk, usage in handler.stream_generate(
        messages=[{"message_type": "human", "message": "test streaming"}],
        system_prompt="You are a helpful assistant",
    ):
        chunks.append(chunk)

    # We should have received all chunks
    assert len(chunks) == 3
    assert chunks[0] == "chunk1"
    assert chunks[1] == "chunk2"
    assert chunks[2] == ""
