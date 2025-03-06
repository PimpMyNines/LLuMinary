"""
Basic tests for OpenAI provider's embedding functionality.

This module focuses on testing the embedding capabilities of the OpenAI provider,
covering basic functionality, batching, and error handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.openai import OpenAILLM


@pytest.fixture
def openai_llm():
    """Fixture for OpenAI LLM instance with proper mocking."""
    with patch.object(OpenAILLM, "auth") as mock_auth:
        # Mock authentication to avoid API errors
        mock_auth.return_value = None

        # Create the LLM instance with mock API key
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Initialize client attribute directly for tests
        llm.client = MagicMock()

        yield llm


@pytest.fixture
def mock_embedding_response():
    """Create a mock embedding response with realistic structure."""
    # Create mock embedding data items
    data = [
        MagicMock(embedding=[0.1, 0.2, 0.3, 0.4]),
        MagicMock(embedding=[0.2, 0.3, 0.4, 0.5]),
        MagicMock(embedding=[0.3, 0.4, 0.5, 0.6]),
    ]

    # Create mock usage data
    usage = MagicMock(total_tokens=30)

    # Create the complete response object
    response = MagicMock()
    response.data = data
    response.usage = usage

    return response


def test_embed_basic_functionality(openai_llm, mock_embedding_response):
    """Test basic embedding functionality."""
    # Mock the OpenAI embeddings.create method
    openai_llm.client.embeddings.create.return_value = mock_embedding_response

    # Call embed with a single text
    texts = ["This is a test sentence."]
    embeddings, usage = openai_llm.embed(texts)

    # Verify OpenAI client was called correctly
    openai_llm.client.embeddings.create.assert_called_once()
    call_args = openai_llm.client.embeddings.create.call_args[1]
    assert call_args["model"] == openai_llm.DEFAULT_EMBEDDING_MODEL
    assert call_args["input"] == texts

    # Verify embeddings structure
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 4  # Our mock has 4-dimensional vectors

    # Verify usage information
    assert "tokens" in usage
    assert "cost" in usage
    assert "model" in usage
    assert usage["tokens"] == 30
    assert usage["cost"] > 0


def test_embed_multiple_texts(openai_llm, mock_embedding_response):
    """Test embedding multiple texts at once."""
    # Mock the OpenAI embeddings.create method
    openai_llm.client.embeddings.create.return_value = mock_embedding_response

    # Call embed with multiple texts
    texts = [
        "This is the first test sentence.",
        "This is the second test sentence.",
        "This is the third test sentence.",
    ]
    embeddings, usage = openai_llm.embed(texts)

    # Verify OpenAI client was called correctly
    openai_llm.client.embeddings.create.assert_called_once()
    call_args = openai_llm.client.embeddings.create.call_args[1]
    assert call_args["model"] == openai_llm.DEFAULT_EMBEDDING_MODEL
    assert call_args["input"] == texts

    # Verify embeddings structure
    assert len(embeddings) == 3
    for embedding in embeddings:
        assert len(embedding) == 4  # Our mock has 4-dimensional vectors

    # Verify usage information
    assert usage["tokens"] == 30


def test_embed_with_batching(openai_llm):
    """Test embedding with batching for large text sets."""
    # Create multiple mock responses for batches
    batch1_response = MagicMock()
    batch1_response.data = [
        MagicMock(embedding=[0.1, 0.2]),
        MagicMock(embedding=[0.3, 0.4]),
    ]
    batch1_response.usage = MagicMock(total_tokens=20)

    batch2_response = MagicMock()
    batch2_response.data = [
        MagicMock(embedding=[0.5, 0.6]),
        MagicMock(embedding=[0.7, 0.8]),
    ]
    batch2_response.usage = MagicMock(total_tokens=20)

    # Set up the client to return different responses for each batch
    openai_llm.client.embeddings.create.side_effect = [batch1_response, batch2_response]

    # Create a list of texts longer than the batch size
    texts = ["Text 1", "Text 2", "Text 3", "Text 4"]

    # Call embed with a small batch size
    embeddings, usage = openai_llm.embed(texts, batch_size=2)

    # Verify the client was called twice (once per batch)
    assert openai_llm.client.embeddings.create.call_count == 2

    # First call should have the first two texts
    first_call_args = openai_llm.client.embeddings.create.call_args_list[0][1]
    assert first_call_args["input"] == texts[:2]

    # Second call should have the last two texts
    second_call_args = openai_llm.client.embeddings.create.call_args_list[1][1]
    assert second_call_args["input"] == texts[2:]

    # Verify we got back all embeddings
    assert len(embeddings) == 4

    # Verify token count is combined from all batches
    assert usage["tokens"] == 40  # 20 + 20


def test_embed_error_handling(openai_llm):
    """Test error handling in the embed method."""
    # Test invalid model
    with pytest.raises(ValueError) as excinfo:
        openai_llm.embed(["Test"], model="invalid-model")
    assert "not supported" in str(excinfo.value)

    # Test API error
    openai_llm.client.embeddings.create.side_effect = Exception("API error")
    with pytest.raises(ValueError) as excinfo:
        openai_llm.embed(["Test"])
    assert "Error getting embeddings" in str(excinfo.value)
    assert "API error" in str(excinfo.value)

    # Test empty input
    openai_llm.client.embeddings.create.side_effect = None
    embeddings, usage = openai_llm.embed([])
    assert embeddings == []
    assert usage["tokens"] == 0
    assert usage["cost"] == 0


def test_embed_with_custom_model(openai_llm, mock_embedding_response):
    """Test embedding with a custom model."""
    # Mock the OpenAI embeddings.create method
    openai_llm.client.embeddings.create.return_value = mock_embedding_response

    # Use a custom embedding model
    custom_model = "text-embedding-3-large"

    # Call embed with custom model
    texts = ["Test sentence"]
    embeddings, usage = openai_llm.embed(texts, model=custom_model)

    # Verify the client was called with the right model
    openai_llm.client.embeddings.create.assert_called_once()
    call_args = openai_llm.client.embeddings.create.call_args[1]
    assert call_args["model"] == custom_model

    # Verify the model is recorded in usage statistics
    assert usage["model"] == custom_model

    # Verify cost is calculated based on the custom model's pricing
    expected_cost = 30 * openai_llm.embedding_costs[custom_model]
    assert usage["cost"] == expected_cost
