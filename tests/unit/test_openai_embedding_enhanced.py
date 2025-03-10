"""
Enhanced tests for OpenAI provider's embedding functionality.

This module extends the basic embedding tests with more complex scenarios,
including rate limit handling, different dimensions, and error cases.
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
        MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5]),
        MagicMock(embedding=[0.2, 0.3, 0.4, 0.5, 0.6]),
        MagicMock(embedding=[0.3, 0.4, 0.5, 0.6, 0.7]),
    ]

    # Create mock usage data
    usage = MagicMock(total_tokens=30)

    # Create the complete response object
    response = MagicMock()
    response.data = data
    response.usage = usage

    return response


def test_embed_cost_calculation_accuracy(openai_llm):
    """Test that embedding cost calculation is accurate."""
    # Set up test cases with different models and token counts
    test_cases = [
        {
            "model": "text-embedding-ada-002",
            "tokens": 1000,
            "expected_cost": 0.1,
        },  # $0.0001 per token
        {
            "model": "text-embedding-3-small",
            "tokens": 1000,
            "expected_cost": 0.02,
        },  # $0.00002 per token
        {
            "model": "text-embedding-3-large",
            "tokens": 1000,
            "expected_cost": 0.13,
        },  # $0.00013 per token
    ]

    for case in test_cases:
        # Mock response with specified token count
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1, 0.2])]
        response.usage = MagicMock(total_tokens=case["tokens"])

        # Set up mock for this case
        openai_llm.client.embeddings.create.reset_mock()
        openai_llm.client.embeddings.create.return_value = response

        # Call embed with specified model
        embeddings, usage = openai_llm.embed(["Test"], model=case["model"])

        # Verify cost calculation
        assert usage["model"] == case["model"]
        assert usage["tokens"] == case["tokens"]
        assert (
            abs(usage["cost"] - case["expected_cost"]) < 0.0001
        )  # Account for float precision


def test_embed_with_empty_batch(openai_llm):
    """Test behavior with an empty batch."""
    # Call embed with an empty batch
    embeddings, usage = openai_llm.embed([])

    # Should return empty list without making API call
    assert embeddings == []
    assert usage["tokens"] == 0
    assert usage["cost"] == 0
    openai_llm.client.embeddings.create.assert_not_called()


def test_embed_with_multiple_models_sequential(openai_llm):
    """Test embedding with different models sequentially."""
    models = [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]

    for model in models:
        # Create mock response
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        response.usage = MagicMock(total_tokens=10)

        # Reset and set mock
        openai_llm.client.embeddings.create.reset_mock()
        openai_llm.client.embeddings.create.return_value = response

        # Call embed with this model
        embeddings, usage = openai_llm.embed(["Test text"], model=model)

        # Verify correct model was used
        call_args = openai_llm.client.embeddings.create.call_args[1]
        assert call_args["model"] == model
        assert usage["model"] == model


def test_embed_with_dimension_validation(openai_llm):
    """Test that embedding dimensions are consistent with the model."""
    # Set up expected dimensions for each model
    model_dimensions = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    for model, expected_dim in model_dimensions.items():
        # Create mock response with correct dimensions
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1] * expected_dim)]
        response.usage = MagicMock(total_tokens=10)

        # Mock the embeddings.create method for this model
        openai_llm.client.embeddings.create.reset_mock()
        openai_llm.client.embeddings.create.return_value = response

        # Generate embedding with this model
        embeddings, usage = openai_llm.embed(["Test text"], model=model)

        # Verify dimension
        assert len(embeddings[0]) == expected_dim

        # Verify client was called with correct model
        call_args = openai_llm.client.embeddings.create.call_args[1]
        assert call_args["model"] == model


def test_embed_with_very_long_text(openai_llm):
    """Test embedding with text that exceeds token limits."""
    # Create a very long text (100K characters)
    very_long_text = (
        "This is a very long text that exceeds normal token limits. " * 2000
    )

    # Mock the embeddings.create method to handle this case
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    response.usage = MagicMock(total_tokens=8000)  # A high token count
    openai_llm.client.embeddings.create.return_value = response

    # Call embed with very long text
    embeddings, usage = openai_llm.embed([very_long_text])

    # Verify we got back an embedding
    assert len(embeddings) == 1

    # Verify high token usage
    assert usage["tokens"] == 8000
    assert usage["cost"] > 0
