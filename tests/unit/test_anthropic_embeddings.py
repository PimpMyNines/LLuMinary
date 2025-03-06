"""
Unit tests for the Anthropic provider embedding functionality.

This module contains tests for the embedding features in the Anthropic provider.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import LLMMistake
from lluminary.models.providers.anthropic import AnthropicLLM


@pytest.fixture
def anthropic_llm():
    """Fixture for Anthropic LLM instance."""
    with patch("anthropic.Anthropic") as mock_anthropic, patch(
        "requests.post"
    ) as mock_post:
        # Create the LLM instance with mock API key
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Ensure client is initialized
        llm.client = MagicMock()

        # Ensure config exists
        if not hasattr(llm, "config"):
            llm.config = {}

        # Add client to config as expected by implementation
        llm.config["client"] = llm.client
        llm.config["api_key"] = "test-key"

        yield llm


def test_embed_basic_functionality(anthropic_llm):
    """Test basic embedding functionality."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]],
        "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
    }
    mock_response.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_response):
        # Test embedding multiple texts
        texts = ["This is the first text", "This is the second text"]
        embeddings, usage = anthropic_llm.embed(texts=texts)

        # Verify embeddings format
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 5
        assert isinstance(embeddings[0][0], float)
        assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert embeddings[1] == [0.2, 0.3, 0.4, 0.5, 0.6]

        # Verify usage information
        assert "total_tokens" in usage
        assert usage["total_tokens"] == 10
        assert "total_cost" in usage


def test_embed_with_custom_model(anthropic_llm):
    """Test embeddings with a custom model."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "usage": {"input_tokens": 5, "output_tokens": 0, "total_tokens": 5},
    }
    mock_response.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_response) as mock_post:
        # Call embed with custom model
        custom_model = "claude-3-5-sonnet-20241022"
        embeddings, usage = anthropic_llm.embed(texts=["Test text"], model=custom_model)

        # Verify API was called with the custom model
        call_args = mock_post.call_args[1]
        request_body = call_args["json"]
        assert request_body["model"] == custom_model

        # Verify results
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]


def test_embed_empty_texts(anthropic_llm):
    """Test embedding with empty input."""
    # Call embed with empty list
    embeddings, usage = anthropic_llm.embed(texts=[])

    # Should return empty results
    assert len(embeddings) == 0
    assert usage["total_tokens"] == 0
    assert usage["total_cost"] == 0


def test_embed_error_handling(anthropic_llm):
    """Test error handling in embed method."""
    # Mock API error
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("API Error")

    with patch("requests.post", return_value=mock_response):
        # Call embed and expect exception
        with pytest.raises(LLMMistake) as excinfo:
            anthropic_llm.embed(texts=["Test text"])

        # Verify error message
        assert "API Error" in str(excinfo.value)
        assert "anthropic" in str(excinfo.value).lower()


def test_embed_rate_limit_error(anthropic_llm):
    """Test handling of rate limit errors in embed method."""
    # Mock rate limit error response
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.raise_for_status.side_effect = Exception("Rate limit exceeded")
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}

    with patch("requests.post", return_value=mock_response):
        # Call embed and expect rate limit exception
        with pytest.raises(LLMMistake) as excinfo:
            anthropic_llm.embed(texts=["Test text"])

        # Verify error message indicates rate limiting
        assert "rate limit" in str(excinfo.value).lower()


def test_embed_cost_calculation(anthropic_llm):
    """Test cost calculation for embeddings."""
    # Mock the API response with token usage
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
    }
    mock_response.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_response):
        # Call embed
        _, usage = anthropic_llm.embed(texts=["Test text"])

        # Get the cost rate for the model
        model = anthropic_llm.DEFAULT_EMBEDDING_MODEL
        cost_per_token = anthropic_llm.embedding_costs.get(
            model, 0.0001
        )  # Default if not found

        # Calculate expected cost
        expected_cost = 10 * cost_per_token

        # Verify cost calculation
        assert abs(usage["total_cost"] - expected_cost) < 0.000001


def test_embed_with_truncation(anthropic_llm):
    """Test embedding with truncation for long texts."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "usage": {"input_tokens": 8192, "output_tokens": 0, "total_tokens": 8192},
    }
    mock_response.raise_for_status.return_value = None

    # Create a very long text that would exceed token limits
    long_text = "This is a test. " * 10000  # Very long text

    with patch("requests.post", return_value=mock_response) as mock_post:
        # Call embed with the long text
        embeddings, usage = anthropic_llm.embed(texts=[long_text])

        # Verify API was called with truncation parameter
        call_args = mock_post.call_args[1]
        request_body = call_args["json"]
        assert request_body.get("truncate") == True

        # Verify we got results despite the length
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]
