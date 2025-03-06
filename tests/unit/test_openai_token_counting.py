"""
Tests for OpenAI provider token counting and cost calculation.

This module focuses specifically on testing the token counting
and cost calculation functionality of the OpenAI provider.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.openai import OpenAILLM


@pytest.fixture
def openai_llm():
    """Fixture for OpenAI LLM instance."""
    with patch.object(OpenAILLM, "auth") as mock_auth, patch(
        "openai.OpenAI"
    ) as mock_openai:
        # Mock authentication to avoid API errors
        mock_auth.return_value = None

        # Create the LLM instance with mock API key
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Initialize client attribute directly for tests
        llm.client = MagicMock()

        # Ensure config exists
        if not hasattr(llm, "config"):
            llm.config = {}

        # Add client to config as expected by implementation
        llm.config["client"] = llm.client

        yield llm


def test_token_count_from_messages(openai_llm):
    """Test token counting from various message types."""
    # Test simple message
    simple_message = [{"role": "user", "content": "Hello world"}]
    count = openai_llm._count_tokens_from_messages(simple_message)
    assert count > 0
    assert isinstance(count, int)

    # Test multiple messages
    multiple_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "How can I help you?"},
    ]
    count = openai_llm._count_tokens_from_messages(multiple_messages)
    assert count > 0
    assert isinstance(count, int)

    # Test longer message
    long_message = [
        {
            "role": "user",
            "content": "This is a longer message that should use more tokens " * 10,
        }
    ]
    long_count = openai_llm._count_tokens_from_messages(long_message)
    assert long_count > count  # Should use more tokens than shorter messages


def test_estimate_cost_calculation(openai_llm):
    """Test cost estimation with various inputs."""
    # Test basic cost calculation
    read_tokens = 100
    write_tokens = 50
    model = "gpt-4o"

    cost = openai_llm._estimate_cost(model, read_tokens, write_tokens)

    # Calculate expected cost based on model pricing
    expected_cost = (
        read_tokens * openai_llm.COST_PER_MODEL[model]["read_token"]
        + write_tokens * openai_llm.COST_PER_MODEL[model]["write_token"]
    )

    assert abs(cost - expected_cost) < 0.000001  # Account for floating point precision

    # Test cost calculation with different models
    for model_name in openai_llm.SUPPORTED_MODELS:
        cost = openai_llm._estimate_cost(model_name, read_tokens, write_tokens)
        assert cost > 0  # Cost should be positive

        # Calculate expected cost based on model pricing
        expected_cost = (
            read_tokens * openai_llm.COST_PER_MODEL[model_name]["read_token"]
            + write_tokens * openai_llm.COST_PER_MODEL[model_name]["write_token"]
        )

        assert abs(cost - expected_cost) < 0.000001


def test_image_token_calculation(openai_llm):
    """Test image token calculation for different dimensions and detail levels."""
    # Test high detail image token calculation
    # 1024x768 image (should be multiple tiles)
    high_detail_tokens = openai_llm._calculate_image_tokens(1024, 768, "high")
    # Tokens should be positive
    assert high_detail_tokens > 0

    # Test low detail image token calculation
    low_detail_tokens = openai_llm._calculate_image_tokens(1024, 768, "low")
    # Low detail should use fewer tokens than high detail
    assert low_detail_tokens < high_detail_tokens

    # Test different detail levels
    assert openai_llm._calculate_image_tokens(
        800, 600, "high"
    ) > openai_llm._calculate_image_tokens(800, 600, "low")
