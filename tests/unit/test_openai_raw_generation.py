"""
Tests for OpenAI provider raw_generate method.

This module tests the raw_generate method of the OpenAI provider, which is
the core method for generating text from the model.
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


def test_basic_raw_generate(openai_llm):
    """Test basic text generation without tools."""
    # Set up test data
    event_id = "test-event-123"
    system_prompt = "You are a helpful assistant."
    messages = [{"message_type": "human", "message": "Tell me a joke."}]

    # Create mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Why did the chicken cross the road?"))
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=8, total_tokens=18
    )

    # Mock the API call
    openai_llm.client.chat.completions.create.return_value = mock_response

    # Call the raw_generate method
    response_text, usage_stats = openai_llm._raw_generate(
        event_id=event_id,
        system_prompt=system_prompt,
        messages=messages,
        max_tokens=100,
    )

    # Verify the API was called with correct parameters
    openai_llm.client.chat.completions.create.assert_called_once()
    call_args = openai_llm.client.chat.completions.create.call_args[1]

    # Check model and messages were passed correctly
    assert call_args["model"] == "gpt-4o"
    assert (
        len(call_args["messages"]) == 2
    )  # One for system prompt, one for user message

    # Verify the response
    assert response_text == "Why did the chicken cross the road?"

    # Verify usage stats
    assert usage_stats["read_tokens"] == 10
    assert usage_stats["write_tokens"] == 8
    assert usage_stats["total_tokens"] == 18
    assert "read_cost" in usage_stats
    assert "write_cost" in usage_stats
    assert "total_cost" in usage_stats


def test_raw_generate_without_tools(openai_llm):
    """Test text generation with no tool use."""
    # Set up test data
    event_id = "test-event-tools"
    system_prompt = "You are a helpful assistant that uses tools."
    messages = [{"message_type": "human", "message": "What's 2+2?"}]

    # Create mock response with standard text content
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="The answer is 4."))]
    mock_response.usage = MagicMock(
        prompt_tokens=15, completion_tokens=10, total_tokens=25
    )

    # Mock the API call
    openai_llm.client.chat.completions.create.return_value = mock_response

    # Call the raw_generate method
    response_text, usage_stats = openai_llm._raw_generate(
        event_id=event_id, system_prompt=system_prompt, messages=messages
    )

    # Verify API call
    openai_llm.client.chat.completions.create.assert_called_once()

    # Verify response
    assert response_text == "The answer is 4."

    # Verify usage stats
    assert usage_stats["read_tokens"] == 15
    assert usage_stats["write_tokens"] == 10
    assert usage_stats["total_tokens"] == 25


def test_raw_generate_with_reasoning_model(openai_llm):
    """Test generation with a reasoning model (o1, o3-mini)."""
    # Change model to reasoning model
    openai_llm.model_name = "o1"

    # Set up test data
    event_id = "test-event-reasoning"
    system_prompt = "You are a helpful assistant."
    messages = [{"message_type": "human", "message": "Solve this complex problem."}]

    # Create mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Detailed step-by-step solution"))
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=20, total_tokens=30
    )

    # Mock the API call
    openai_llm.client.chat.completions.create.return_value = mock_response

    # Call the raw_generate method
    response_text, usage_stats = openai_llm._raw_generate(
        event_id=event_id, system_prompt=system_prompt, messages=messages
    )

    # Verify API call included reasoning_effort parameter
    openai_llm.client.chat.completions.create.assert_called_once()
    call_args = openai_llm.client.chat.completions.create.call_args[1]
    assert "reasoning_effort" in call_args
    assert call_args["reasoning_effort"] == "high"

    # Verify response text
    assert response_text == "Detailed step-by-step solution"

    # Verify usage stats
    assert usage_stats["read_tokens"] == 10
    assert usage_stats["write_tokens"] == 20
    assert usage_stats["total_tokens"] == 30


def test_raw_generate_error_handling(openai_llm):
    """Test error handling in raw_generate method."""
    # Set up test data
    event_id = "test-event-error"
    system_prompt = "You are a helpful assistant."
    messages = [{"message_type": "human", "message": "Tell me a joke."}]

    # Mock API error
    openai_llm.client.chat.completions.create.side_effect = Exception(
        "API error: rate limit exceeded"
    )

    # Call the raw_generate method and expect exception
    with pytest.raises(Exception) as excinfo:
        openai_llm._raw_generate(
            event_id=event_id, system_prompt=system_prompt, messages=messages
        )

    # Verify error message is passed through
    assert "API error: rate limit exceeded" in str(excinfo.value)


def test_raw_generate_with_null_content(openai_llm):
    """Test handling null content in response."""
    # Set up test data
    event_id = "test-event-null"
    system_prompt = "You are a helpful assistant."
    messages = [{"message_type": "human", "message": "Tell me a joke."}]

    # Create mock response with null content
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=None))]
    mock_response.usage = MagicMock(
        prompt_tokens=10, completion_tokens=0, total_tokens=10
    )

    # Mock the API call
    openai_llm.client.chat.completions.create.return_value = mock_response

    # Call the raw_generate method
    response_text, usage_stats = openai_llm._raw_generate(
        event_id=event_id, system_prompt=system_prompt, messages=messages
    )

    # Verify the API was called
    openai_llm.client.chat.completions.create.assert_called_once()

    # Verify response with null content is handled properly
    assert response_text == ""

    # Verify usage stats
    assert usage_stats["read_tokens"] == 10
    assert usage_stats["write_tokens"] == 0
    assert usage_stats["total_tokens"] == 10
