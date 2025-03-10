"""
Unit tests for the Anthropic provider thinking budget functionality.

This module tests the thinking budget features available in Claude 3.7+ models.
"""

from unittest.mock import MagicMock, patch

import pytest
from lluminary.exceptions import ThinkingError
from lluminary.models.providers.anthropic import AnthropicLLM


@pytest.fixture
def anthropic_llm():
    """Fixture for Anthropic LLM instance with a thinking-capable model."""
    with patch("anthropic.Anthropic") as mock_anthropic, patch(
        "requests.post"
    ) as mock_post:
        # Configure mock response for requests.post
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "test response"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Create the LLM instance with thinking-capable model
        llm = AnthropicLLM("claude-3-7-sonnet-20250219", api_key="test-key")

        # Ensure client is initialized
        llm.client = MagicMock()

        # Ensure config exists
        if not hasattr(llm, "config"):
            llm.config = {}

        # Add client to config as expected by implementation
        llm.config["client"] = llm.client
        llm.config["api_key"] = "test-key"

        yield llm


def test_thinking_model_detection(anthropic_llm):
    """Test detection of thinking capability in models."""
    # Test with a thinking-capable model
    assert anthropic_llm.is_thinking_model("claude-3-7-sonnet-20250219") is True

    # Test with a non-thinking model
    assert anthropic_llm.is_thinking_model("claude-3-5-sonnet-20241022") is False

    # Verify THINKING_MODELS list contains expected models
    assert "claude-3-7-sonnet-20250219" in anthropic_llm.THINKING_MODELS
    assert "claude-3-7-opus-20250619" in anthropic_llm.THINKING_MODELS


def test_thinking_budget_generation(anthropic_llm):
    """Test generation with thinking budget."""
    # Mock the response with thinking content
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "content": [
            {
                "type": "thinking",
                "thinking": "Let me analyze this problem step by step...",
                "thinking_signature": "ef8356b9",
            },
            {"type": "text", "text": "Based on my analysis, the answer is 42."},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 15},
    }
    mock_response.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_response) as mock_post:
        # Generate with thinking budget
        response, usage, messages = anthropic_llm.generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "What is 6 × 7?"}],
            thinking_budget=1000,
        )

        # Verify API was called with thinking parameter
        call_args = mock_post.call_args[1]
        request_body = json.loads(call_args["data"])

        # Ensure thinking is enabled in the request
        assert request_body["thinking"] == "enabled"

        # Verify response contains both thinking and answer
        assert "the answer is 42" in response

        # Verify thinking was added to messages
        thinking_message = next((m for m in messages if "thinking" in m), None)
        assert thinking_message is not None
        assert (
            "Let me analyze this problem step by step"
            in thinking_message["thinking"]["thinking"]
        )
        assert thinking_message["thinking"]["thinking_signature"] == "ef8356b9"


def test_thinking_budget_with_non_thinking_model(anthropic_llm):
    """Test behavior when thinking budget is set with a non-thinking model."""
    # Change model to non-thinking model
    anthropic_llm.model_name = "claude-3-5-sonnet-20241022"

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Regular response without thinking"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Generate with thinking budget but non-thinking model
        response, usage, messages = anthropic_llm.generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "What is 6 × 7?"}],
            thinking_budget=1000,
        )

        # Verify API was called without thinking parameter
        call_args = mock_post.call_args[1]
        request_body = json.loads(call_args["data"])

        # Ensure thinking is not enabled in the request
        assert "thinking" not in request_body

        # Verify normal response was returned
        assert "Regular response without thinking" in response

        # Verify no thinking was added to messages
        thinking_message = next((m for m in messages if "thinking" in m), None)
        assert thinking_message is None

    # Change back to thinking model for other tests
    anthropic_llm.model_name = "claude-3-7-sonnet-20250219"


def test_thinking_budget_limits(anthropic_llm):
    """Test handling of thinking budget limits."""
    # Test with different thinking budget values
    test_cases = [
        # (budget, expected_thinking_value)
        (100, "enabled"),  # Low but valid budget
        (5000, "enabled"),  # Medium budget
        (50000, "enabled"),  # High budget
        (0, "disabled"),  # Zero budget should disable thinking
        (None, "enabled"),  # Default should enable thinking
    ]

    for budget, expected_value in test_cases:
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "content": [{"type": "text", "text": "Test response"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            # Generate with the specified thinking budget
            anthropic_llm.generate(
                event_id=f"test-{budget}",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "What is 6 × 7?"}],
                thinking_budget=budget,
            )

            # Verify the request
            call_args = mock_post.call_args[1]
            request_body = json.loads(call_args["data"])

            # Check thinking value or absence
            if expected_value == "disabled":
                if "thinking" in request_body:
                    assert request_body["thinking"] == "disabled"
            else:
                assert request_body["thinking"] == "enabled"


def test_thinking_streaming(anthropic_llm):
    """Test streaming with thinking budget."""
    # Setup streaming response chunks
    stream_chunks = [
        # Thinking content
        json.dumps(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "Let me think about this...",
                    "thinking_signature": "abc123",
                },
            }
        ).encode(),
        json.dumps({"type": "content_block_stop", "index": 0}).encode(),
        # Text content in chunks
        json.dumps(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text"},
            }
        ).encode(),
        json.dumps(
            {"type": "content_block_delta", "index": 1, "delta": {"text": "First "}}
        ).encode(),
        json.dumps(
            {"type": "content_block_delta", "index": 1, "delta": {"text": "chunk of "}}
        ).encode(),
        json.dumps(
            {"type": "content_block_delta", "index": 1, "delta": {"text": "text."}}
        ).encode(),
        json.dumps({"type": "content_block_stop", "index": 1}).encode(),
        # End of message and usage
        json.dumps({"type": "message_stop"}).encode(),
        json.dumps(
            {"type": "message_delta", "usage": {"input_tokens": 10, "output_tokens": 8}}
        ).encode(),
    ]

    with patch("requests.post") as mock_post:
        # Configure mock to return streaming response
        mock_post.return_value.iter_lines.return_value = stream_chunks

        # Collected chunks and thinking content
        collected_chunks = []
        thinking_content = None

        def test_callback(chunk, finish_reason=None, thinking=None):
            nonlocal thinking_content
            if chunk:
                collected_chunks.append(chunk)
            if thinking:
                thinking_content = thinking

        # Call stream_generate with thinking budget
        result = anthropic_llm.stream_generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "What is 6 × 7?"}],
            callback=test_callback,
            thinking_budget=1000,
        )

        # Verify API was called with thinking parameter
        call_args = mock_post.call_args[1]
        request_headers = call_args["headers"]
        assert "anthropic-thinking" in json.dumps(request_headers).lower()

        # Verify text chunks were received
        assert len(collected_chunks) == 3
        assert "".join(collected_chunks) == "First chunk of text."

        # Verify thinking content was received
        assert thinking_content is not None
        assert "Let me think about this" in thinking_content["thinking"]
        assert thinking_content["thinking_signature"] == "abc123"

        # Verify final result contains thinking
        assert "thinking_content" in result
        assert result["thinking_content"]["thinking"] == "Let me think about this..."
        assert result["total_response"] == "First chunk of text."


def test_thinking_error_handling(anthropic_llm):
    """Test error handling related to thinking budget."""
    # Test invalid thinking budget (negative)
    with pytest.raises(ThinkingError) as excinfo:
        anthropic_llm._raw_generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "What is 6 × 7?"}],
            thinking_budget=-100,
        )
    assert "thinking budget" in str(excinfo.value).lower()
    assert "negative" in str(excinfo.value).lower()

    # Test using thinking with unsupported model
    anthropic_llm.model_name = "claude-instant-1.2"  # Very old model

    # API error when using thinking with unsupported model
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.raise_for_status.side_effect = Exception(
        "Thinking is not supported by this model"
    )
    mock_response.json.return_value = {
        "error": {"message": "Thinking is not supported by this model"}
    }

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(Exception) as excinfo:
            anthropic_llm._raw_generate(
                event_id="test",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "What is 6 × 7?"}],
                thinking_budget=1000,
            )
        assert "thinking is not supported" in str(excinfo.value).lower()
