"""
Unit tests for the Anthropic provider implementation.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from lluminary.models.providers.anthropic import AnthropicLLM


@pytest.fixture
def anthropic_llm():
    """Fixture for Anthropic LLM instance."""
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


def test_supported_model_lists(anthropic_llm):
    """Test that the model lists are properly configured."""
    # Make sure appropriate lists are populated
    assert len(anthropic_llm.SUPPORTED_MODELS) > 0
    assert len(anthropic_llm.THINKING_MODELS) > 0

    # Check that core models are included
    assert "claude-3-5-sonnet-20241022" in anthropic_llm.SUPPORTED_MODELS
    assert "claude-3-7-sonnet-20250219" in anthropic_llm.THINKING_MODELS

    # Verify model list relationships
    # Thinking models should be a subset of supported models
    assert all(
        model in anthropic_llm.SUPPORTED_MODELS
        for model in anthropic_llm.THINKING_MODELS
    )

    # Verify that cost and context window data is properly configured for each model
    for model_name in anthropic_llm.SUPPORTED_MODELS:
        assert model_name in anthropic_llm.CONTEXT_WINDOW
        assert model_name in anthropic_llm.COST_PER_MODEL

        # Verify cost structure is correct
        model_costs = anthropic_llm.COST_PER_MODEL[model_name]
        assert "read_token" in model_costs
        assert "write_token" in model_costs

        # Verify image cost for Claude 3 models
        if "claude-3" in model_name:
            assert "image_cost" in model_costs

        # Verify values are of the expected type
        assert isinstance(model_costs["read_token"], (int, float))
        assert isinstance(model_costs["write_token"], (int, float))

        # Verify context window is a number
        assert isinstance(anthropic_llm.CONTEXT_WINDOW[model_name], int)


def test_anthropic_initialization(anthropic_llm):
    """Test Anthropic provider initialization."""
    # Verify basic initialization properties
    assert anthropic_llm.model_name == "claude-3-5-sonnet-20241022"
    assert anthropic_llm.config["api_key"] == "test-key"

    # Test validate_model
    assert anthropic_llm.validate_model("claude-3-5-sonnet-20241022") is True
    assert anthropic_llm.validate_model("invalid-model") is False


def test_message_formatting(anthropic_llm):
    """Test Anthropic message formatting."""
    # Test basic message
    messages = [{"message_type": "human", "message": "test message"}]
    formatted = anthropic_llm._format_messages_for_model(messages)
    assert formatted[0]["role"] == "user"
    assert isinstance(formatted[0]["content"], list)
    assert formatted[0]["content"][0]["type"] == "text"
    assert formatted[0]["content"][0]["text"] == "test message"

    # Test AI message
    messages = [{"message_type": "ai", "message": "assistant response"}]
    formatted = anthropic_llm._format_messages_for_model(messages)
    assert formatted[0]["role"] == "assistant"
    assert formatted[0]["content"][0]["type"] == "text"
    assert formatted[0]["content"][0]["text"] == "assistant response"


def test_convert_function_to_tool(anthropic_llm):
    """Test converting a function to a tool."""

    def test_func(param1: str, param2: int = 0) -> str:
        """Test function docstring"""
        return f"{param1} {param2}"

    tools = anthropic_llm._convert_functions_to_tools([test_func])

    assert len(tools) > 0
    assert "type" in tools[0]
    assert tools[0]["type"] == "function"
    assert "function" in tools[0]
    assert "name" in tools[0]["function"]
    assert tools[0]["function"]["name"] == "test_func"
    assert "description" in tools[0]["function"]
    assert "parameters" in tools[0]["function"]


def test_raw_generate(anthropic_llm):
    """Test Anthropic raw generation."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "test response"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response, usage = anthropic_llm._raw_generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "test"}],
            max_tokens=100,
        )

        assert response == "test response"
        assert "read_tokens" in usage
        assert "write_tokens" in usage
        assert "total_tokens" in usage
        assert "total_cost" in usage

        # Verify API call
        mock_post.assert_called_once()


def test_generate(anthropic_llm):
    """Test the generate method."""
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "test response"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response, usage, _ = anthropic_llm.generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "test"}],
            max_tokens=100,
        )

        assert response == "test response"
        assert "read_tokens" in usage
        assert "write_tokens" in usage
        assert "total_tokens" in usage
        assert "total_cost" in usage


def test_supports_image_input(anthropic_llm):
    """Test Anthropic image support check."""
    assert isinstance(anthropic_llm.supports_image_input(), bool)


def test_get_supported_models(anthropic_llm):
    """Test retrieving supported models."""
    models = anthropic_llm.get_supported_models()
    assert isinstance(models, list)
    assert "claude-3-5-sonnet-20241022" in models


def test_error_handling():
    """Test Anthropic error handling."""
    with patch("anthropic.Anthropic"), patch("requests.post") as mock_post:
        # Configure mock to simulate an API error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception(
            "API Error: Invalid API key"
        )
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_post.return_value = mock_response

        # Create LLM instance
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="invalid-key")

        # Try to generate and expect an exception
        with pytest.raises(Exception) as exc_info:
            llm.generate(
                event_id="test-event",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "test"}],
            )

        # Verify the error message
        assert "API Error" in str(exc_info.value)

    # Test rate limiting error
    with patch("anthropic.Anthropic"), patch("requests.post") as mock_post:
        # Configure mock to simulate a rate limit error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Rate limit exceeded")
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_post.return_value = mock_response

        # Create LLM instance
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Try to generate and expect an exception
        with pytest.raises(Exception) as exc_info:
            llm.generate(
                event_id="test-event",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "test"}],
            )

        # Verify the error message
        assert "Rate limit" in str(exc_info.value)


def test_image_handling():
    """Test Anthropic image handling."""
    with patch("anthropic.Anthropic"), patch("requests.post") as mock_post, patch(
        "requests.get"
    ) as mock_get, patch("PIL.Image.open") as mock_pil_open:
        # Configure mocks
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Image description"}],
            "usage": {"input_tokens": 15, "output_tokens": 5},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Mock image download response
        mock_image_response = MagicMock()
        mock_image_response.content = b"fake_image_data"
        mock_get.return_value = mock_image_response

        # Mock PIL image
        mock_image = MagicMock()
        mock_image.format = "JPEG"
        mock_image.size = (800, 600)
        mock_pil_open.return_value = mock_image

        # Create LLM instance
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Test message with image URL
        response, usage, _ = llm.generate(
            event_id="test-event",
            system_prompt="You are a helpful assistant",
            messages=[
                {
                    "message_type": "human",
                    "message": "What's in this image?",
                    "image_urls": ["http://example.com/image.jpg"],
                }
            ],
        )

        # Verify response
        assert response == "Image description"

        # Verify the request was made with image content
        call_args = mock_post.call_args[1]
        request_body = json.loads(call_args["data"])

        # Extract the content parts from the request
        user_message = next(m for m in request_body["messages"] if m["role"] == "user")
        content_parts = user_message["content"]

        # Verify there's a text part
        text_part = next(
            (part for part in content_parts if part["type"] == "text"), None
        )
        assert text_part is not None
        assert "What's in this image?" in text_part["text"]

        # Verify there's an image part
        image_part = next(
            (part for part in content_parts if part["type"] == "image"), None
        )
        assert image_part is not None
        assert "source" in image_part
        assert "media_type" in image_part["source"]
        assert "data" in image_part["source"]


def test_tool_handling():
    """Test Anthropic tool handling."""

    # Define a test tool
    def test_tool(x: int, y: int) -> int:
        """A test tool that adds two numbers."""
        return x + y

    with patch("anthropic.Anthropic"), patch("requests.post") as mock_post:
        # Configure mock response with tool calls
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": "I will add those numbers"},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "test_tool",
                    "input": {"x": 5, "y": 7},
                },
            ],
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Create LLM instance
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Generate with tools
        response, usage, tool_calls = llm.generate(
            event_id="test-event",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "Add 5 and 7"}],
            functions=[test_tool],
        )

        # Verify response
        assert "add those numbers" in response

        # Verify tool calls
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "test_tool"
        assert tool_calls[0]["arguments"]["x"] == 5
        assert tool_calls[0]["arguments"]["y"] == 7

        # Verify the request was made with tool definitions
        call_args = mock_post.call_args[1]
        request_body = json.loads(call_args["data"])

        # Verify tools were included in the request
        assert "tools" in request_body
        assert len(request_body["tools"]) == 1
        assert request_body["tools"][0]["function"]["name"] == "test_tool"


def test_streaming():
    """Test Anthropic streaming capability."""

    class MockStreamingResponse:
        """Mock for Anthropic streaming response."""

        def __init__(self):
            self.chunks = [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"text": "Hello, "},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"text": "world!"},
                },
                {"type": "content_block_stop", "index": 0},
                {"type": "message_stop"},
                {
                    "type": "message_delta",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            ]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index < len(self.chunks):
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk
            raise StopIteration

    with patch("anthropic.Anthropic"), patch("requests.post") as mock_post:
        # Configure mock to return a streaming response
        mock_post.return_value.iter_lines.return_value = [
            json.dumps(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                }
            ).encode(),
            json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"text": "Hello, "},
                }
            ).encode(),
            json.dumps(
                {"type": "content_block_delta", "index": 0, "delta": {"text": "world!"}}
            ).encode(),
            json.dumps({"type": "content_block_stop", "index": 0}).encode(),
            json.dumps({"type": "message_stop"}).encode(),
            json.dumps(
                {
                    "type": "message_delta",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            ).encode(),
        ]

        # Create LLM instance
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Prepare callback to collect chunks
        chunks = []

        def callback(chunk, finish_reason=None):
            chunks.append(chunk)

        # Call stream_generate
        result = llm.stream_generate(
            event_id="test-event",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "Say hello world"}],
            callback=callback,
        )

        # Verify chunks were received
        assert len(chunks) == 2
        assert chunks[0] == "Hello, "
        assert chunks[1] == "world!"

        # Verify final result
        assert result["total_response"] == "Hello, world!"
        assert result["total_tokens"] == 15
        assert result["read_tokens"] == 10
        assert result["write_tokens"] == 5
        assert "total_cost" in result


def test_streaming_with_tools():
    """Test Anthropic streaming with tool calls."""

    def test_tool(x: int, y: int) -> int:
        """A test tool that adds two numbers."""
        return x + y

    with patch("anthropic.Anthropic"), patch("requests.post") as mock_post:
        # Configure mock to return a streaming response with tool use
        mock_post.return_value.iter_lines.return_value = [
            json.dumps(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                }
            ).encode(),
            json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"text": "I'll calculate that for you."},
                }
            ).encode(),
            json.dumps({"type": "content_block_stop", "index": 0}).encode(),
            json.dumps(
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "test_tool",
                        "input": {"x": 5, "y": 7},
                    },
                }
            ).encode(),
            json.dumps({"type": "content_block_stop", "index": 1}).encode(),
            json.dumps({"type": "message_stop"}).encode(),
            json.dumps(
                {
                    "type": "message_delta",
                    "usage": {"input_tokens": 12, "output_tokens": 8},
                }
            ).encode(),
        ]

        # Create LLM instance
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Prepare callback to collect chunks and tool calls
        chunks = []
        tool_calls_received = []

        def callback(chunk, finish_reason=None, tool_calls=None):
            if chunk:
                chunks.append(chunk)
            if tool_calls:
                tool_calls_received.extend(tool_calls)

        # Call stream_generate with tools
        result = llm.stream_generate(
            event_id="test-event",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "Add 5 and 7"}],
            callback=callback,
            functions=[test_tool],
        )

        # Verify chunks were received
        assert len(chunks) == 1
        assert "calculate that for you" in chunks[0]

        # Verify tool calls were received
        assert len(tool_calls_received) == 1
        assert tool_calls_received[0]["name"] == "test_tool"
        assert tool_calls_received[0]["arguments"]["x"] == 5
        assert tool_calls_received[0]["arguments"]["y"] == 7

        # Verify final result
        assert "calculate that for you" in result["total_response"]
        assert "tool_calls" in result
        assert result["tool_calls"][0]["name"] == "test_tool"


def test_classification():
    """Test Anthropic classification support."""
    # Define test categories and messages
    categories = {
        "positive": "Content with a positive sentiment",
        "negative": "Content with a negative sentiment",
        "neutral": "Content with a neutral sentiment",
    }

    messages = [{"message_type": "human", "message": "I love this product!"}]

    # Mock the generate method
    with patch.object(AnthropicLLM, "generate") as mock_generate:
        # Set up mock to return classification result
        mock_generate.return_value = (
            ["positive"],
            {
                "read_tokens": 15,
                "write_tokens": 5,
                "total_tokens": 20,
                "total_cost": 0.0002,
            },
            None,
        )

        # Create LLM instance
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Perform classification
        result, usage = llm.classify(messages=messages, categories=categories)

        # Verify result
        assert "positive" in result
        assert usage["total_tokens"] == 20
        assert usage["total_cost"] == 0.0002

        # Verify generate was called with classification prompt
        call_args = mock_generate.call_args[1]
        assert "system_prompt" in call_args
        assert "categories" in call_args["system_prompt"]
