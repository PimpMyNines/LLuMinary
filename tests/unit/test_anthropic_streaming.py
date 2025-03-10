"""
Unit tests for Anthropic streaming functionality.

This module tests the streaming capabilities of the Anthropic provider.
"""

from unittest.mock import MagicMock, patch

import pytest
from lluminary.exceptions import RateLimitError, ServiceUnavailableError
from lluminary.models.providers.anthropic import AnthropicLLM


@pytest.fixture
def anthropic_llm():
    """Fixture for Anthropic LLM instance."""
    with patch("anthropic.Anthropic"):
        # Create the LLM instance with mock API key
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Ensure config exists
        llm.config = {"api_key": "test-key"}

        # Ensure client is initialized
        llm.client = MagicMock()

        yield llm


def test_basic_streaming(anthropic_llm):
    """Test basic streaming functionality."""
    # Skip the test if anthropic package isn't available
    pytest.importorskip("anthropic")

    # Configure mock streaming response
    mock_stream_gen = [
        # First chunk
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "Hello, "}),
                "type": "content_block_delta",
            },
        ),
        # Second chunk
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "world!"}),
                "type": "content_block_delta",
            },
        ),
        # Message delta with usage info
        type(
            "MockChunk",
            (),
            {
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "type": "message_delta",
            },
        ),
    ]

    # Setup the messages.create mock to return our stream generator
    anthropic_llm.client.messages.create.return_value = mock_stream_gen

    # Create a handler to collect chunks
    collected_chunks = []
    usage_data = None

    def callback(chunk, usage):
        nonlocal usage_data
        if chunk:
            collected_chunks.append(chunk)
        else:
            usage_data = usage

    # Call stream_generate
    result_generator = anthropic_llm.stream_generate(
        event_id="test-stream",
        system_prompt="You are a helpful assistant.",
        messages=[{"message_type": "human", "message": "Say hello world"}],
        callback=callback,
    )

    # Process each chunk from the generator
    chunks_from_gen = []
    final_usage = None

    for chunk, usage in result_generator:
        if chunk:
            chunks_from_gen.append(chunk)
        else:
            final_usage = usage

    # Verify chunks received through generator
    assert chunks_from_gen == ["Hello, ", "world!"]

    # Verify chunks received through callback
    assert collected_chunks == ["Hello, ", "world!"]

    # Verify usage information
    assert final_usage is not None
    assert "total_tokens" in final_usage
    assert final_usage["read_tokens"] == 10
    assert final_usage["write_tokens"] == 5
    assert final_usage["total_tokens"] == 15
    assert "total_cost" in final_usage


def test_streaming_with_tool_use(anthropic_llm):
    """Test streaming with tool use."""
    # Skip the test if anthropic package isn't available
    pytest.importorskip("anthropic")

    # Configure mock streaming response with tool use
    mock_stream_gen = [
        # Text chunk
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "I will calculate that for you."}),
                "type": "content_block_delta",
            },
        ),
        # Tool use chunk (partial)
        type(
            "MockChunk",
            (),
            {
                "delta": type(
                    "Delta",
                    (),
                    {
                        "tool_use": type(
                            "ToolUse",
                            (),
                            {
                                "id": "tool_123",
                                "name": "calculate",
                                "input": {"expression": "5 + "},
                            },
                        )
                    },
                ),
                "type": "tool_use_delta",
            },
        ),
        # Tool use chunk (completion)
        type(
            "MockChunk",
            (),
            {
                "delta": type(
                    "Delta",
                    (),
                    {"tool_use": type("ToolUse", (), {"input": {"expression": "7"}})},
                ),
                "type": "tool_use_delta",
            },
        ),
        # Usage info
        type(
            "MockChunk",
            (),
            {
                "usage": {"input_tokens": 12, "output_tokens": 8},
                "type": "message_delta",
            },
        ),
    ]

    # Setup the messages.create mock to return our stream generator
    anthropic_llm.client.messages.create.return_value = mock_stream_gen

    # Define test function
    def calculate(expression: str) -> float:
        """Calculate a mathematical expression."""
        return eval(expression)

    # Create a handler to collect tool calls
    collected_tool_calls = []

    def callback(chunk, usage, tool_calls=None):
        if tool_calls:
            collected_tool_calls.extend(tool_calls)

    # Call stream_generate
    chunks = []
    for chunk, usage in anthropic_llm.stream_generate(
        event_id="test-stream-tool",
        system_prompt="You are a helpful assistant.",
        messages=[{"message_type": "human", "message": "Calculate 5 + 7"}],
        functions=[calculate],
        callback=callback,
    ):
        if chunk:
            chunks.append(chunk)

    # Verify text chunks
    assert "I will calculate that for you." in chunks

    # Verify tool calls
    assert len(collected_tool_calls) > 0
    assert collected_tool_calls[0]["name"] == "calculate"
    assert collected_tool_calls[0]["arguments"]["expression"] == "5 + 7"

    # Verify client was called correctly
    call_kwargs = anthropic_llm.client.messages.create.call_args[1]
    assert call_kwargs["stream"] is True
    assert any(t["function"]["name"] == "calculate" for t in call_kwargs["tools"])


def test_streaming_with_thinking(anthropic_llm):
    """Test streaming with thinking capability."""
    # Skip the test if anthropic package isn't available
    pytest.importorskip("anthropic")

    # Use a model that supports thinking
    anthropic_llm.model_name = "claude-3-7-sonnet-20250219"

    # Configure mock streaming response with thinking
    mock_stream_gen = [
        # Thinking content
        type(
            "MockChunk",
            (),
            {
                "delta": type(
                    "Delta",
                    (),
                    {
                        "thinking": "Let me work through this step by step...",
                        "signature": "abc123",
                    },
                ),
                "type": "thinking_delta",
            },
        ),
        # Text chunk
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "Based on my analysis, "}),
                "type": "content_block_delta",
            },
        ),
        # Text chunk
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "the answer is 42."}),
                "type": "content_block_delta",
            },
        ),
        # Usage info
        type(
            "MockChunk",
            (),
            {
                "usage": {"input_tokens": 15, "output_tokens": 10},
                "type": "message_delta",
            },
        ),
    ]

    # Setup the messages.create mock to return our stream generator
    anthropic_llm.client.messages.create.return_value = mock_stream_gen

    # Create a handler to collect thinking content
    thinking_captured = None

    def callback(chunk, usage, thinking=None):
        nonlocal thinking_captured
        if thinking:
            thinking_captured = thinking

    # Call stream_generate with thinking budget
    result = None
    for _, result in anthropic_llm.stream_generate(
        event_id="test-stream-thinking",
        system_prompt="You are a helpful assistant.",
        messages=[{"message_type": "human", "message": "What is the meaning of life?"}],
        thinking_budget=1000,
        callback=callback,
    ):
        pass  # Process all chunks

    # Verify thinking content was captured
    assert thinking_captured is not None
    assert "step by step" in thinking_captured["thinking"]
    assert thinking_captured["signature"] == "abc123"

    # Verify client was called with thinking parameters
    call_kwargs = anthropic_llm.client.messages.create.call_args[1]
    assert "thinking" in call_kwargs


def test_streaming_with_retry(anthropic_llm):
    """Test streaming with retry on transient error."""
    # Skip the test if anthropic package isn't available
    anthropic = pytest.importorskip("anthropic")

    # Configure client.messages.create to fail first with rate limit, then succeed
    rate_limit_error = anthropic.APIError(
        message="Rate limit exceeded", http_status=429, header={"retry-after": "1"}
    )
    rate_limit_error.status_code = 429

    # Success response after retry
    mock_stream_gen = [
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "Response after retry"}),
                "type": "content_block_delta",
            },
        ),
        type(
            "MockChunk",
            (),
            {
                "usage": {"input_tokens": 10, "output_tokens": 5},
                "type": "message_delta",
            },
        ),
    ]

    # Set up the sequence of responses: first error, then success
    anthropic_llm.client.messages.create.side_effect = [
        rate_limit_error,
        mock_stream_gen,
    ]

    # Call stream_generate (should retry after rate limit)
    with patch("time.sleep") as mock_sleep:  # Avoid actual sleep in tests
        chunks = []
        for chunk, _ in anthropic_llm.stream_generate(
            event_id="test-stream-retry",
            system_prompt="You are a helpful assistant.",
            messages=[{"message_type": "human", "message": "Test retry"}],
            max_retries=3,
        ):
            if chunk:
                chunks.append(chunk)

        # Verify retry behavior
        assert mock_sleep.call_count == 1
        assert anthropic_llm.client.messages.create.call_count == 2

        # Verify response after retry
        assert "Response after retry" in chunks


def test_streaming_error_handling(anthropic_llm):
    """Test error handling during streaming."""
    # Skip the test if anthropic package isn't available
    anthropic = pytest.importorskip("anthropic")

    # Configure client.messages.create to raise various errors
    errors_to_test = [
        # Rate limit error
        (
            anthropic.APIError(
                message="Rate limit exceeded", http_status=429, header={}
            ),
            RateLimitError,
        ),
        # Service unavailable
        (
            anthropic.APIError(
                message="Service temporarily unavailable", http_status=503, header={}
            ),
            ServiceUnavailableError,
        ),
        # Authentication error
        (
            anthropic.APIError(message="Invalid API key", http_status=401, header={}),
            Exception,  # Actual error type depends on mapping
        ),
    ]

    for api_error, expected_error_type in errors_to_test:
        # Set status code (needed for mapping)
        api_error.status_code = api_error.http_status

        # Configure the client to raise this error
        anthropic_llm.client.messages.create.side_effect = api_error

        # Call stream_generate and expect the mapped error
        with pytest.raises(expected_error_type):
            for _ in anthropic_llm.stream_generate(
                event_id=f"test-error-{api_error.http_status}",
                system_prompt="You are a helpful assistant.",
                messages=[{"message_type": "human", "message": "Test error handling"}],
                max_retries=0,  # Disable retries for error testing
            ):
                pass


def test_streaming_with_callbacks(anthropic_llm):
    """Test streaming with complex callback handling."""
    # Skip the test if anthropic package isn't available
    pytest.importorskip("anthropic")

    # Configure mock streaming response with multiple chunks
    mock_stream_gen = [
        # First chunk
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "This "}),
                "type": "content_block_delta",
            },
        ),
        # Second chunk
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "is "}),
                "type": "content_block_delta",
            },
        ),
        # Third chunk
        type(
            "MockChunk",
            (),
            {"delta": type("Delta", (), {"text": "a "}), "type": "content_block_delta"},
        ),
        # Fourth chunk
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "test."}),
                "type": "content_block_delta",
            },
        ),
        # Usage info
        type(
            "MockChunk",
            (),
            {
                "usage": {"input_tokens": 10, "output_tokens": 4},
                "type": "message_delta",
            },
        ),
    ]

    # Setup the messages.create mock to return our stream generator
    anthropic_llm.client.messages.create.return_value = mock_stream_gen

    # Tracking variables for callback
    accumulated_text = ""
    callback_call_count = 0
    last_usage = None

    # Callback that tracks more information
    def advanced_callback(chunk, usage):
        nonlocal accumulated_text, callback_call_count, last_usage
        callback_call_count += 1

        if chunk:
            accumulated_text += chunk

        last_usage = usage

    # Call stream_generate with callback
    final_result = None
    for _, final_result in anthropic_llm.stream_generate(
        event_id="test-streaming-callback",
        system_prompt="You are a helpful assistant.",
        messages=[{"message_type": "human", "message": "Say something for a test"}],
        callback=advanced_callback,
    ):
        pass  # Process all chunks

    # Verify callback behavior
    assert callback_call_count == 5  # 4 text chunks + 1 completion notification
    assert accumulated_text == "This is a test."
    assert last_usage is not None
    assert last_usage["read_tokens"] == 10
    assert last_usage["write_tokens"] == 4
    assert last_usage["total_tokens"] == 14
    assert last_usage["is_complete"] is True

    # Verify final result
    assert final_result is not None
    assert final_result["total_tokens"] == 14
    assert final_result["total_cost"] > 0


def test_streaming_large_output(anthropic_llm):
    """Test streaming with a large output that comes in many chunks."""
    # Skip the test if anthropic package isn't available
    pytest.importorskip("anthropic")

    # Create a large number of small chunks to simulate a large response
    num_chunks = 100
    chunk_text = "word "

    # Generate mock response chunks
    mock_stream_gen = []

    # Add text chunks
    for _ in range(num_chunks):
        mock_stream_gen.append(
            type(
                "MockChunk",
                (),
                {
                    "delta": type("Delta", (), {"text": chunk_text}),
                    "type": "content_block_delta",
                },
            )
        )

    # Add final usage info
    mock_stream_gen.append(
        type(
            "MockChunk",
            (),
            {
                "usage": {"input_tokens": 20, "output_tokens": 200},
                "type": "message_delta",
            },
        )
    )

    # Setup the messages.create mock to return our stream generator
    anthropic_llm.client.messages.create.return_value = mock_stream_gen

    # Call stream_generate
    chunks = []
    for chunk, _ in anthropic_llm.stream_generate(
        event_id="test-large-output",
        system_prompt="You are a helpful assistant.",
        messages=[{"message_type": "human", "message": "Generate a long response"}],
        max_tokens=500,
    ):
        if chunk:
            chunks.append(chunk)

    # Verify all chunks were processed
    assert len(chunks) == num_chunks
    assert all(chunk == chunk_text for chunk in chunks)

    # Verify client was called with correct parameters
    call_kwargs = anthropic_llm.client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 500
    assert call_kwargs["stream"] is True


def test_streaming_with_images(anthropic_llm):
    """Test streaming with image input."""
    # Skip the test if anthropic package isn't available
    pytest.importorskip("anthropic")

    # Configure mock streaming response
    mock_stream_gen = [
        # Text chunk describing the image
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "I can see a cat in the image."}),
                "type": "content_block_delta",
            },
        ),
        # Usage info (higher input tokens due to image)
        type(
            "MockChunk",
            (),
            {
                "usage": {"input_tokens": 1000, "output_tokens": 10},
                "type": "message_delta",
            },
        ),
    ]

    # Setup the messages.create mock
    anthropic_llm.client.messages.create.return_value = mock_stream_gen

    # Mock image encoding
    with patch.object(anthropic_llm, "_encode_image", return_value="base64_image_data"):
        # Call stream_generate with image input
        with patch("os.path.exists", return_value=True):
            chunks = []
            final_usage = None
            for chunk, usage in anthropic_llm.stream_generate(
                event_id="test-image-stream",
                system_prompt="Describe what you see in the image.",
                messages=[
                    {
                        "message_type": "human",
                        "message": "What's in this image?",
                        "image_paths": ["/path/to/cat.jpg"],
                    }
                ],
            ):
                if chunk:
                    chunks.append(chunk)
                else:
                    final_usage = usage

            # Verify response content
            assert "".join(chunks) == "I can see a cat in the image."

            # Verify usage data includes image
            assert final_usage["images"] == 1
            assert "image_cost" in final_usage
            assert final_usage["image_cost"] > 0

            # Verify client was called with image content
            call_kwargs = anthropic_llm.client.messages.create.call_args[1]
            request_messages = call_kwargs["messages"]

            # Find user message
            user_message = next(m for m in request_messages if m["role"] == "user")

            # Verify image content is included
            image_parts = [p for p in user_message["content"] if p["type"] == "image"]
            assert len(image_parts) == 1
            assert image_parts[0]["source"]["data"] == "base64_image_data"


def test_streaming_with_system_prompt(anthropic_llm):
    """Test streaming with different system prompts."""
    # Skip the test if anthropic package isn't available
    pytest.importorskip("anthropic")

    # Configure mock streaming response
    mock_stream_gen = [
        # Text chunk
        type(
            "MockChunk",
            (),
            {
                "delta": type("Delta", (), {"text": "I am a helpful assistant."}),
                "type": "content_block_delta",
            },
        ),
        # Usage info
        type(
            "MockChunk",
            (),
            {
                "usage": {"input_tokens": 15, "output_tokens": 8},
                "type": "message_delta",
            },
        ),
    ]

    # Setup the messages.create mock
    anthropic_llm.client.messages.create.return_value = mock_stream_gen

    # Test system prompts
    system_prompts = [
        "You are a helpful assistant.",
        "You are a concise expert who always gives brief answers.",
        """You are a poetic assistant who speaks in rhymes.
        Try to make your answers rhyme whenever possible.""",
    ]

    for system_prompt in system_prompts:
        # Call stream_generate with different system prompts
        for _, _ in anthropic_llm.stream_generate(
            event_id=f"test-system-{hash(system_prompt)}",
            system_prompt=system_prompt,
            messages=[{"message_type": "human", "message": "Who are you?"}],
        ):
            pass

        # Verify the system prompt was passed correctly
        call_kwargs = anthropic_llm.client.messages.create.call_args[1]
        assert call_kwargs["system"] == system_prompt
