"""
Tests for OpenAI provider streaming functionality.

This module focuses specifically on testing the streaming
capabilities of the OpenAI provider, including chunk handling,
error recovery, and usage statistics calculation.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from lluminary.models.providers.openai import OpenAILLM


@pytest.fixture
def openai_llm():
    """Fixture for OpenAI LLM instance."""
    with patch.object(OpenAILLM, "auth") as mock_auth, patch(
        "openai.OpenAI"
    ) as _:  # We don't use this mock directly
        # Mock authentication
        mock_auth.return_value = None

        # Create LLM instance
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Mock client
        llm.client = MagicMock()
        llm.client.chat.completions.create = MagicMock()

        yield llm


class MockChunk:
    """Mock class for OpenAI streaming response chunks."""

    def __init__(self, content=None, tool_calls=None, is_empty=False):
        """Initialize mock chunk with content and/or tool calls."""
        self.choices = []
        if not is_empty:
            delta = MagicMock()
            if content is not None:
                delta.content = content
            if tool_calls is not None:
                delta.tool_calls = tool_calls
            choice = MagicMock(delta=delta)
            self.choices = [choice]


class TestOpenAIStreamGenerate:
    """Tests for streaming generation functionality."""

    def test_basic_streaming(self, openai_llm):
        """Test basic streaming functionality."""
        # Create a sequence of chunks
        chunks = [
            MockChunk(content="Hello"),
            MockChunk(content=", "),
            MockChunk(content="world"),
            MockChunk(content="!"),
            MockChunk(is_empty=True),  # End of stream
        ]

        # Mock the create method to return our stream
        openai_llm.client.chat.completions.create.return_value = chunks

        # Create callback to track streaming
        callback_data = []

        def callback(text, stats):
            callback_data.append((text, stats))

        # Call stream_generate
        collected_chunks = []
        for chunk, stats in openai_llm.stream_generate(
            event_id="test-stream",
            system_prompt="You are a test assistant",
            messages=[{"message_type": "human", "message": "Say hello"}],
            callback=callback,
        ):
            collected_chunks.append((chunk, stats))

        # Verify chunks were collected
        assert len(collected_chunks) == 5  # 4 content chunks + end marker

        # Verify content of chunks
        assert collected_chunks[0][0] == "Hello"
        assert collected_chunks[1][0] == ", "
        assert collected_chunks[2][0] == "world"
        assert collected_chunks[3][0] == "!"
        assert collected_chunks[4][0] == ""  # End marker is empty string

        # Verify stats in last chunk
        final_stats = collected_chunks[-1][1]
        assert final_stats["is_complete"] is True
        assert "read_tokens" in final_stats
        assert "write_tokens" in final_stats
        assert "total_tokens" in final_stats
        assert "read_cost" in final_stats
        assert "write_cost" in final_stats
        assert "total_cost" in final_stats

        # Verify callback was called
        assert len(callback_data) == 5
        assert callback_data[-1][0] == ""  # Last callback gets empty string
        assert callback_data[-1][1]["is_complete"] is True

    def test_streaming_with_tool_calls(self, openai_llm):
        """Test streaming with tool calls."""
        # Create a mock tool call object
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function = MagicMock()
        tool_call.function.name = "weather"
        tool_call.function.arguments = '{"location": "San Francisco"}'

        # Create a sequence of chunks with tool call
        chunks = [
            MockChunk(content="I'll check the weather"),
            MockChunk(tool_calls=[tool_call]),
            MockChunk(is_empty=True),  # End of stream
        ]

        # Mock the create method to return our stream
        openai_llm.client.chat.completions.create.return_value = chunks

        # Call stream_generate
        collected_chunks = []
        for chunk, stats in openai_llm.stream_generate(
            event_id="test-tool-stream",
            system_prompt="You are a weather assistant",
            messages=[
                {"message_type": "human", "message": "What's the weather in SF?"}
            ],
        ):
            collected_chunks.append((chunk, stats))

        # Verify tool call information in stats
        final_stats = collected_chunks[-1][1]
        assert "tool_use" in final_stats
        assert "call_123" in final_stats["tool_use"]
        assert final_stats["tool_use"]["call_123"]["name"] == "weather"
        assert (
            final_stats["tool_use"]["call_123"]["arguments"]
            == '{"location": "San Francisco"}'
        )

    def test_streaming_progressive_tool_calls(self, openai_llm):
        """Test streaming with tool calls that come in multiple chunks."""
        # Create a sequence of tool call chunks that build progressively
        tool_call_1 = MagicMock()
        tool_call_1.id = "call_123"
        tool_call_1.function = MagicMock()
        tool_call_1.function.name = "weather"
        tool_call_1.function.arguments = '{"location": "'

        tool_call_2 = MagicMock()
        tool_call_2.id = "call_123"
        tool_call_2.function = MagicMock()
        tool_call_2.function.name = "weather"
        tool_call_2.function.arguments = "San "

        tool_call_3 = MagicMock()
        tool_call_3.id = "call_123"
        tool_call_3.function = MagicMock()
        tool_call_3.function.name = "weather"
        tool_call_3.function.arguments = 'Francisco"}'

        # Create a sequence of chunks
        chunks = [
            MockChunk(content="I'll check the weather"),
            MockChunk(tool_calls=[tool_call_1]),
            MockChunk(tool_calls=[tool_call_2]),
            MockChunk(tool_calls=[tool_call_3]),
            MockChunk(is_empty=True),  # End of stream
        ]

        # Mock the create method to return our stream
        openai_llm.client.chat.completions.create.return_value = chunks

        # Call stream_generate
        final_chunk = None
        for chunk, stats in openai_llm.stream_generate(
            event_id="test-progressive-tools",
            system_prompt="You are a weather assistant",
            messages=[
                {"message_type": "human", "message": "What's the weather in SF?"}
            ],
        ):
            final_chunk = (chunk, stats)

        # Verify complete tool call information in final stats
        final_stats = final_chunk[1]
        assert "tool_use" in final_stats
        assert "call_123" in final_stats["tool_use"]
        assert final_stats["tool_use"]["call_123"]["name"] == "weather"
        assert (
            final_stats["tool_use"]["call_123"]["arguments"]
            == '{"location": "San Francisco"}'
        )

    def test_streaming_error_handling(self, openai_llm):
        """Test error handling during streaming."""
        # Mock a connection error during streaming
        openai_llm.client.chat.completions.create.side_effect = Exception(
            "Connection error"
        )

        # Call stream_generate and expect an exception
        with pytest.raises(Exception) as exc_info:
            # Consume the generator to trigger the error
            list(
                openai_llm.stream_generate(
                    event_id="test-error",
                    system_prompt="You are a test assistant",
                    messages=[{"message_type": "human", "message": "Say hello"}],
                )
            )

        # Verify error message
        assert "error" in str(exc_info.value).lower()
        assert "openai" in str(exc_info.value).lower()

    def test_streaming_image_handling(self, openai_llm):
        """Test streaming with image inputs and cost calculation."""
        # Mock the count_tokens method to return predictable values
        with patch.object(
            openai_llm, "_count_tokens_from_messages", return_value=100
        ), patch.object(
            openai_llm,
            "_format_messages_for_model",
            return_value=[{"role": "user", "content": "mock formatted content"}],
        ):
            # Create a sequence of chunks
            chunks = [
                MockChunk(content="I see an image"),
                MockChunk(content=" of a cat"),
                MockChunk(is_empty=True),  # End of stream
            ]

            # Mock the create method to return our stream
            openai_llm.client.chat.completions.create.return_value = chunks

            # Call stream_generate with image
            final_chunk = None
            for chunk, stats in openai_llm.stream_generate(
                event_id="test-image",
                system_prompt="Describe the image",
                messages=[
                    {
                        "message_type": "human",
                        "message": "What's in this image?",
                        "image_paths": ["/path/to/image.jpg"],
                    }
                ],
            ):
                final_chunk = (chunk, stats)

            # Verify stats contain image-related information
            final_stats = final_chunk[1]
            assert "images" in final_stats

    def test_streaming_with_functions(self, openai_llm):
        """Test streaming with function definitions."""
        # Define a test function signature
        with patch.object(
            openai_llm, "_convert_function_to_tool"
        ) as mock_convert, patch.object(
            openai_llm,
            "_format_messages_for_model",
            return_value=[{"role": "user", "content": "Call function"}],
        ):

            # Mock function conversion
            mock_convert.return_value = {
                "name": "test_function",
                "description": "A test function",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "number"},
                    },
                },
            }

            # Create a sequence of chunks with tool call
            tool_call = MagicMock()
            tool_call.id = "call_456"
            tool_call.function = MagicMock()
            tool_call.function.name = "test_function"
            tool_call.function.arguments = '{"param1": "test", "param2": 123}'

            chunks = [
                MockChunk(content="I'll call a function"),
                MockChunk(tool_calls=[tool_call]),
                MockChunk(is_empty=True),  # End of stream
            ]

            # Mock the create method to return our stream
            openai_llm.client.chat.completions.create.return_value = chunks

            # Define a mock function
            mock_function = MagicMock()
            mock_function.__name__ = "test_function"

            # Call stream_generate with function
            final_chunk = None
            for chunk, stats in openai_llm.stream_generate(
                event_id="test-function",
                system_prompt="You can call functions",
                messages=[
                    {"message_type": "human", "message": "Call the test function"}
                ],
                functions=[mock_function],
            ):
                final_chunk = (chunk, stats)

            # Verify stream was created with tools parameter
            openai_llm.client.chat.completions.create.assert_called_once()
            assert "stream" in openai_llm.client.chat.completions.create.call_args[1]
            assert (
                openai_llm.client.chat.completions.create.call_args[1]["stream"] is True
            )

            # Verify final stats include tool information
            final_stats = final_chunk[1]
            assert "tool_use" in final_stats

    def test_streaming_token_accumulation(self, openai_llm):
        """Test token accumulation during streaming."""
        # Create a sequence of chunks
        chunks = [
            MockChunk(content="Token"),
            MockChunk(content=" by"),
            MockChunk(content=" token"),
            MockChunk(is_empty=True),  # End of stream
        ]

        # Mock the create method to return our stream
        openai_llm.client.chat.completions.create.return_value = chunks

        # Call stream_generate
        collected_stats = []
        for _chunk, stats in openai_llm.stream_generate(
            event_id="test-tokens",
            system_prompt="Count tokens",
            messages=[
                {"message_type": "human", "message": "Say something token by token"}
            ],
        ):
            collected_stats.append(stats)

        # Verify token counts increase with each chunk
        for i in range(1, len(collected_stats)):
            assert (
                collected_stats[i]["write_tokens"]
                >= collected_stats[i - 1]["write_tokens"]
            )

        # Verify final stats have cost information
        assert "read_cost" in collected_stats[-1]
        assert "write_cost" in collected_stats[-1]
        assert "total_cost" in collected_stats[-1]
        assert collected_stats[-1]["is_complete"] is True

    def test_streaming_empty_response(self, openai_llm):
        """Test handling of empty response in streaming."""
        # Create a sequence with only empty chunks
        chunks = [MockChunk(is_empty=True), MockChunk(is_empty=True)]

        # Mock the create method to return our stream
        openai_llm.client.chat.completions.create.return_value = chunks

        # Call stream_generate
        collected_chunks = []
        for chunk, stats in openai_llm.stream_generate(
            event_id="test-empty",
            system_prompt="Be silent",
            messages=[
                {"message_type": "human", "message": "You don't need to respond"}
            ],
        ):
            collected_chunks.append((chunk, stats))

        # Verify we still get the completion marker
        assert collected_chunks[-1][0] == ""
        assert collected_chunks[-1][1]["is_complete"] is True

    def test_streaming_callback_behavior(self, openai_llm):
        """Test callback behavior during streaming."""
        # Create a sequence of chunks
        chunks = [
            MockChunk(content="First"),
            MockChunk(content=" Second"),
            MockChunk(content=" Third"),
            MockChunk(is_empty=True),  # End of stream
        ]

        # Mock the create method to return our stream
        openai_llm.client.chat.completions.create.return_value = chunks

        # Create a mock callback
        mock_callback = Mock()

        # Call stream_generate with callback
        list(
            openai_llm.stream_generate(
                event_id="test-callback",
                system_prompt="Test callback",
                messages=[{"message_type": "human", "message": "Say three things"}],
                callback=mock_callback,
            )
        )

        # Verify callback was called for each chunk including completion
        assert mock_callback.call_count == 4

        # Verify callback was called with correct arguments
        for i, call_args in enumerate(mock_callback.call_args_list):
            if i < 3:  # Content chunks
                assert call_args[0][0] in ["First", " Second", " Third"]
                assert isinstance(call_args[0][1], dict)
                assert "is_complete" in call_args[0][1]
                assert call_args[0][1]["is_complete"] is False
            else:  # Completion marker
                assert call_args[0][0] == ""
                assert call_args[0][1]["is_complete"] is True

    def test_streaming_api_parameters(self, openai_llm):
        """Test that API parameters are correctly passed during streaming."""
        # Create a minimal successful response
        chunks = [MockChunk(content="Ok"), MockChunk(is_empty=True)]

        # Mock the create method to return our stream
        openai_llm.client.chat.completions.create.return_value = chunks

        # Call stream_generate with specific parameters
        list(
            openai_llm.stream_generate(
                event_id="test-params",
                system_prompt="Test parameters",
                messages=[{"message_type": "human", "message": "Hello"}],
                max_tokens=100,
                temp=0.7,
            )
        )

        # Verify parameters were passed correctly
        call_args = openai_llm.client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4o"
        assert call_args["max_tokens"] == 100
        assert call_args["temperature"] == 0.7
        assert call_args["stream"] is True

    def test_streaming_message_conversion(self, openai_llm):
        """Test that messages are correctly converted for streaming."""
        # Mock _format_messages_for_model to track conversions
        with patch.object(openai_llm, "_format_messages_for_model") as mock_format:
            # Set up return value for format function
            mock_format.return_value = [
                {"role": "user", "content": "Converted message"}
            ]

            # Create a minimal successful response
            chunks = [MockChunk(content="Ok"), MockChunk(is_empty=True)]

            # Mock the create method to return our stream
            openai_llm.client.chat.completions.create.return_value = chunks

            # Call stream_generate with complex message
            complex_message = {
                "message_type": "human",
                "message": "Test with images",
                "image_urls": ["https://example.com/image.jpg"],
            }

            list(
                openai_llm.stream_generate(
                    event_id="test-conversion",
                    system_prompt="Test message conversion",
                    messages=[complex_message],
                )
            )

            # Verify format method was called with our message
            mock_format.assert_called_once()
            assert len(mock_format.call_args[0][0]) == 1
            assert mock_format.call_args[0][0][0]["message_type"] == "human"
            assert "image_urls" in mock_format.call_args[0][0][0]
