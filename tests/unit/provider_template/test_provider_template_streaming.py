"""
Unit tests for the streaming functionality of the Provider Template.
"""

from unittest.mock import patch

import pytest
from lluminary.exceptions import LLMMistake
from lluminary.models.providers.provider_template import ProviderNameLLM


@pytest.fixture
def provider_instance():
    """Create a basic provider instance for testing."""
    # Mock the auth method to avoid actual authentication
    with patch.object(ProviderNameLLM, "auth", return_value=None):
        provider = ProviderNameLLM(model_name="provider-model-1")
    return provider


class TestProviderTemplateStreaming:
    """Tests for streaming functionality of ProviderNameLLM."""

    def test_stream_generate_implementation(self, provider_instance):
        """Test that _stream_generate is properly implemented."""

        # Add a _stream_generate implementation to the provider instance for testing
        def mock_stream_generate(
            self,
            event_id,
            system_prompt,
            messages,
            max_tokens=1000,
            temp=0.0,
            top_k=200,
            tools=None,
            thinking_budget=None,
        ):
            """Simulated streaming implementation."""
            # Format messages
            formatted_messages = self._format_messages_for_model(messages)

            # Configure basic usage stats
            base_usage = {
                "read_tokens": 10,
                "write_tokens": 0,
                "total_tokens": 10,
                "event_id": event_id,
                "model": self.model_name,
            }

            # Yield chunks with usage stats
            usage1 = base_usage.copy()
            usage1["write_tokens"] = 2
            yield "Hello", usage1

            # Second chunk
            usage2 = base_usage.copy()
            usage2["write_tokens"] = 3
            yield " world", usage2

            # Final chunk should have is_complete flag
            final_usage = base_usage.copy()
            final_usage["write_tokens"] = 5
            final_usage["is_complete"] = True
            yield "!", final_usage

        # Add the method to the instance
        provider_instance._stream_generate = mock_stream_generate.__get__(
            provider_instance
        )

        # Mock the format_messages method to avoid dependency
        with patch.object(
            provider_instance, "_format_messages_for_model"
        ) as mock_format:
            mock_format.return_value = [{"role": "user", "content": "Hello"}]

            # Call stream_generate and collect results
            chunks = []
            for chunk, usage in provider_instance._stream_generate(
                event_id="test-event",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "Hello"}],
            ):
                chunks.append((chunk, usage))

            # Verify results
            assert len(chunks) == 3
            assert chunks[0][0] == "Hello"
            assert chunks[1][0] == " world"
            assert chunks[2][0] == "!"

            # Verify usage stats
            assert chunks[0][1]["write_tokens"] == 2
            assert chunks[1][1]["write_tokens"] == 3
            assert chunks[2][1]["write_tokens"] == 5
            assert chunks[2][1]["is_complete"] is True

            # Verify other usage stats
            assert all("read_tokens" in usage for _, usage in chunks)
            assert all("event_id" in usage for _, usage in chunks)
            assert all("model" in usage for _, usage in chunks)
            assert all(usage["model"] == "provider-model-1" for _, usage in chunks)

    def test_stream_generate_with_error(self, provider_instance):
        """Test error handling in _stream_generate."""

        # Add a streaming implementation that raises an error
        def mock_stream_generate_with_error(
            self,
            event_id,
            system_prompt,
            messages,
            max_tokens=1000,
            temp=0.0,
            top_k=200,
            tools=None,
            thinking_budget=None,
        ):
            """Simulated streaming implementation with error."""
            # Yield a few chunks first
            yield "Starting response", {"read_tokens": 10, "write_tokens": 2}

            # Then raise an error
            raise Exception("Simulated API error during streaming")

        # Add the method to the instance
        provider_instance._stream_generate = mock_stream_generate_with_error.__get__(
            provider_instance
        )

        # Add error handling wrapper method
        def stream_generate_with_error_handler(
            self,
            event_id,
            system_prompt,
            messages,
            max_tokens=1000,
            temp=0.0,
            top_k=200,
            tools=None,
            thinking_budget=None,
        ):
            """Error handling wrapper for streaming."""
            try:
                for chunk, usage in self._stream_generate(
                    event_id,
                    system_prompt,
                    messages,
                    max_tokens,
                    temp,
                    top_k,
                    tools,
                    thinking_budget,
                ):
                    yield chunk, usage
            except Exception as e:
                # Convert to LLMMistake exception
                raise LLMMistake(
                    f"Error during streaming with {self.__class__.__name__}: {e!s}",
                    error_type="stream_error",
                    provider=self.__class__.__name__,
                    details={"original_error": str(e)},
                )

        # Replace the _stream_generate method with error handling wrapper
        provider_instance.stream_generate_with_error_handler = (
            stream_generate_with_error_handler.__get__(provider_instance)
        )

        # Mock format_messages
        with patch.object(
            provider_instance, "_format_messages_for_model", return_value=[]
        ):
            # Call stream_generate and expect exception
            chunks = []
            with pytest.raises(LLMMistake) as excinfo:
                for (
                    chunk,
                    usage,
                ) in provider_instance.stream_generate_with_error_handler(
                    event_id="test-event",
                    system_prompt="Test prompt",
                    messages=[{"message_type": "human", "message": "Hello"}],
                ):
                    chunks.append((chunk, usage))

            # Verify we got the first chunk before the error
            assert len(chunks) == 1
            assert chunks[0][0] == "Starting response"

            # Verify exception details
            assert "Error during streaming" in str(excinfo.value)
            assert excinfo.value.error_type == "stream_error"
            assert excinfo.value.provider == "ProviderNameLLM"
            assert "Simulated API error" in str(
                excinfo.value.details.get("original_error", "")
            )

    def test_stream_generate_with_tools(self, provider_instance):
        """Test streaming with tools parameter."""
        # Define test tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        # Mock streaming implementation that returns tool calls
        def mock_stream_with_tools(
            self,
            event_id,
            system_prompt,
            messages,
            max_tokens=1000,
            temp=0.0,
            top_k=200,
            tools=None,
            thinking_budget=None,
        ):
            """Simulated streaming with tool calls."""
            # Verify tools were passed correctly
            assert tools is not None
            assert len(tools) == 1
            assert tools[0]["function"]["name"] == "get_weather"

            # Base usage stats
            base_usage = {
                "read_tokens": 15,
                "write_tokens": 0,
                "total_tokens": 15,
                "event_id": event_id,
                "model": self.model_name,
                "tool_use": {},
            }

            # Yield initial text
            usage1 = base_usage.copy()
            usage1["write_tokens"] = 3
            yield "I'll check the weather for you.", usage1

            # Yield tool call
            usage2 = base_usage.copy()
            usage2["write_tokens"] = 5
            usage2["tool_use"] = {
                "name": "get_weather",
                "arguments": {"location": "Seattle"},
            }
            yield "", usage2

            # Final response after tool call
            final_usage = base_usage.copy()
            final_usage["write_tokens"] = 7
            final_usage["is_complete"] = True
            yield "The weather in Seattle is sunny.", final_usage

        # Add the method to the instance
        provider_instance._stream_generate = mock_stream_with_tools.__get__(
            provider_instance
        )

        # Mock format_messages
        with patch.object(
            provider_instance, "_format_messages_for_model", return_value=[]
        ):

            # Call stream_generate and collect results
            chunks = []
            for chunk, usage in provider_instance._stream_generate(
                event_id="test-event",
                system_prompt="You are a helpful assistant",
                messages=[
                    {
                        "message_type": "human",
                        "message": "What's the weather in Seattle?",
                    }
                ],
                tools=tools,
            ):
                chunks.append((chunk, usage))

            # Verify results
            assert len(chunks) == 3

            # Verify tool use data
            assert "tool_use" in chunks[1][1]
            assert chunks[1][1]["tool_use"]["name"] == "get_weather"
            assert chunks[1][1]["tool_use"]["arguments"]["location"] == "Seattle"

            # Verify final response
            assert chunks[2][0] == "The weather in Seattle is sunny."
            assert chunks[2][1]["is_complete"] is True

    def test_stream_generate_token_counting(self, provider_instance):
        """Test token counting in streaming mode."""

        # Mock streaming implementation with progressive token counting
        def mock_stream_with_token_counting(
            self,
            event_id,
            system_prompt,
            messages,
            max_tokens=1000,
            temp=0.0,
            top_k=200,
            tools=None,
            thinking_budget=None,
        ):
            """Simulated streaming with progressive token counting."""
            # Starting usage
            usage1 = {
                "read_tokens": 20,
                "write_tokens": 5,
                "total_tokens": 25,
                "event_id": event_id,
                "model": self.model_name,
            }
            yield "First chunk", usage1

            # Second chunk - tokens should accumulate
            usage2 = {
                "read_tokens": 20,  # Same read tokens
                "write_tokens": 10,  # Accumulated write tokens
                "total_tokens": 30,
                "event_id": event_id,
                "model": self.model_name,
            }
            yield " second chunk", usage2

            # Final chunk - complete count
            final_usage = {
                "read_tokens": 20,
                "write_tokens": 18,  # Final accumulated write tokens
                "total_tokens": 38,
                "event_id": event_id,
                "model": self.model_name,
                "is_complete": True,
            }
            yield " final chunk", final_usage

        # Add the method to the instance
        provider_instance._stream_generate = mock_stream_with_token_counting.__get__(
            provider_instance
        )

        # Mock format_messages
        with patch.object(
            provider_instance, "_format_messages_for_model", return_value=[]
        ):

            # Call stream_generate and collect results
            chunks = []
            for chunk, usage in provider_instance._stream_generate(
                event_id="test-event",
                system_prompt="Test prompt",
                messages=[{"message_type": "human", "message": "Hello"}],
            ):
                chunks.append((chunk, usage))

            # Verify progressive token counting
            assert chunks[0][1]["write_tokens"] == 5
            assert chunks[1][1]["write_tokens"] == 10
            assert chunks[2][1]["write_tokens"] == 18

            # Verify final usage stats
            assert chunks[2][1]["total_tokens"] == 38
            assert chunks[2][1]["is_complete"] is True

            # Calculate the combined response
            full_response = "".join(chunk for chunk, _ in chunks)
            assert full_response == "First chunk second chunk final chunk"
