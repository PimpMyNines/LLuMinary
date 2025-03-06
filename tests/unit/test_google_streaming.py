"""
Tests for Google Provider streaming functionality.

These tests use module mocking to create fake versions of the Google modules
required for streaming tests, allowing us to test the streaming functionality
without requiring the actual dependencies to be installed.
"""

from unittest.mock import ANY, MagicMock, patch

import pytest

from lluminary.models.providers.google import GoogleLLM
from tests.unit.test_google_module_mock import patch_google_modules


class TestGoogleLLMStreaming:
    """Tests for the streaming functionality of GoogleLLM."""

    @patch.object(GoogleLLM, "auth")
    def test_stream_generate_basic(self, mock_auth):
        """Test basic streaming functionality."""
        # Apply our module patches for Google dependencies
        patches, mocks = patch_google_modules()
        for p in patches:
            p.start()

        try:
            llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

            # Create test messages
            messages = [{"message_type": "human", "message": "Tell me a story"}]

            # Setup mock model and content type instances
            mock_model = MagicMock()
            mocks["google.generativeai"].GenerativeModel.return_value = mock_model

            # Mock the Content and Part classes for system prompt
            mock_system_content = MagicMock()
            mock_system_part = MagicMock()
            mocks["google.genai.types.content_types"].Content.return_value = (
                mock_system_content
            )
            mocks["google.genai.types.content_types"].Part.from_text.return_value = (
                mock_system_part
            )

            # Mock _format_messages_for_model
            with patch.object(
                llm, "_format_messages_for_model", return_value=[MagicMock()]
            ):
                # Create mock streaming response
                mock_chunk1 = MagicMock()
                mock_chunk1.text = "Once upon "

                mock_chunk2 = MagicMock()
                mock_chunk2.text = "a time"

                mock_chunk3 = MagicMock()
                mock_chunk3.text = " there was"

                # Final chunk with empty text
                mock_chunk4 = MagicMock()
                mock_chunk4.text = ""
                mock_chunk4.candidates = [MagicMock()]

                # Set up the model to return chunks
                mock_model.generate_content.return_value = [
                    mock_chunk1,
                    mock_chunk2,
                    mock_chunk3,
                    mock_chunk4,
                ]

                # Call stream_generate and collect results
                chunks = []
                for chunk, usage in llm.stream_generate(
                    event_id="test_event",
                    system_prompt="Tell a fairy tale",
                    messages=messages,
                    max_tokens=100,
                ):
                    chunks.append((chunk, usage))

                # Verify GenerativeModel was created with correct parameters
                mocks["google.generativeai"].GenerativeModel.assert_called_with(
                    model_name="gemini-2.0-flash",
                    generation_config={
                        "max_output_tokens": 100,
                        "temperature": 0.0,
                    },
                    tools=None,
                    safety_settings=None,
                )

                # Verify streaming was enabled
                mock_model.generate_content.assert_called_with(ANY, stream=True)

                # Verify chunks were processed correctly
                assert len(chunks) == 4
                assert chunks[0][0] == "Once upon "
                assert chunks[1][0] == "a time"
                assert chunks[2][0] == " there was"
                assert chunks[3][0] == ""  # Final empty chunk

                # Verify usage data in first chunk
                assert "read_tokens" in chunks[0][1]
                assert "write_tokens" in chunks[0][1]
                assert "is_complete" in chunks[0][1]
                assert chunks[0][1]["is_complete"] is False

                # Verify final chunk has is_complete=True
                assert chunks[3][1]["is_complete"] is True
                assert "total_cost" in chunks[3][1]

        finally:
            # Stop all patches
            for p in patches:
                p.stop()

    @patch.object(GoogleLLM, "auth")
    def test_stream_generate_with_callback(self, mock_auth):
        """Test streaming with callback function."""
        # Apply our module patches for Google dependencies
        patches, mocks = patch_google_modules()
        for p in patches:
            p.start()

        try:
            llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

            # Create test messages
            messages = [{"message_type": "human", "message": "Hello"}]

            # Setup callback mock
            callback_mock = MagicMock()

            # Setup mock model and content type instances
            mock_model = MagicMock()
            mocks["google.generativeai"].GenerativeModel.return_value = mock_model

            # Mock the Content and Part classes for system prompt
            mock_system_content = MagicMock()
            mock_system_part = MagicMock()
            mocks["google.genai.types.content_types"].Content.return_value = (
                mock_system_content
            )
            mocks["google.genai.types.content_types"].Part.from_text.return_value = (
                mock_system_part
            )

            # Mock _format_messages_for_model
            with patch.object(
                llm, "_format_messages_for_model", return_value=[MagicMock()]
            ):
                # Create mock chunks
                mock_chunk1 = MagicMock()
                mock_chunk1.text = "Hello"

                mock_chunk2 = MagicMock()
                mock_chunk2.text = " there"

                # Final chunk with empty text
                mock_chunk3 = MagicMock()
                mock_chunk3.text = ""
                mock_chunk3.candidates = [MagicMock()]

                # Set up the model to return chunks
                mock_model.generate_content.return_value = [
                    mock_chunk1,
                    mock_chunk2,
                    mock_chunk3,
                ]

                # Call stream_generate with callback
                chunks = []
                for chunk, usage in llm.stream_generate(
                    event_id="test_event",
                    system_prompt="You are a helpful assistant",
                    messages=messages,
                    callback=callback_mock,
                ):
                    chunks.append((chunk, usage))

                # Verify callback was called for each chunk
                assert callback_mock.call_count == 3

                # Verify callback was called with correct arguments
                callback_mock.assert_any_call("Hello", ANY)
                callback_mock.assert_any_call(" there", ANY)
                callback_mock.assert_any_call("", ANY)  # Final empty chunk

        finally:
            # Stop all patches
            for p in patches:
                p.stop()

    @patch.object(GoogleLLM, "auth")
    def test_stream_generate_with_function(self, mock_auth):
        """Test streaming with function calling."""
        # Apply our module patches for Google dependencies
        patches, mocks = patch_google_modules()
        for p in patches:
            p.start()

        try:
            llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

            # Create test messages
            messages = [
                {"message_type": "human", "message": "What's the weather in New York?"}
            ]

            # Define test function
            def get_weather(location: str) -> str:
                """Get weather for a location."""
                return f"Weather in {location}: Sunny"

            # Setup mock model and content type instances
            mock_model = MagicMock()
            mocks["google.generativeai"].GenerativeModel.return_value = mock_model

            # Mock the Content and Part classes for system prompt
            mock_system_content = MagicMock()
            mock_system_part = MagicMock()
            mocks["google.genai.types.content_types"].Content.return_value = (
                mock_system_content
            )
            mocks["google.genai.types.content_types"].Part.from_text.return_value = (
                mock_system_part
            )

            # Mock _format_messages_for_model
            with patch.object(
                llm, "_format_messages_for_model", return_value=[MagicMock()]
            ):
                # Create mock chunks with function call
                mock_chunk1 = MagicMock()
                mock_chunk1.text = "I'll check"

                mock_chunk2 = MagicMock()
                mock_chunk2.text = " the weather"

                # Final chunk with function call
                mock_chunk3 = MagicMock()
                mock_chunk3.text = ""

                # Setup function call in the candidate
                mock_candidate = MagicMock()
                mock_content = MagicMock()
                mock_part = MagicMock()
                mock_function_call = MagicMock()

                mock_chunk3.candidates = [mock_candidate]
                mock_candidate.content = mock_content
                mock_content.parts = [mock_part]
                mock_part.function_call = mock_function_call
                mock_function_call.name = "get_weather"
                mock_function_call.args = {"location": "New York"}

                # Set up the model to return chunks
                mock_model.generate_content.return_value = [
                    mock_chunk1,
                    mock_chunk2,
                    mock_chunk3,
                ]

                # Call stream_generate with function
                chunks = []
                for chunk, usage in llm.stream_generate(
                    event_id="test_event",
                    system_prompt="You are a helpful assistant",
                    messages=messages,
                    functions=[get_weather],
                ):
                    chunks.append((chunk, usage))

                # Verify model was created with functions
                create_args = mocks["google.generativeai"].GenerativeModel.call_args[1]
                assert "tools" in create_args
                assert create_args["tools"] is not None

                # Verify function call was extracted from chunks
                final_usage = chunks[-1][1]
                assert "tool_use" in final_usage
                assert len(final_usage["tool_use"]) > 0

                # Get the tool use data (key is dynamically generated)
                tool_use_key = list(final_usage["tool_use"].keys())[0]
                tool_use = final_usage["tool_use"][tool_use_key]
                assert tool_use["name"] == "get_weather"
                assert "location" in tool_use["arguments"]

        finally:
            # Stop all patches
            for p in patches:
                p.stop()

    @patch.object(GoogleLLM, "auth")
    def test_stream_generate_error_handling(self, mock_auth):
        """Test error handling in stream_generate."""
        # Apply our module patches for Google dependencies
        patches, mocks = patch_google_modules()
        for p in patches:
            p.start()

        try:
            llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

            # Create test messages
            messages = [{"message_type": "human", "message": "Hello"}]

            # Make GenerativeModel raise an error
            mocks["google.generativeai"].GenerativeModel.side_effect = Exception(
                "API Error"
            )

            # Call stream_generate and expect exception
            with pytest.raises(Exception) as exc_info:
                list(
                    llm.stream_generate(
                        event_id="test_event",
                        system_prompt="You are a helpful assistant",
                        messages=messages,
                    )
                )

            # Verify error message
            assert "Error streaming from Google" in str(exc_info.value)
            assert "API Error" in str(exc_info.value)

        finally:
            # Stop all patches
            for p in patches:
                p.stop()
