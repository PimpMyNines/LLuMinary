"""
Simplified tests for Google Gemini provider.
"""

from unittest.mock import MagicMock, patch

import pytest

# Skip all tests in this file
pytestmark = pytest.mark.skip("Skip Google provider tests")


def test_basic_functionality():
    """Test the basic functionality of GoogleLLM."""
    with patch("google.genai", create=True):
        with patch("requests.get"):
            with patch("PIL.Image.open"):
                # Import GoogleLLM after patching dependencies
                from lluminary.models.providers.google import GoogleLLM

                # Further patch auth to avoid actual API calls
                with patch.object(GoogleLLM, "auth"):
                    # Create an instance
                    llm = GoogleLLM("gemini-2.0-flash")

                    # Test basic methods
                    assert llm.model_name == "gemini-2.0-flash"
                    assert llm.supports_image_input() is True
                    assert "gemini-2.0-flash" in llm.get_supported_models()

                    # Test model costs are available
                    costs = llm.get_model_costs()
                    assert "read_token" in costs
                    assert "write_token" in costs
                    assert costs["read_token"] > 0


def test_with_mocked_generation():
    """Test GoogleLLM with mocked generation."""
    with patch("google.genai", create=True), patch("PIL.Image.open"), patch(
        "requests.get"
    ):
        # Import GoogleLLM after patching
        from lluminary.models.providers.google import GoogleLLM

        # First, patch LLM.auth to avoid authentication attempts
        with patch.object(GoogleLLM, "auth"):
            # Create a test instance
            llm = GoogleLLM("gemini-2.0-flash")

            # Create mocks for the response
            mock_response = MagicMock()
            mock_response.text = "Mock response"

            # Add usage metadata
            mock_usage_metadata = MagicMock()
            mock_usage_metadata.prompt_token_count = 10
            mock_usage_metadata.candidates_token_count = 5
            mock_usage_metadata.total_token_count = 15

            # Assign usage metadata to response
            type(mock_response).usage_metadata = mock_usage_metadata

            # Create mock client and configure it to return our mock response
            mock_client = MagicMock()
            mock_client.models.generate_content.return_value = mock_response

            # Directly replace the client
            llm.client = mock_client

            # Create simplified formatted messages to avoid complex mocking
            formatted_messages = [MagicMock()]

            # Mock _format_messages_for_model to return our simplified messages
            with patch.object(
                llm, "_format_messages_for_model", return_value=formatted_messages
            ):
                # Test raw generation
                response, usage = llm._raw_generate(
                    event_id="test_event",
                    system_prompt="Test prompt",
                    messages=[{"message_type": "human", "message": "Test message"}],
                )

                # Verify response
                assert response == "Mock response"

                # Verify usage stats
                assert usage["read_tokens"] == 10
                assert usage["write_tokens"] == 5
                assert "total_cost" in usage


def test_with_image_input():
    """Test GoogleLLM with image input."""
    with patch("google.genai", create=True), patch("PIL.Image.open"), patch(
        "requests.get"
    ):
        # Import GoogleLLM after patching
        from lluminary.models.providers.google import GoogleLLM

        with patch.object(GoogleLLM, "auth"):
            # Create a test instance
            llm = GoogleLLM("gemini-2.0-flash")

            # Create a message with images
            message_with_images = [
                {
                    "message_type": "human",
                    "message": "What's in this image?",
                    "image_paths": ["/path/to/image.jpg"],
                    "image_urls": ["https://example.com/image.jpg"],
                }
            ]

            # Mock _process_image to return a mock image
            mock_image = MagicMock()
            with patch.object(llm, "_process_image", return_value=mock_image):
                # Mock client and API response
                mock_response = MagicMock()
                mock_response.text = "I see a cat in the image"
                mock_response.usage_metadata.prompt_token_count = (
                    100  # Higher for images
                )
                mock_response.usage_metadata.candidates_token_count = 10
                mock_response.usage_metadata.total_token_count = 110

                mock_client = MagicMock()
                mock_client.models.generate_content.return_value = mock_response
                llm.client = mock_client

                # Mock message formatting
                with patch.object(
                    llm, "_format_messages_for_model", return_value=[MagicMock()]
                ):
                    # Test raw generation with images
                    response, usage = llm._raw_generate(
                        event_id="test_image",
                        system_prompt="",
                        messages=message_with_images,
                    )

                    # Verify response
                    assert response == "I see a cat in the image"

                    # Verify image costs are included
                    assert usage["images"] == 2
                    assert "image_cost" in usage
                    assert usage["total_cost"] > 0


@patch("google.generativeai.GenerativeModel")
def test_stream_generation(mock_generative_model):
    """Test streaming generation."""
    with patch("google.genai", create=True), patch("PIL.Image.open"), patch(
        "requests.get"
    ):
        # Import GoogleLLM after patching
        from lluminary.models.providers.google import GoogleLLM

        with patch.object(GoogleLLM, "auth"):
            # Create a test instance
            llm = GoogleLLM("gemini-2.0-flash")

            # Create mock model
            mock_model = MagicMock()

            # Create mock streaming response
            mock_stream = [
                MagicMock(text="Hello"),
                MagicMock(text=" world"),
                MagicMock(text="!"),
                MagicMock(text="", candidates=[MagicMock(content=MagicMock(parts=[]))]),
            ]

            # Setup mock model to return the mock stream
            mock_model.generate_content.return_value = mock_stream
            mock_generative_model.return_value = mock_model

            # Create test message
            messages = [
                {"message_type": "human", "message": "Please stream a response"}
            ]

            # Count chunks received
            chunks = []
            for chunk, usage in llm.stream_generate(
                event_id="test_stream",
                system_prompt="You are a helpful assistant",
                messages=messages,
            ):
                chunks.append((chunk, usage))

            # Verify chunks received
            assert len(chunks) == 4
            assert chunks[0][0] == "Hello"
            assert chunks[1][0] == " world"
            assert chunks[2][0] == "!"
            assert chunks[3][0] == ""  # Final empty chunk

            # Verify final usage contains cost information
            final_usage = chunks[-1][1]
            assert "total_cost" in final_usage
            assert final_usage["is_complete"] is True
