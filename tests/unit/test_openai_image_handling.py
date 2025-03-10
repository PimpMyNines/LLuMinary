"""
Tests for OpenAI provider image handling functionality.

This module focuses specifically on testing the image handling
capabilities of the OpenAI provider, including encoding, processing,
and cost calculation for images.
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

import pytest
from lluminary.models.providers.openai import OpenAILLM
from PIL import Image


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


def test_image_input_processing(openai_llm):
    """Test processing of image inputs from various sources."""
    # Test local image encoding
    image_path = "test_image.jpg"
    mock_image_content = b"fake image content"

    # Use mock_open to avoid actual file operations
    with patch("builtins.open", mock_open(read_data=mock_image_content)):
        encoded = openai_llm._encode_image(image_path)
        assert encoded == base64.b64encode(mock_image_content).decode("utf-8")

    # Test image URL encoding
    image_url = "http://example.com/image.jpg"

    # Create a mock PIL image
    mock_img = MagicMock(spec=Image.Image)
    mock_img.mode = "RGB"
    mock_img.size = (100, 100)

    # Mock BytesIO to capture saved image
    mock_output = BytesIO()

    # Test normal image format
    with patch("requests.get") as mock_get, patch(
        "PIL.Image.open", return_value=mock_img
    ), patch("io.BytesIO", return_value=mock_output):
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = b"mock image data"
        mock_get.return_value = mock_response

        # Mock save method to write something to BytesIO
        def mock_save(output, format, quality):
            output.write(b"mock jpeg data")

        mock_img.save = mock_save

        encoded = openai_llm._encode_image_url(image_url)

        # Verify the image was properly processed
        mock_get.assert_called_once_with(image_url, timeout=10)

        # Verify the result is base64 encoded
        assert isinstance(encoded, str)
        assert encoded == base64.b64encode(b"mock jpeg data").decode("utf-8")


def test_multiple_images_in_message(openai_llm):
    """Test handling of multiple images in a single message."""
    # Create a message with multiple images
    message = {
        "message_type": "human",
        "message": "Look at these images",
        "image_paths": ["image1.jpg", "image2.jpg"],
        "image_urls": ["http://example.com/image3.jpg"],
    }

    # Mock all image encoding methods
    with patch.object(
        openai_llm, "_encode_image", return_value="base64_local_image"
    ), patch.object(openai_llm, "_encode_image_url", return_value="base64_url_image"):

        formatted = openai_llm._format_messages_for_model([message])

        # Verify the result is a list with multi-part content
        assert isinstance(formatted, list)
        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert isinstance(formatted[0]["content"], list)

        # Verify the content has text and image parts
        content_list = formatted[0]["content"]

        # Count text and image parts
        text_parts = [part for part in content_list if part.get("type") == "text"]
        image_parts = [part for part in content_list if part.get("type") == "image_url"]

        # Should have 1 text part and 3 image parts (2 local, 1 URL)
        assert len(text_parts) == 1
        assert len(image_parts) == 3

        # Verify text content
        assert "Look at these images" in text_parts[0]["text"]

        # Verify image parts
        for img_part in image_parts:
            assert img_part["type"] == "image_url"
            assert "url" in img_part["image_url"]
            assert img_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_image_cost_calculation(openai_llm):
    """Test cost calculation for image inputs."""
    # Create messages with text and images
    messages = [
        {
            "message_type": "human",
            "message": "What's in this image?",
            "image_paths": ["image.jpg"],
        }
    ]

    # Setup mocks
    with patch.object(
        openai_llm, "_encode_image", return_value="base64_encoded"
    ), patch.object(
        openai_llm, "_calculate_image_tokens", return_value=300
    ), patch.object(
        openai_llm, "client"
    ) as mock_client:

        # Mock API response
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="I see a cat in the image"))],
            usage=MagicMock(prompt_tokens=350, completion_tokens=10, total_tokens=360),
        )

        # Generate with image
        response, usage, _ = openai_llm.generate(
            event_id="test",
            system_prompt="Describe the image",
            messages=messages,
            max_tokens=100,
        )

        # Verify response and API call
        assert response == "I see a cat in the image"
        assert mock_client.chat.completions.create.called

        # Verify usage calculation includes image tokens
        assert usage["read_tokens"] == 350  # Should match what the API returned
        assert usage["write_tokens"] == 10
        assert usage["total_tokens"] == 360
        assert "total_cost" in usage
        assert usage["total_cost"] > 0
