"""
Unit tests for Anthropic image processing functionality.

This module tests the image processing capabilities of the Anthropic provider,
including handling local images, image URLs, and format conversion.
"""

import base64
import json
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

import pytest
from lluminary.exceptions import FormatError, ProviderError
from lluminary.models.providers.anthropic import AnthropicLLM
from PIL import Image


@pytest.fixture
def anthropic_llm():
    """Fixture for Anthropic LLM instance."""
    with patch("anthropic.Anthropic"):
        # Create the LLM instance with mock API key
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Ensure config exists
        llm.config = {"api_key": "test-key"}

        yield llm


def test_encode_image(anthropic_llm):
    """Test encoding a local image to base64."""
    # Create a mock image file and PIL Image object
    mock_img = MagicMock(spec=Image.Image)
    mock_img.mode = "RGB"
    mock_img.size = (100, 100)

    # Mock BytesIO output with fake image data
    fake_image_data = b"fake_jpeg_image_data"
    mock_output = BytesIO(fake_image_data)

    with patch("PIL.Image.open", return_value=mock_img) as mock_open_image, patch(
        "os.path.exists", return_value=True
    ), patch.object(BytesIO, "getvalue", return_value=fake_image_data), patch(
        "builtins.open", mock_open(read_data=b"test")
    ):

        # Configure mock_img.save to write to our BytesIO
        def mock_save(output, format, quality):
            # Just use the already prepared BytesIO
            pass

        mock_img.save = mock_save

        # Call _encode_image
        result = anthropic_llm._encode_image("/path/to/test_image.jpg")

        # Verify the image was opened
        mock_open_image.assert_called_once_with(mock_open().return_value)

        # Verify result is base64 encoded
        expected_base64 = base64.b64encode(fake_image_data).decode("utf-8")
        assert result == expected_base64


def test_encode_image_with_transparency(anthropic_llm):
    """Test encoding an image with transparency (RGBA mode)."""
    # Create a mock image with RGBA mode
    mock_img = MagicMock(spec=Image.Image)
    mock_img.mode = "RGBA"
    mock_img.size = (100, 100)

    # Create a mock for the background and split result
    mock_background = MagicMock(spec=Image.Image)

    # Mock BytesIO output with fake image data
    fake_image_data = b"fake_jpeg_image_data"

    with patch("PIL.Image.open", return_value=mock_img) as mock_open_image, patch(
        "PIL.Image.new", return_value=mock_background
    ) as mock_new_image, patch("os.path.exists", return_value=True), patch.object(
        BytesIO, "getvalue", return_value=fake_image_data
    ), patch(
        "builtins.open", mock_open(read_data=b"test")
    ):

        # Configure mock_img.split to return alpha channel
        mock_img.split.return_value = [None, None, None, MagicMock()]  # Alpha is last

        # Configure mock_background.save to write to BytesIO
        def mock_save(output, format, quality):
            # Just use the already prepared BytesIO
            pass

        mock_background.save = mock_save

        # Call _encode_image
        result = anthropic_llm._encode_image("/path/to/test_image.png")

        # Verify Image.new was called to create white background
        mock_new_image.assert_called_once_with("RGB", (100, 100), (255, 255, 255))

        # Verify paste was called with mask
        mock_background.paste.assert_called_once_with(
            mock_img, mask=mock_img.split.return_value[-1]
        )

        # Verify result is base64 encoded
        expected_base64 = base64.b64encode(fake_image_data).decode("utf-8")
        assert result == expected_base64


def test_encode_image_file_not_found(anthropic_llm):
    """Test error handling when image file is not found."""
    with patch("os.path.exists", return_value=False):
        # Call _encode_image with non-existent file
        with pytest.raises(ProviderError) as excinfo:
            anthropic_llm._encode_image("/path/to/nonexistent.jpg")

        # Verify error details
        assert "Image file not found" in str(excinfo.value)
        assert "/path/to/nonexistent.jpg" in str(excinfo.value)
        assert excinfo.value.provider == "anthropic"


def test_encode_image_io_error(anthropic_llm):
    """Test error handling when image file can't be read."""
    with patch("os.path.exists", return_value=True), patch(
        "builtins.open"
    ) as mock_file:

        # Configure open to raise IOError
        mock_file.side_effect = OSError("Permission denied")

        # Call _encode_image
        with pytest.raises(ProviderError) as excinfo:
            anthropic_llm._encode_image("/path/to/test_image.jpg")

        # Verify error details
        assert "Failed to read image file" in str(excinfo.value)
        assert "Permission denied" in str(excinfo.value.details["error"])
        assert excinfo.value.provider == "anthropic"


def test_encode_image_format_error(anthropic_llm):
    """Test error handling when image format is invalid."""
    with patch("os.path.exists", return_value=True), patch(
        "builtins.open", mock_open(read_data=b"invalid_data")
    ), patch("PIL.Image.open") as mock_open_image:

        # Configure PIL.Image.open to raise an error
        mock_open_image.side_effect = Exception("Invalid image format")

        # Call _encode_image
        with pytest.raises(FormatError) as excinfo:
            anthropic_llm._encode_image("/path/to/test_image.jpg")

        # Verify error details
        assert "Failed to process image" in str(excinfo.value)
        assert "Invalid image format" in str(excinfo.value.details["error"])
        assert excinfo.value.provider == "anthropic"


def test_download_image_from_url(anthropic_llm):
    """Test downloading and encoding an image from URL."""
    # Create a mock image
    mock_img = MagicMock(spec=Image.Image)
    mock_img.mode = "RGB"
    mock_img.size = (100, 100)

    # Mock BytesIO output with fake image data
    fake_image_data = b"fake_jpeg_image_data"

    with patch("requests.get") as mock_get, patch(
        "PIL.Image.open", return_value=mock_img
    ) as mock_open_image, patch.object(
        BytesIO, "getvalue", return_value=fake_image_data
    ):

        # Configure requests.get mock
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b"image_data_from_url"
        mock_get.return_value = mock_response

        # Configure mock_img.save to write to BytesIO
        def mock_save(output, format, quality):
            # Just use the already prepared BytesIO
            pass

        mock_img.save = mock_save

        # Call _download_image_from_url
        result = anthropic_llm._download_image_from_url("https://example.com/image.jpg")

        # Verify requests.get was called correctly
        mock_get.assert_called_once_with("https://example.com/image.jpg", timeout=10)

        # Verify PIL.Image.open was called with BytesIO containing response content
        assert mock_open_image.call_args[0][0].getvalue() == b"image_data_from_url"

        # Verify result is base64 encoded
        expected_base64 = base64.b64encode(fake_image_data).decode("utf-8")
        assert result == expected_base64


def test_download_image_from_url_network_error(anthropic_llm):
    """Test error handling for URL download network errors."""
    with patch("requests.get") as mock_get:
        # Configure requests.get to raise network error
        mock_get.side_effect = Exception("Network connection error")

        # Call _download_image_from_url
        with pytest.raises(ProviderError) as excinfo:
            anthropic_llm._download_image_from_url("https://example.com/image.jpg")

        # Verify error details
        assert "Failed to download image from URL" in str(excinfo.value)
        assert "Network connection error" in str(excinfo.value.details["error"])
        assert excinfo.value.provider == "anthropic"


def test_download_image_from_url_http_error(anthropic_llm):
    """Test error handling for HTTP errors during URL download."""
    with patch("requests.get") as mock_get:
        # Configure requests.get mock to return 404
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        # Call _download_image_from_url
        with pytest.raises(ProviderError) as excinfo:
            anthropic_llm._download_image_from_url("https://example.com/image.jpg")

        # Verify error details
        assert "Failed to download image from URL" in str(excinfo.value)
        assert "404" in str(excinfo.value.details["status_code"])
        assert excinfo.value.provider == "anthropic"


def test_download_image_from_url_format_error(anthropic_llm):
    """Test error handling for image format errors after URL download."""
    with patch("requests.get") as mock_get, patch("PIL.Image.open") as mock_open_image:

        # Configure requests.get mock to return success
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b"invalid_image_data"
        mock_get.return_value = mock_response

        # Configure PIL.Image.open to raise format error
        mock_open_image.side_effect = Exception("Invalid image format")

        # Call _download_image_from_url
        with pytest.raises(FormatError) as excinfo:
            anthropic_llm._download_image_from_url("https://example.com/image.jpg")

        # Verify error details
        assert "Failed to process image from URL" in str(excinfo.value)
        assert "Invalid image format" in str(excinfo.value.details["error"])
        assert excinfo.value.provider == "anthropic"


def test_message_formatting_with_images(anthropic_llm):
    """Test message formatting with image paths and URLs."""
    # Mock for image encoding functions
    with patch.object(
        anthropic_llm, "_encode_image", return_value="base64_local_image"
    ), patch.object(
        anthropic_llm, "_download_image_from_url", return_value="base64_url_image"
    ):

        # Create messages with both image paths and URLs
        messages = [
            {
                "message_type": "human",
                "message": "What's in these images?",
                "image_paths": ["/path/to/local1.jpg", "/path/to/local2.jpg"],
                "image_urls": [
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg",
                ],
            }
        ]

        # Format messages
        formatted = anthropic_llm._format_messages_for_model(messages)

        # Verify the structure
        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"

        # Extract content parts
        content_parts = formatted[0]["content"]

        # Verify image parts
        image_parts = [p for p in content_parts if p["type"] == "image"]
        assert len(image_parts) == 4  # 2 local + 2 URLs

        # Verify local images were processed correctly
        local_images = image_parts[:2]  # First two should be from local paths
        for img in local_images:
            assert img["source"]["type"] == "base64"
            assert img["source"]["media_type"] == "image/jpeg"
            assert img["source"]["data"] == "base64_local_image"

        # Verify URL images were processed correctly
        url_images = image_parts[2:]  # Last two should be from URLs
        for img in url_images:
            assert img["source"]["type"] == "base64"
            assert img["source"]["media_type"] == "image/jpeg"
            assert img["source"]["data"] == "base64_url_image"

        # Verify text part is included
        text_parts = [p for p in content_parts if p["type"] == "text"]
        assert len(text_parts) == 1
        assert text_parts[0]["text"] == "What's in these images?"


def test_generate_with_images(anthropic_llm):
    """Test generating a response with images."""
    # Mock for API requests and image processing
    with patch.object(
        anthropic_llm, "_encode_image", return_value="base64_encoded_image"
    ), patch("requests.post") as mock_post:

        # Configure mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "I see a cat in the image."}],
            "usage": {
                "input_tokens": 20,  # Higher due to image tokens
                "output_tokens": 10,
            },
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Call generate with image
        with patch("os.path.exists", return_value=True):
            response, usage, _ = anthropic_llm.generate(
                event_id="test-images",
                system_prompt="Describe what you see in the images.",
                messages=[
                    {
                        "message_type": "human",
                        "message": "What's in this image?",
                        "image_paths": ["/path/to/cat.jpg"],
                    }
                ],
            )

        # Verify response
        assert "I see a cat in the image" in response

        # Verify token usage includes image costs
        assert usage["images"] == 1
        assert "image_cost" in usage

        # Verify API was called with image data
        call_args = mock_post.call_args[1]
        request_data = json.loads(call_args["data"])

        # Find user message in request
        user_message = next(m for m in request_data["messages"] if m["role"] == "user")

        # Verify image content is included
        image_parts = [p for p in user_message["content"] if p["type"] == "image"]
        assert len(image_parts) == 1
        assert image_parts[0]["source"]["data"] == "base64_encoded_image"


def test_multiple_images_cost_calculation(anthropic_llm):
    """Test cost calculation with multiple images."""
    # Mock for API requests and image processing
    with patch.object(
        anthropic_llm, "_encode_image", return_value="base64_encoded_image"
    ), patch.object(
        anthropic_llm, "_download_image_from_url", return_value="base64_url_image"
    ), patch(
        "requests.post"
    ) as mock_post:

        # Configure mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": "I see multiple objects in these images."}
            ],
            "usage": {
                "input_tokens": 50,  # Higher due to multiple images
                "output_tokens": 10,
            },
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Call generate with multiple images
        with patch("os.path.exists", return_value=True):
            response, usage, _ = anthropic_llm.generate(
                event_id="test-multiple-images",
                system_prompt="Describe what you see in the images.",
                messages=[
                    {
                        "message_type": "human",
                        "message": "What do you see in these images?",
                        "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
                        "image_urls": ["https://example.com/image3.jpg"],
                    }
                ],
            )

        # Verify response
        assert "multiple objects" in response

        # Verify token usage includes cost for 3 images
        assert usage["images"] == 3
        assert usage["image_cost"] > 0

        # Get the image cost for this model
        model_costs = anthropic_llm.get_model_costs()
        image_cost_per_image = model_costs["image_cost"] or 0

        # Verify total image cost is correct
        expected_image_cost = 3 * image_cost_per_image
        assert abs(usage["image_cost"] - expected_image_cost) < 0.0001
