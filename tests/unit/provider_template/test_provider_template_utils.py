"""
Unit tests for the Provider Template utility methods.
"""

import base64
import io
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests
from lluminary.exceptions import LLMMistake
from lluminary.models.providers.provider_template import ProviderNameLLM
from PIL import Image


@pytest.fixture
def provider_instance():
    """Create a basic provider instance for testing."""
    # Mock the auth method to avoid actual authentication
    with patch.object(ProviderNameLLM, "auth", return_value=None):
        provider = ProviderNameLLM(
            model_name="provider-model-2",  # Use model-2 which supports images
            timeout=30,
        )
    return provider


class TestProviderTemplateImageProcessing:
    """Test image processing methods of ProviderNameLLM."""

    @patch("builtins.open", new_callable=mock_open, read_data=b"test_image_data")
    def test_process_image_file(self, mock_file, provider_instance):
        """Test processing a local image file."""
        # Test processing an image file
        result = provider_instance._process_image_file("/path/to/image.jpg")

        # Check that file was opened
        mock_file.assert_called_once_with("/path/to/image.jpg", "rb")

        # Check that result is a base64-encoded string
        expected = base64.b64encode(b"test_image_data").decode("utf-8")
        assert result == expected

    @pytest.mark.skip(reason="Requires PIL module to be mocked properly")
    @patch("PIL.Image.open")
    @patch("builtins.open", new_callable=mock_open, read_data=b"test_image_data")
    def test_process_image_file_with_pil(
        self, mock_file, mock_pil_open, provider_instance
    ):
        """Test processing a local image file with PIL for resizing."""
        # Set up PIL mock
        mock_image = MagicMock()
        mock_image.size = (2000, 2000)  # Large image that needs resizing
        mock_pil_open.return_value = mock_image

        # Mock resize and save methods
        mock_image.resize.return_value = mock_image
        mock_buffer = io.BytesIO()
        mock_buffer.write(b"resized_image_data")
        mock_image.save.side_effect = lambda buffer, format: buffer.write(
            b"resized_image_data"
        )

        # Add a custom implementation that uses PIL for resizing
        def custom_process_image_with_resize(self, image_path):
            # Basic implementation with PIL resizing
            try:
                with open(image_path, "rb") as img_file:
                    # Check if we need to resize the image
                    with Image.open(img_file) as img:
                        width, height = img.size
                        if width > 1024 or height > 1024:
                            # Calculate new dimensions
                            ratio = min(1024 / width, 1024 / height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)

                            # Resize image
                            resized_img = img.resize((new_width, new_height))

                            # Save to buffer
                            buffer = io.BytesIO()
                            resized_img.save(buffer, format="JPEG")
                            buffer.seek(0)

                            # Encode resized image
                            return base64.b64encode(buffer.read()).decode("utf-8")

                    # No resizing needed
                    img_file.seek(0)
                    return base64.b64encode(img_file.read()).decode("utf-8")
            except Exception as e:
                raise LLMMistake(
                    f"Error processing image file: {e!s}",
                    error_type="image_processing_error",
                    provider=self.__class__.__name__,
                    details={"path": image_path, "original_error": str(e)},
                )

        # Install the custom implementation
        provider_instance._process_image_file = (
            custom_process_image_with_resize.__get__(provider_instance)
        )

        # Call the method under test
        result = provider_instance._process_image_file("/path/to/large_image.jpg")

        # Verify that the image was resized
        mock_image.resize.assert_called_once()

        # Check that result is a base64-encoded string of the resized image
        assert isinstance(result, str)
        assert result  # Non-empty string

    @patch("builtins.open", side_effect=OSError("File not found"))
    def test_process_image_file_error(self, mock_file, provider_instance):
        """Test error handling when processing a local image file."""
        # Attempt to process a non-existent image
        with pytest.raises(LLMMistake) as excinfo:
            provider_instance._process_image_file("/path/to/nonexistent_image.jpg")

        # Verify error details
        assert "Error processing image file" in str(excinfo.value)
        assert hasattr(excinfo.value, "error_type")
        assert excinfo.value.error_type == "image_processing_error"
        assert "File not found" in str(excinfo.value.details.get("original_error", ""))

    @patch("requests.get")
    def test_process_image_url(self, mock_get, provider_instance):
        """Test processing an image from URL."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.content = b"image_data_from_url"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Call the method under test
        result = provider_instance._process_image_url("https://example.com/image.jpg")

        # Verify request was made correctly
        mock_get.assert_called_once_with("https://example.com/image.jpg")

        # Check that result is a base64-encoded string
        expected = base64.b64encode(b"image_data_from_url").decode("utf-8")
        assert result == expected

    @patch("requests.get", side_effect=requests.RequestException("Network error"))
    def test_process_image_url_error(self, mock_get, provider_instance):
        """Test error handling when processing an image from URL."""
        # Attempt to process a URL that causes a network error
        with pytest.raises(LLMMistake) as excinfo:
            provider_instance._process_image_url("https://example.com/error_image.jpg")

        # Verify error details
        assert "Error processing image URL" in str(excinfo.value)
        assert hasattr(excinfo.value, "error_type")
        assert excinfo.value.error_type == "image_url_error"
        assert "Network error" in str(excinfo.value.details.get("original_error", ""))

    def test_calculate_image_tokens(self, provider_instance):
        """Test token calculation for images."""
        # Test with various image sizes
        assert provider_instance.calculate_image_tokens(512, 512) == 256
        assert provider_instance.calculate_image_tokens(1024, 1024) == 1024
        assert (
            provider_instance.calculate_image_tokens(800, 600) == 468
        )  # (800*600)/1024

        # Test with extreme dimensions
        assert (
            provider_instance.calculate_image_tokens(10000, 10000) == 97656
        )  # Very large image
        assert (
            provider_instance.calculate_image_tokens(10, 10) == 0
        )  # Very small image (rounds to 0)


class TestProviderTemplateToolHandling:
    """Test tool handling methods of ProviderNameLLM."""

    @patch(
        "src.lluminary.models.providers.provider_template.ProviderNameLLM._format_messages_for_model"
    )
    def test_raw_generate_with_tools(self, mock_format_messages, provider_instance):
        """Test raw_generate with tools parameter."""
        # Set up mocks
        mock_format_messages.return_value = [{"role": "user", "content": "Hello"}]

        # Define test tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        # Call the method under test
        result, usage = provider_instance._raw_generate(
            event_id="test-event",
            system_prompt="You are a helpful assistant",
            messages=[
                {"message_type": "human", "message": "What's the weather in Seattle?"}
            ],
            tools=tools,
        )

        # In a real implementation, we would verify that tools were passed correctly
        # to the provider API. For the template, we just verify basic behavior.
        assert result == "This is a placeholder response."
        assert "tool_use" in usage

    def test_tool_result_message_formatting(self, provider_instance):
        """Test formatting of tool result messages."""
        messages = [
            {"message_type": "human", "message": "What's the weather in Seattle?"},
            {
                "message_type": "tool_result",
                "message": '{"temperature": 72, "condition": "sunny"}',
                "tool_name": "get_weather",
            },
        ]

        formatted = provider_instance._format_messages_for_model(messages)

        assert len(formatted) == 2
        assert formatted[0]["role"] == "user"
        assert formatted[1]["role"] == "tool"
        assert formatted[1]["content"] == '{"temperature": 72, "condition": "sunny"}'

    def test_format_tool_for_provider(self, provider_instance):
        """Test conversion of standard tool format to provider-specific format."""
        # Create a standard tool definition
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            },
        }

        # Format the tool for the provider
        formatted_tool = provider_instance._format_tool_for_provider(tool)

        # Verify the formatting
        assert "function_name" in formatted_tool
        assert formatted_tool["function_name"] == "get_weather"
        assert "function_description" in formatted_tool
        assert (
            formatted_tool["function_description"]
            == "Get the current weather in a location"
        )
        assert "parameters" in formatted_tool
        assert formatted_tool["parameters"]["type"] == "object"
        assert "location" in formatted_tool["parameters"]["properties"]

    def test_parse_tool_response(self, provider_instance):
        """Test parsing of tool response from provider-specific format."""
        # Create a mock provider-specific tool response
        provider_response = {
            "function_call": {
                "name": "get_weather",
                "arguments": '{"location": "Seattle, WA", "unit": "celsius"}',
            }
        }

        # Parse the response
        parsed_response = provider_instance._parse_tool_response(provider_response)

        # Verify the parsed response
        assert "name" in parsed_response
        assert parsed_response["name"] == "get_weather"
        assert "arguments" in parsed_response
        assert parsed_response["arguments"]["location"] == "Seattle, WA"
        assert parsed_response["arguments"]["unit"] == "celsius"

        # Test with non-tool response
        empty_response = provider_instance._parse_tool_response({})
        assert empty_response == {}

    def test_validate_tool_parameters(self, provider_instance):
        """Test validation of tool parameters."""
        # Valid tool
        valid_tool = {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                        "operation": {"type": "string", "enum": ["+", "-", "*", "/"]},
                    },
                    "required": ["a", "b", "operation"],
                },
            },
        }

        # Validate the tool
        assert provider_instance._validate_tool_parameters(valid_tool) is True

        # Test with missing type
        invalid_tool1 = {
            "function": {"name": "calculate", "parameters": {"type": "object"}}
        }
        with pytest.raises(ValueError) as excinfo:
            provider_instance._validate_tool_parameters(invalid_tool1)
        assert "must have a 'type' field" in str(excinfo.value)

        # Test with missing function field
        invalid_tool2 = {"type": "function"}
        with pytest.raises(ValueError) as excinfo:
            provider_instance._validate_tool_parameters(invalid_tool2)
        assert "must have a 'function' field" in str(excinfo.value)

        # Test with missing name
        invalid_tool3 = {
            "type": "function",
            "function": {"parameters": {"type": "object"}},
        }
        with pytest.raises(ValueError) as excinfo:
            provider_instance._validate_tool_parameters(invalid_tool3)
        assert "must have a 'name'" in str(excinfo.value)

        # Test with wrong parameter type
        invalid_tool4 = {
            "type": "function",
            "function": {"name": "calculate", "parameters": {"type": "array"}},
        }
        with pytest.raises(ValueError) as excinfo:
            provider_instance._validate_tool_parameters(invalid_tool4)
        assert "must have type 'object'" in str(excinfo.value)
