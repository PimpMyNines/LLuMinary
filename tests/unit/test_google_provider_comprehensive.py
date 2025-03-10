"""
Comprehensive unit tests for the Google Gemini provider implementation.
"""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from lluminary.models.providers.google import GoogleLLM


class MockGoogleLLM(GoogleLLM):
    """Mock implementation of GoogleLLM for testing."""

    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """Mock implementation of abstract method."""
        pass


@pytest.fixture
def mock_genai_types():
    """Mock the Google Gemini types."""
    with patch("google.genai.types") as mock_types:
        # Mock Content class
        mock_content = MagicMock()
        mock_types.Content.return_value = mock_content

        # Mock Part class
        mock_part = MagicMock()
        mock_types.Part.from_text.return_value = mock_part
        mock_types.Part.from_function_call.return_value = mock_part
        mock_types.Part.from_function_response.return_value = mock_part

        # Mock GenerateContentConfig
        mock_config = MagicMock()
        mock_types.GenerateContentConfig.return_value = mock_config

        yield mock_types


# Removed unused fixtures


@pytest.fixture
def google_llm():
    """Create a Google LLM instance with mocks."""
    # Create patches for auth
    with patch("lluminary.models.providers.google.get_secret") as mock_get_secret:
        with patch("google.genai.Client") as mock_client:
            # Configure mocks
            mock_get_secret.return_value = {"api_key": "test-api-key"}
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance

            # Set up mock generation response
            mock_response = MagicMock()
            mock_response.text = "Generated text from Google Gemini"
            mock_response.usage_metadata = MagicMock()
            mock_response.usage_metadata.prompt_token_count = 10
            mock_response.usage_metadata.candidates_token_count = 5
            mock_response.usage_metadata.total_token_count = 15

            mock_client_instance.models = MagicMock()
            mock_client_instance.models.generate_content = MagicMock(
                return_value=mock_response
            )

            # Create and initialize LLM
            llm = MockGoogleLLM("gemini-2.0-flash")
            llm.auth()  # Initialize with mock client

            yield llm


class TestGoogleLLMInitialization:
    """Test the initialization of the GoogleLLM class."""

    def test_init(self):
        """Test basic initialization."""
        with patch.object(GoogleLLM, "auth"):
            llm = MockGoogleLLM("gemini-2.0-flash")
            assert llm.model_name == "gemini-2.0-flash"
            assert llm.client is None  # Client not initialized until auth() is called

    def test_init_with_kwargs(self):
        """Test initialization with additional kwargs."""
        with patch.object(GoogleLLM, "auth"):
            llm = MockGoogleLLM(
                "gemini-2.0-flash", api_key="test-key", custom_param="value"
            )
            assert llm.model_name == "gemini-2.0-flash"
            assert llm.config["api_key"] == "test-key"
            assert llm.config["custom_param"] == "value"

    def test_supported_models(self):
        """Test that the supported models list is properly defined."""
        expected_models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite-preview-02-05",
            "gemini-2.0-pro-exp-02-05",
            "gemini-2.0-flash-thinking-exp-01-21",
        ]
        assert sorted(GoogleLLM.SUPPORTED_MODELS) == sorted(expected_models)

    def test_context_window(self):
        """Test that context windows are properly defined for all models."""
        for model in GoogleLLM.SUPPORTED_MODELS:
            assert model in GoogleLLM.CONTEXT_WINDOW
            assert GoogleLLM.CONTEXT_WINDOW[model] > 0

    def test_cost_per_model(self):
        """Test that cost information is properly defined for all models."""
        for model in GoogleLLM.SUPPORTED_MODELS:
            assert model in GoogleLLM.COST_PER_MODEL
            model_costs = GoogleLLM.COST_PER_MODEL[model]
            assert "read_token" in model_costs
            assert "write_token" in model_costs
            assert "image_cost" in model_costs
            assert model_costs["read_token"] > 0
            assert model_costs["write_token"] > 0
            assert model_costs["image_cost"] > 0


class TestGoogleLLMAuthentication:
    """Test the authentication of the GoogleLLM class."""

    def test_auth_standard_model(self):
        """Test authentication with a standard model."""
        # Create patches
        with patch("lluminary.models.providers.google.get_secret") as mock_get_secret:
            with patch("google.genai.Client") as mock_client:
                # Configure mocks
                mock_get_secret.return_value = {"api_key": "test-api-key"}
                mock_client_instance = MagicMock()
                mock_client.return_value = mock_client_instance

                # Create and authenticate LLM
                llm = MockGoogleLLM("gemini-2.0-flash")
                llm.auth()

                # Verify get_secret was called correctly
                mock_get_secret.assert_called_once_with(
                    "google_api_key", required_keys=["api_key"]
                )

                # Verify client was initialized correctly for standard model
                mock_client.assert_called_once_with(api_key="test-api-key")
                assert llm.client is not None

    def test_auth_thinking_model(self):
        """Test authentication with a thinking model (requires alpha API)."""
        # Create patches
        with patch("lluminary.models.providers.google.get_secret") as mock_get_secret:
            with patch("google.genai.Client") as mock_client:
                # Configure mocks
                mock_get_secret.return_value = {"api_key": "test-api-key"}
                mock_client_instance = MagicMock()
                mock_client.return_value = mock_client_instance

                # Create and authenticate LLM
                llm = MockGoogleLLM("gemini-2.0-flash-thinking-exp-01-21")
                llm.auth()

                # Verify get_secret was called correctly
                mock_get_secret.assert_called_once_with(
                    "google_api_key", required_keys=["api_key"]
                )

                # Verify client was initialized with alpha API version
                mock_client.assert_called_once_with(
                    api_key="test-api-key", http_options={"api_version": "v1alpha"}
                )
                assert llm.client is not None

    def test_auth_error(self):
        """Test authentication error handling."""
        # Create patch for get_secret
        with patch("lluminary.models.providers.google.get_secret") as mock_get_secret:
            # Make get_secret raise an exception
            mock_get_secret.side_effect = Exception("Failed to get secret")

            llm = MockGoogleLLM("gemini-2.0-flash")

            # Authentication should raise an exception
            with pytest.raises(Exception) as exc_info:
                llm.auth()

            assert "Google authentication failed" in str(exc_info.value)


class TestGoogleLLMImageProcessing:
    """Test the image processing methods of the GoogleLLM class."""

    def test_process_local_image(self, google_llm):
        """Test processing a local image file."""
        # Mock PIL.Image.open
        mock_image = MagicMock()
        with patch("PIL.Image.open", return_value=mock_image) as mock_open:
            result = google_llm._process_image("/path/to/image.jpg")

            # Verify Image.open was called with the correct path
            mock_open.assert_called_once_with("/path/to/image.jpg")
            assert result == mock_image

    def test_process_url_image(self, google_llm):
        """Test processing an image from a URL."""
        # Mock requests.get response
        mock_response = MagicMock()
        mock_response.content = b"mock_image_bytes"

        # Mock PIL Image
        mock_image = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("PIL.Image.open", return_value=mock_image) as mock_open:
                result = google_llm._process_image(
                    "https://example.com/image.jpg", is_url=True
                )

                # Verify requests.get was called with the correct URL
                mock_get.assert_called_once_with(
                    "https://example.com/image.jpg", timeout=10
                )

                # Verify PIL.Image.open was called with BytesIO containing the response content
                assert mock_open.call_count == 1
                # Cannot directly check BytesIO equality, so check that it was called
                assert result == mock_image

    def test_process_local_image_error(self, google_llm):
        """Test error handling when processing a local image fails."""
        # Make PIL.Image.open raise an exception
        with patch("PIL.Image.open", side_effect=Exception("Failed to open image")):
            with pytest.raises(Exception) as exc_info:
                google_llm._process_image("/path/to/bad_image.jpg")

            assert "Failed to process image file" in str(exc_info.value)
            assert "/path/to/bad_image.jpg" in str(exc_info.value)

    def test_process_url_image_error(self, google_llm):
        """Test error handling when processing an image URL fails."""
        # Make requests.get raise an exception
        with patch("requests.get", side_effect=Exception("Failed to fetch image")):
            with pytest.raises(Exception) as exc_info:
                google_llm._process_image(
                    "https://example.com/bad_image.jpg", is_url=True
                )

            assert "Failed to process image URL" in str(exc_info.value)
            assert "https://example.com/bad_image.jpg" in str(exc_info.value)


class TestGoogleLLMMessageFormatting:
    """Test the message formatting methods of the GoogleLLM class."""

    def test_format_human_message(self, google_llm, mock_genai_types):
        """Test formatting a human message."""
        # Create a simple human message
        message = {
            "message_type": "human",
            "message": "Hello, how are you?",
            "image_paths": [],
            "image_urls": [],
        }

        # Format the message
        formatted = google_llm._format_messages_for_model([message])

        # Verify the formatting
        assert len(formatted) == 1
        mock_content = formatted[0]

        # Check role
        assert mock_content.role == "user"

        # Verify Part.from_text was called with the correct text
        mock_genai_types.Part.from_text.assert_called_once_with(
            text="Hello, how are you?"
        )

        # Verify parts were set on the Content
        assert mock_content.parts == [mock_genai_types.Part.from_text.return_value]

    def test_format_ai_message(self, google_llm, mock_genai_types):
        """Test formatting an AI message."""
        # Create an AI message
        message = {
            "message_type": "ai",
            "message": "I'm an AI assistant.",
            "image_paths": [],
            "image_urls": [],
        }

        # Format the message
        formatted = google_llm._format_messages_for_model([message])

        # Verify the formatting
        assert len(formatted) == 1
        mock_content = formatted[0]

        # Check role
        assert mock_content.role == "model"

        # Verify Part.from_text was called with the correct text
        mock_genai_types.Part.from_text.assert_called_once_with(
            text="I'm an AI assistant."
        )

    def test_format_tool_message(self, google_llm, mock_genai_types):
        """Test formatting a tool message."""
        # Create a tool message
        message = {
            "message_type": "tool",
            "message": "Tool response",
            "tool_result": {
                "tool_id": "weather",
                "success": True,
                "result": {"temperature": 72, "condition": "sunny"},
            },
        }

        # Format the message
        formatted = google_llm._format_messages_for_model([message])

        # Verify the formatting
        assert len(formatted) == 1
        mock_content = formatted[0]

        # Check role
        assert mock_content.role == "tool"

        # Verify Part.from_text was called with the correct text
        mock_genai_types.Part.from_text.assert_called_once_with(text="Tool response")

        # Verify from_function_response was called correctly
        mock_genai_types.Part.from_function_response.assert_called_once()
        function_call_args = mock_genai_types.Part.from_function_response.call_args[1]
        assert function_call_args["name"] == "weather"
        assert "result" in function_call_args["response"]
        assert function_call_args["response"]["result"] == {
            "temperature": 72,
            "condition": "sunny",
        }

    def test_format_tool_error_message(self, google_llm, mock_genai_types):
        """Test formatting a tool error message."""
        # Create a tool error message
        message = {
            "message_type": "tool",
            "message": "Tool error",
            "tool_result": {
                "tool_id": "weather",
                "success": False,
                "error": "Location not found",
            },
        }

        # Format the message
        formatted = google_llm._format_messages_for_model([message])

        # Verify the formatting
        assert len(formatted) == 1
        mock_content = formatted[0]

        # Verify from_function_response was called correctly with error
        mock_genai_types.Part.from_function_response.assert_called_once()
        function_call_args = mock_genai_types.Part.from_function_response.call_args[1]
        assert function_call_args["name"] == "weather"
        assert "error" in function_call_args["response"]
        assert function_call_args["response"]["error"] == "Location not found"

    def test_format_tool_use_message(self, google_llm, mock_genai_types):
        """Test formatting a message with tool_use."""
        # Create a message with tool_use
        message = {
            "message_type": "human",
            "message": "What's the weather like?",
            "tool_use": {"name": "get_weather", "input": {"location": "New York"}},
        }

        # Format the message
        formatted = google_llm._format_messages_for_model([message])

        # Verify from_function_call was called correctly
        mock_genai_types.Part.from_function_call.assert_called_once()
        function_call_args = mock_genai_types.Part.from_function_call.call_args[1]
        assert function_call_args["name"] == "get_weather"
        assert function_call_args["args"] == {"location": "New York"}

    def test_format_message_with_images(self, google_llm, mock_genai_types):
        """Test formatting a message with images."""
        # Create a message with images
        message = {
            "message_type": "human",
            "message": "What's in these images?",
            "image_paths": ["/path/to/image1.jpg"],
            "image_urls": ["https://example.com/image2.jpg"],
        }

        # Mock _process_image to return a mock image
        mock_image = MagicMock()
        with patch.object(google_llm, "_process_image", return_value=mock_image):
            # Format the message
            formatted = google_llm._format_messages_for_model([message])

            # Verify the formatting
            assert len(formatted) == 1
            mock_content = formatted[0]

            # Check that _process_image was called twice (once for each image)
            assert google_llm._process_image.call_count == 2

            # Verify the parts contains the text and two images
            assert len(mock_content.parts) == 3

            # Verify Part.from_text was called
            mock_genai_types.Part.from_text.assert_called_once_with(
                text="What's in these images?"
            )

    def test_format_image_processing_error(self, google_llm, mock_genai_types):
        """Test error handling when image processing fails."""
        # Create a message with images
        message = {
            "message_type": "human",
            "message": "What's in this image?",
            "image_paths": ["/path/to/bad_image.jpg"],
            "image_urls": [],
        }

        # Mock _process_image to raise an exception
        with patch.object(
            google_llm,
            "_process_image",
            side_effect=Exception("Failed to process image"),
        ):
            # Formatting should raise an exception
            with pytest.raises(Exception) as exc_info:
                google_llm._format_messages_for_model([message])

            assert "Failed to process image file" in str(exc_info.value)


class TestGoogleLLMGeneration:
    """Test the generation methods of the GoogleLLM class."""

    def test_raw_generate_basic(self, google_llm, mock_genai_types):
        """Test basic text generation."""
        # Create a simple message
        message = {
            "message_type": "human",
            "message": "Tell me about quantum computing",
            "image_paths": [],
            "image_urls": [],
        }

        # Mock _format_messages_for_model to return a simple list
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Generate text
            response, usage = google_llm._raw_generate(
                event_id="test123",
                system_prompt="You are a helpful assistant",
                messages=[message],
                max_tokens=1000,
                temp=0.7,
            )

            # Verify the response
            assert response == "Generated text from Google Gemini"

            # Verify usage statistics
            assert usage["event_id"] == "test123"
            assert usage["read_tokens"] == 10
            assert usage["write_tokens"] == 5
            assert usage["total_tokens"] == 15
            assert usage["images"] == 0
            assert "read_cost" in usage
            assert "write_cost" in usage
            assert "image_cost" in usage
            assert "total_cost" in usage

            # Verify client was called correctly
            client = google_llm.client
            client.models.generate_content.assert_called_once()

            # Check arguments passed to generate_content
            call_args = client.models.generate_content.call_args[1]
            assert call_args["model"] == "gemini-2.0-flash"
            assert call_args["contents"] == formatted_messages

            # Check config
            config = mock_genai_types.GenerateContentConfig.return_value
            assert config.max_output_tokens == 1000
            assert config.temperature == 0.7
            assert config.system_instruction == "You are a helpful assistant"

    def test_raw_generate_with_tools(self, google_llm, mock_genai_types):
        """Test generation with tools."""
        # Create a simple message
        message = {
            "message_type": "human",
            "message": "What's the weather in New York?",
            "image_paths": [],
            "image_urls": [],
        }

        # Create tools
        tools = [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        }
                    },
                    "required": ["location"],
                },
            }
        ]

        # Mock _format_messages_for_model to return a simple list
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Generate text with tools
            response, usage = google_llm._raw_generate(
                event_id="test123",
                system_prompt="You are a helpful assistant",
                messages=[message],
                tools=tools,
            )

            # Verify config was created correctly with tools
            config = mock_genai_types.GenerateContentConfig.return_value
            assert config.tools == tools
            assert config.automatic_function_calling == {"disable": True}

    def test_raw_generate_with_tool_use(self, google_llm, mock_client):
        """Test generation that returns a tool call."""
        # Create a simple message
        message = {
            "message_type": "human",
            "message": "What's the weather in New York?",
            "image_paths": [],
            "image_urls": [],
        }

        # Create mock response with function calls
        tool_call = MagicMock(
            id="tool123", name="get_weather", args={"location": "New York"}
        )

        mock_response = MagicMock()
        mock_response.text = "I'll check the weather for you"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15
        mock_response.function_calls = [tool_call]

        # Update mock client
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content.return_value = mock_response

        # Mock _format_messages_for_model to return a simple list
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Generate text
            response, usage = google_llm._raw_generate(
                event_id="test123",
                system_prompt="You are a helpful assistant",
                messages=[message],
            )

            # Verify tool use in usage statistics
            assert "tool_use" in usage
            assert usage["tool_use"]["id"] == "tool123"
            assert usage["tool_use"]["name"] == "get_weather"
            assert usage["tool_use"]["input"] == {"location": "New York"}

    def test_raw_generate_with_images(self, google_llm):
        """Test generation with images."""
        # Create a message with images
        message = {
            "message_type": "human",
            "message": "What's in these images?",
            "image_paths": ["/path/to/image1.jpg"],
            "image_urls": ["https://example.com/image2.jpg"],
        }

        # Mock _format_messages_for_model to return a simple list
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Generate text
            response, usage = google_llm._raw_generate(
                event_id="test123",
                system_prompt="",
                messages=[message],
            )

            # Verify image count and cost
            assert usage["images"] == 2
            assert usage["image_cost"] > 0

    def test_raw_generate_missing_client(self, google_llm):
        """Test generation when client is not initialized."""
        # Set client to None
        google_llm.client = None

        # Create a simple message
        message = {
            "message_type": "human",
            "message": "Hello",
            "image_paths": [],
            "image_urls": [],
        }

        # Mock _format_messages_for_model and auth
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            with patch.object(google_llm, "auth") as mock_auth:
                # Generate text
                google_llm._raw_generate(
                    event_id="test123",
                    system_prompt="",
                    messages=[message],
                )

                # Verify auth was called
                mock_auth.assert_called_once()

    def test_raw_generate_error(self, google_llm, mock_client):
        """Test error handling during generation."""
        # Create a simple message
        message = {
            "message_type": "human",
            "message": "Hello",
            "image_paths": [],
            "image_urls": [],
        }

        # Make client raise an exception
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content.side_effect = Exception("API Error")

        # Mock _format_messages_for_model
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Generate text should raise an exception
            with pytest.raises(Exception) as exc_info:
                google_llm._raw_generate(
                    event_id="test123",
                    system_prompt="",
                    messages=[message],
                )

            assert "Google API generation failed" in str(exc_info.value)

    def test_raw_generate_missing_fields(self, google_llm, mock_client):
        """Test generation when response is missing fields."""
        # Create a simple message
        message = {
            "message_type": "human",
            "message": "Hello",
            "image_paths": [],
            "image_urls": [],
        }

        # Create a response missing some fields
        mock_response = MagicMock()
        # No text or usage_metadata fields
        delattr(mock_response, "text")
        delattr(mock_response, "usage_metadata")

        # Update mock client
        mock_instance = mock_client.return_value
        mock_instance.models.generate_content.return_value = mock_response

        # Mock _format_messages_for_model
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Generate text
            response, usage = google_llm._raw_generate(
                event_id="test123",
                system_prompt="",
                messages=[message],
            )

            # Should handle missing fields gracefully
            assert response == ""
            assert usage["read_tokens"] == 0
            assert usage["write_tokens"] == 0
            assert usage["total_tokens"] == 0


class TestGoogleLLMStreaming:
    """Test the streaming methods of the GoogleLLM class."""

    @patch("google.generativeai.GenerativeModel")
    def test_stream_generate_basic(self, mock_generative_model, google_llm):
        """Test basic streaming generation."""
        # Create mock model
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        # Create mock streaming response
        mock_chunk1 = MagicMock(text="Hello")
        mock_chunk2 = MagicMock(text=" world")
        mock_chunk3 = MagicMock(text="!")
        mock_chunk4 = MagicMock()  # Final chunk without text
        mock_chunk4.candidates = [MagicMock(content=MagicMock(parts=[]))]

        mock_model.generate_content.return_value = [
            mock_chunk1,
            mock_chunk2,
            mock_chunk3,
            mock_chunk4,
        ]

        # Create message
        message = {
            "message_type": "human",
            "message": "Generate a greeting",
            "image_paths": [],
            "image_urls": [],
        }

        # Mock _format_messages_for_model
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Stream generate
            chunks = []
            for chunk, usage in google_llm.stream_generate(
                event_id="test123",
                system_prompt="You are a helpful assistant",
                messages=[message],
                max_tokens=1000,
                temp=0.7,
            ):
                chunks.append((chunk, usage))

            # Verify chunks
            assert len(chunks) == 4
            assert chunks[0][0] == "Hello"
            assert chunks[1][0] == " world"
            assert chunks[2][0] == "!"
            assert chunks[3][0] == ""  # Final empty chunk

            # Verify final usage contains complete flag
            assert chunks[-1][1]["is_complete"] is True
            assert "total_cost" in chunks[-1][1]

    @patch("google.generativeai.GenerativeModel")
    def test_stream_generate_with_callback(self, mock_generative_model, google_llm):
        """Test streaming with callback."""
        # Create mock model
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        # Create mock streaming response
        mock_chunk1 = MagicMock(text="Hello")
        mock_chunk2 = MagicMock(text=" world")

        mock_model.generate_content.return_value = [mock_chunk1, mock_chunk2]

        # Create message
        message = {
            "message_type": "human",
            "message": "Generate a greeting",
            "image_paths": [],
            "image_urls": [],
        }

        # Create callback
        callback_called = []

        def callback(chunk, usage):
            callback_called.append((chunk, usage))

        # Mock _format_messages_for_model
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Stream generate with callback
            chunks = []
            for chunk, usage in google_llm.stream_generate(
                event_id="test123",
                system_prompt="",
                messages=[message],
                callback=callback,
            ):
                chunks.append((chunk, usage))

            # Verify callback was called for each chunk
            assert len(callback_called) == len(chunks)
            assert callback_called[0][0] == chunks[0][0]
            assert callback_called[1][0] == chunks[1][0]

    @patch("google.generativeai.GenerativeModel")
    def test_stream_generate_with_function_calls(
        self, mock_generative_model, google_llm
    ):
        """Test streaming with function calls."""
        # Create mock model
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        # Create mock function call response at the end
        mock_function_call = MagicMock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "New York"}

        mock_part = MagicMock()
        mock_part.function_call = mock_function_call

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_final_chunk = MagicMock()
        mock_final_chunk.candidates = [mock_candidate]
        mock_final_chunk.text = ""

        # Create streaming response
        mock_chunk1 = MagicMock(text="I'll check")
        mock_chunk2 = MagicMock(text=" the weather")

        mock_model.generate_content.return_value = [
            mock_chunk1,
            mock_chunk2,
            mock_final_chunk,
        ]

        # Create message
        message = {
            "message_type": "human",
            "message": "What's the weather in New York?",
            "image_paths": [],
            "image_urls": [],
        }

        # Create functions
        def get_weather(location):
            """Get weather for a location."""
            return f"Weather in {location}: Sunny"

        # Mock _format_messages_for_model
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Stream generate with functions
            chunks = []
            for chunk, usage in google_llm.stream_generate(
                event_id="test123",
                system_prompt="",
                messages=[message],
                functions=[get_weather],
            ):
                chunks.append((chunk, usage))

            # Verify function call was detected
            final_usage = chunks[-1][1]
            assert "tool_use" in final_usage
            assert len(final_usage["tool_use"]) > 0

            # Get first tool use
            tool_id = list(final_usage["tool_use"].keys())[0]
            tool_use = final_usage["tool_use"][tool_id]

            assert tool_use["name"] == "get_weather"
            assert "location" in tool_use["arguments"]

    @patch("google.generativeai.GenerativeModel")
    def test_stream_generate_error(self, mock_generative_model, google_llm):
        """Test error handling during streaming."""
        # Create mock model that raises an exception
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Streaming error")
        mock_generative_model.return_value = mock_model

        # Create message
        message = {
            "message_type": "human",
            "message": "Generate a greeting",
            "image_paths": [],
            "image_urls": [],
        }

        # Mock _format_messages_for_model
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Streaming should raise an exception
            with pytest.raises(Exception) as exc_info:
                for _ in google_llm.stream_generate(
                    event_id="test123",
                    system_prompt="",
                    messages=[message],
                ):
                    pass

            assert "Error streaming from Google" in str(exc_info.value)

    @patch("google.generativeai.GenerativeModel")
    def test_stream_generate_system_prompt_cleaning(
        self, mock_generative_model, google_llm
    ):
        """Test that system prompt echoing is cleaned up."""
        # Create mock model
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        # Create response with system prompt echoed
        system_prompt = "You are a helpful assistant."
        mock_chunk = MagicMock(text=f"{system_prompt} Hello")

        mock_model.generate_content.return_value = [mock_chunk]

        # Create message
        message = {
            "message_type": "human",
            "message": "Say hello",
            "image_paths": [],
            "image_urls": [],
        }

        # Mock _format_messages_for_model
        formatted_messages = [MagicMock()]
        with patch.object(
            google_llm, "_format_messages_for_model", return_value=formatted_messages
        ):
            # Stream generate
            chunks = []
            for chunk, usage in google_llm.stream_generate(
                event_id="test123",
                system_prompt=system_prompt,
                messages=[message],
            ):
                chunks.append((chunk, usage))

            # Verify system prompt was stripped
            assert chunks[0][0] == " Hello"

    def test_stream_generate_import_error(self, google_llm):
        """Test handling of missing google.generativeai package."""
        # Mock import error
        with patch("google.generativeai", side_effect=ImportError("Package not found")):
            # Create message
            message = {
                "message_type": "human",
                "message": "Generate a greeting",
                "image_paths": [],
                "image_urls": [],
            }

            # Streaming should raise an ImportError
            with pytest.raises(ImportError) as exc_info:
                for _ in google_llm.stream_generate(
                    event_id="test123",
                    system_prompt="",
                    messages=[message],
                ):
                    pass

            assert "Google Generative AI package not installed" in str(exc_info.value)


class TestGoogleLLMFeatureSupport:
    """Test the feature support methods of the GoogleLLM class."""

    def test_supports_image_input(self, google_llm):
        """Test the supports_image_input method."""
        assert google_llm.supports_image_input() is True
        assert google_llm.supports_image_input() == GoogleLLM.SUPPORTS_IMAGES

    def test_get_supported_models(self, google_llm):
        """Test the get_supported_models method."""
        models = google_llm.get_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gemini-2.0-flash" in models
        assert models == GoogleLLM.SUPPORTED_MODELS

    def test_model_costs(self, google_llm):
        """Test get_model_costs method."""
        # Get costs for current model
        costs = google_llm.get_model_costs()
        assert "read_token" in costs
        assert "write_token" in costs
        assert "image_cost" in costs

        # Check that all models have cost info
        for model in GoogleLLM.SUPPORTED_MODELS:
            # Change the model name temporarily
            original_model = google_llm.model_name
            google_llm.model_name = model

            costs = google_llm.get_model_costs()
            assert "read_token" in costs
            assert "write_token" in costs
            assert "image_cost" in costs

            # Restore the original model name
            google_llm.model_name = original_model
