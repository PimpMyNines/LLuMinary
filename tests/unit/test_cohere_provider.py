"""
Unit tests for the Cohere provider.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from lluminary.exceptions import LLMMistake
from lluminary.models.providers.cohere import CohereLLM


class MockCohereClient:
    """Mock Cohere client for testing."""

    def __init__(self, **kwargs):
        self.chat = MagicMock()
        self.chat.return_value = {
            "text": "This is a test response from Cohere",
            "generation_id": "test-generation-id",
            "meta": {"billed_units": {"input_tokens": 10, "output_tokens": 8}},
            "tool_calls": [],
        }

        self.embed = MagicMock()
        self.embed.return_value = MagicMock(
            embeddings=[[0.1, 0.2, 0.3] for _ in range(2)],
            meta={"billed_units": {"input": 10}},
        )

        self.rerank = MagicMock()
        self.rerank.return_value = MagicMock(
            results=[
                MagicMock(index=1, relevance_score=0.8, document="Second document"),
                MagicMock(index=0, relevance_score=0.6, document="First document"),
            ]
        )

        self.chat_stream = MagicMock()


@pytest.fixture
def mock_cohere_env():
    """Set up mock environment variables for testing."""
    original_env = os.environ.copy()
    os.environ["COHERE_API_KEY"] = "test-api-key"
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_requests_session():
    """Mock requests.Session for Cohere API calls."""
    with patch("requests.Session") as mock_session:
        session_instance = MagicMock()
        session_instance.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(
                return_value={
                    "text": "This is a test response from Cohere",
                    "generation_id": "test-generation-id",
                    "meta": {"billed_units": {"input_tokens": 10, "output_tokens": 8}},
                    "tool_calls": [],
                }
            ),
        )
        mock_session.return_value = session_instance
        yield mock_session


@pytest.fixture
def cohere_llm(mock_cohere_env):
    """Create a CohereLLM instance for testing with mocked auth."""
    with patch.object(CohereLLM, "auth"):
        llm = CohereLLM("command")
        llm.api_key = "test-api-key"
        llm.session = MagicMock()
        llm.session.post.return_value.json.return_value = {
            "text": "This is a test response from Cohere",
            "generation_id": "test-generation-id",
            "meta": {"billed_units": {"input_tokens": 10, "output_tokens": 8}},
            "tool_calls": [],
        }
        llm.session.post.return_value.status_code = 200
        llm.session.post.return_value.raise_for_status = MagicMock()
        return llm


class TestCohereInitialization:
    """Test initialization of the CohereLLM class."""

    @patch("lluminary.models.base.LLM.__init__")
    def test_init(self, mock_base_init):
        """Test initialization of CohereLLM."""
        # Skip calling the parent class init which would call auth()
        mock_base_init.return_value = None

        llm = CohereLLM("command")
        llm.model_name = (
            "command"  # Manually set attributes that would be set by parent init
        )
        llm.provider = "cohere"
        assert llm.model_name == "command"
        assert llm.provider == "cohere"
        assert llm.api_base == "https://api.cohere.ai/v1"
        assert llm.timeout == 60

        # Test with custom parameters
        llm = CohereLLM(
            "command-light", api_base="https://custom-api.cohere.ai", timeout=30
        )
        llm.model_name = (
            "command-light"  # Manually set attributes that would be set by parent init
        )
        llm.provider = "cohere"
        assert llm.model_name == "command-light"
        assert llm.api_base == "https://custom-api.cohere.ai"
        assert llm.timeout == 30

    def test_supported_models_list(self):
        """Test that the supported models list is properly defined."""
        expected_models = ["command", "command-light", "command-r", "command-r-plus"]
        assert set(CohereLLM.SUPPORTED_MODELS) == set(expected_models)

    def test_embedding_models_list(self):
        """Test that the embedding models list is properly defined."""
        expected_models = [
            "embed-english-v3.0",
            "embed-multilingual-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-light-v3.0",
        ]
        assert set(CohereLLM.EMBEDDING_MODELS) == set(expected_models)

    def test_reranking_models_list(self):
        """Test that the reranking models list is properly defined."""
        expected_models = ["rerank-english-v3.0", "rerank-multilingual-v3.0"]
        assert set(CohereLLM.RERANKING_MODELS) == set(expected_models)

    def test_context_window_defined(self):
        """Test that context windows are defined for all models."""
        for model in CohereLLM.SUPPORTED_MODELS:
            assert model in CohereLLM.CONTEXT_WINDOW
            assert CohereLLM.CONTEXT_WINDOW[model] > 0

    def test_cost_per_token_defined(self):
        """Test that cost per token is defined for all models."""
        for model in CohereLLM.SUPPORTED_MODELS:
            assert model in CohereLLM.COST_PER_MODEL
            assert "read_token" in CohereLLM.COST_PER_MODEL[model]
            assert "write_token" in CohereLLM.COST_PER_MODEL[model]


class TestCohereAuthentication:
    """Test authentication for the Cohere provider."""

    @patch("lluminary.models.base.LLM.__init__")
    @patch("requests.Session")
    def test_auth(self, mock_session, mock_base_init, mock_cohere_env):
        """Test authentication with API key."""
        # Skip calling the parent class init which would call auth() again
        mock_base_init.return_value = None

        # Set up mock session
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        # Create instance and call auth
        llm = CohereLLM("command")
        llm.model_name = "command"
        llm.config = {}
        llm.auth()

        # Check results
        assert llm.api_key == "test-api-key"
        mock_session_instance.headers.update.assert_called_once_with(
            {"Authorization": "Bearer test-api-key", "Content-Type": "application/json"}
        )

    @patch("lluminary.models.base.LLM.__init__")
    @patch("requests.Session")
    def test_auth_no_api_key(self, mock_session, mock_base_init):
        """Test authentication fails when no API key is available."""
        # Skip calling the parent class init
        mock_base_init.return_value = None

        # Ensure no API key in environment
        with patch.dict(os.environ, {}, clear=True):
            # Create instance
            llm = CohereLLM("command")
            llm.model_name = "command"
            llm.config = {}

            # Auth should raise ValueError for missing API key
            with pytest.raises(ValueError) as exc_info:
                llm.auth()

            assert "API key not found" in str(exc_info.value)

    @patch("lluminary.models.base.LLM.__init__")
    @patch("requests.Session")
    def test_auth_with_aws_secrets(self, mock_session, mock_base_init):
        """Test authentication using AWS Secrets Manager."""
        # Skip calling the parent class init
        mock_base_init.return_value = None

        # Skip the test if the _get_api_key_from_aws method is not available in the class
        if not hasattr(CohereLLM, "_get_api_key_from_aws"):
            pytest.skip("CohereLLM._get_api_key_from_aws not available")

        # Create a patch for the AWS method
        with patch.object(CohereLLM, "_get_api_key_from_aws") as mock_get_aws:
            # Setup AWS mock
            mock_get_aws.return_value = "aws-secret-key"

            # Ensure no API key in environment
            with patch.dict(os.environ, {}, clear=True):
                # Set up mock session
                mock_session_instance = MagicMock()
                mock_session.return_value = mock_session_instance

                # Create instance with AWS config
                llm = CohereLLM("command", aws_secret_name="cohere_api_key")
                llm.model_name = "command"
                llm.config = {"aws_secret_name": "cohere_api_key"}
                llm.auth()

                # Check results
                assert llm.api_key == "aws-secret-key"
                mock_get_aws.assert_called_once_with("cohere_api_key")
                mock_session_instance.headers.update.assert_called_once_with(
                    {
                        "Authorization": "Bearer aws-secret-key",
                        "Content-Type": "application/json",
                    }
                )


class TestCohereMessageFormatting:
    """Test message formatting for the Cohere provider."""

    def test_format_messages_for_model(self, cohere_llm):
        """Test formatting messages for Cohere."""
        # Set up test messages
        messages = [
            {
                "message_type": "human",
                "message": "Hello, how are you?",
                "image_paths": [],
                "image_urls": [],
            },
            {
                "message_type": "ai",
                "message": "I'm doing well, thank you!",
                "image_paths": [],
                "image_urls": [],
            },
            {
                "message_type": "tool_result",
                "message": "Tool execution result",
                "tool_result": {"result": "42"},
            },
        ]

        # Format the messages
        formatted = cohere_llm._format_messages_for_model(messages)

        # Check the formatted messages
        assert len(formatted) == 3
        assert formatted[0]["role"] == "USER"
        assert formatted[0]["message"] == "Hello, how are you?"
        assert formatted[1]["role"] == "CHATBOT"
        assert formatted[1]["message"] == "I'm doing well, thank you!"
        assert formatted[2]["role"] == "SYSTEM"
        assert "Tool Result" in formatted[2]["message"]
        assert "42" in formatted[2]["message"]

    @patch("lluminary.models.providers.cohere.CohereLLM._process_image_file")
    def test_format_messages_with_images(self, mock_process_image, cohere_llm):
        """Test formatting messages with image attachments."""
        # Make this model support images
        cohere_llm.model_name = "command-r"

        # Mock image processing
        mock_process_image.return_value = "base64-image-data"

        # Set up test message with images
        messages = [
            {
                "message_type": "human",
                "message": "What's in this image?",
                "image_paths": ["/path/to/image1.jpg", "/path/to/image2.png"],
                "image_urls": ["https://example.com/image.jpg"],
            }
        ]

        # Format the messages
        formatted = cohere_llm._format_messages_for_model(messages)

        # Check the formatted messages
        assert len(formatted) == 1
        assert formatted[0]["role"] == "USER"
        assert formatted[0]["message"] == "What's in this image?"
        assert "attachments" in formatted[0]
        assert len(formatted[0]["attachments"]) == 3

        # Check file attachments
        assert formatted[0]["attachments"][0]["source"]["type"] == "base64"
        assert formatted[0]["attachments"][0]["source"]["media_type"] == "image/jpeg"
        assert formatted[0]["attachments"][0]["source"]["data"] == "base64-image-data"

        # Check URL attachment
        assert formatted[0]["attachments"][2]["source"]["type"] == "url"
        assert (
            formatted[0]["attachments"][2]["source"]["url"]
            == "https://example.com/image.jpg"
        )

    @patch("lluminary.models.providers.cohere.CohereLLM._process_image_file")
    def test_format_messages_image_error_handling(self, mock_process_image, cohere_llm):
        """Test error handling when processing images."""
        # Make this model support images
        cohere_llm.model_name = "command-r"

        # Mock image processing failure
        mock_process_image.side_effect = Exception("Image processing failed")

        # Set up test message with images
        messages = [
            {
                "message_type": "human",
                "message": "What's in this image?",
                "image_paths": ["/path/to/image1.jpg"],
                "image_urls": [],
            }
        ]

        # Format should continue despite image error
        formatted = cohere_llm._format_messages_for_model(messages)

        # Check the formatted message (without attachments)
        assert len(formatted) == 1
        assert formatted[0]["role"] == "USER"
        assert formatted[0]["message"] == "What's in this image?"
        assert "attachments" not in formatted[0]


class TestCohereGeneration:
    """Test text generation for the Cohere provider."""

    def test_raw_generate(self, cohere_llm):
        """Test raw generation with Cohere."""
        # Set up test data
        event_id = "test-event"
        system_prompt = "You are a helpful assistant."
        messages = [
            {
                "message_type": "human",
                "message": "Hello, how are you?",
                "image_paths": [],
                "image_urls": [],
            }
        ]

        # Ensure the session is properly mocked
        cohere_llm.session.post.return_value.json.return_value = {
            "text": "This is a test response from Cohere",
            "generation_id": "test-generation-id",
            "meta": {"billed_units": {"input_tokens": 10, "output_tokens": 8}},
            "tool_calls": [],
        }

        # Call the method
        response, usage = cohere_llm._raw_generate(
            event_id=event_id,
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=100,
            temp=0.7,
            tools=None,
        )

        # Check the response
        assert response == "This is a test response from Cohere"

        # Check usage data
        assert usage["event_id"] == event_id
        assert usage["model"] == "command"
        assert usage["read_tokens"] == 10
        assert usage["write_tokens"] == 8
        assert usage["total_tokens"] == 18
        assert "read_cost" in usage
        assert "write_cost" in usage
        assert "total_cost" in usage

        # Verify API call
        cohere_llm.session.post.assert_called_once()
        args, kwargs = cohere_llm.session.post.call_args
        assert args[0].endswith("/chat")
        assert kwargs["json"]["model"] == "command"
        assert kwargs["json"]["message"] == "Hello, how are you?"
        assert kwargs["json"]["max_tokens"] == 100
        assert kwargs["json"]["temperature"] == 0.7
        assert not kwargs["json"]["stream"]
        assert len(kwargs["json"]["chat_history"]) == 1
        assert kwargs["json"]["chat_history"][0]["role"] == "SYSTEM"
        assert (
            kwargs["json"]["chat_history"][0]["message"]
            == "You are a helpful assistant."
        )

    def test_with_functions(self, cohere_llm):
        """Test generation with function calling."""
        # Define a test function schema
        tools = [
            {
                "name": "test_function",
                "description": "Test function for function calling",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "A string argument"},
                        "arg2": {
                            "type": "integer",
                            "description": "An integer argument",
                        },
                    },
                    "required": ["arg1", "arg2"],
                },
            }
        ]

        # Set up test data
        event_id = "test-event"
        system_prompt = "You are a helpful assistant."
        messages = [
            {
                "message_type": "human",
                "message": "Call the test function with 'hello' and 42",
                "image_paths": [],
                "image_urls": [],
            }
        ]

        # Setup mock response with tool calls
        cohere_llm.session.post.return_value.json.return_value = {
            "text": "I'll call the test function",
            "generation_id": "test-generation-id",
            "meta": {"billed_units": {"input_tokens": 15, "output_tokens": 10}},
            "tool_calls": [
                {
                    "name": "test_function",
                    "parameters": {"arg1": "hello", "arg2": 42},
                    "id": "tool-1",
                }
            ],
        }

        # Call the method
        response, usage = cohere_llm._raw_generate(
            event_id=event_id,
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=100,
            temp=0.7,
            tools=tools,
        )

        # Check the response
        assert response == "I'll call the test function"

        # Check usage data
        assert usage["total_tokens"] == 25

        # Check tool calls in usage
        assert "tool_use" in usage
        assert usage["tool_use"]["name"] == "test_function"
        assert usage["tool_use"]["arguments"]["arg1"] == "hello"
        assert usage["tool_use"]["arguments"]["arg2"] == 42

        # Verify API call includes tools
        args, kwargs = cohere_llm.session.post.call_args
        assert "tools" in kwargs["json"]
        assert len(kwargs["json"]["tools"]) == 1
        assert kwargs["json"]["tools"][0]["name"] == "test_function"
        assert "parameter_definitions" in kwargs["json"]["tools"][0]
        assert "arg1" in kwargs["json"]["tools"][0]["parameter_definitions"]
        assert "arg2" in kwargs["json"]["tools"][0]["parameter_definitions"]

    def test_error_handling(self, cohere_llm):
        """Test error handling with Cohere API."""
        # Create a mock HTTP error
        mock_response = MagicMock()
        mock_response.status_code = 401
        http_error = requests.exceptions.HTTPError("401 Client Error")
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        mock_response.json.return_value = {"message": "Invalid API Key"}

        # Replace the session's post method
        cohere_llm.session.post.return_value = mock_response

        # Call generate method - should raise LLMMistake
        with pytest.raises(LLMMistake) as exc_info:
            cohere_llm._raw_generate(
                event_id="test-error",
                system_prompt="You are a helpful assistant.",
                messages=[
                    {
                        "message_type": "human",
                        "message": "Hello, how are you?",
                        "image_paths": [],
                        "image_urls": [],
                    }
                ],
            )

        # Verify error details
        error = exc_info.value
        assert "Cohere API error" in str(error)
        assert error.error_type == "api_error"
        assert error.provider == "cohere"
        assert error.details["status_code"] == 401

    def test_non_http_error_handling(self, cohere_llm):
        """Test handling of non-HTTP errors."""
        # Make the session raise a generic exception
        cohere_llm.session.post.side_effect = Exception("Generic error")

        # Call generate method - should raise LLMMistake
        with pytest.raises(LLMMistake) as exc_info:
            cohere_llm._raw_generate(
                event_id="test-error",
                system_prompt="You are a helpful assistant.",
                messages=[
                    {
                        "message_type": "human",
                        "message": "Hello, how are you?",
                        "image_paths": [],
                        "image_urls": [],
                    }
                ],
            )

        # Verify error details
        error = exc_info.value
        assert "Error generating text with Cohere" in str(error)
        assert error.error_type == "general_error"
        assert error.provider == "cohere"


class TestCohereFeatureSupport:
    """Test feature support detection for Cohere provider."""

    @patch("lluminary.models.base.LLM.__init__")
    def test_supports_image_input(self, mock_base_init):
        """Test detection of image input support."""
        # Skip calling the parent class init
        mock_base_init.return_value = None

        # Command model doesn't support images
        llm = CohereLLM("command")
        llm.model_name = "command"
        assert not llm.supports_image_input()

        # Command-r model supports images
        llm = CohereLLM("command-r")
        llm.model_name = "command-r"
        assert llm.supports_image_input()

        # Command-r-plus also supports images
        llm = CohereLLM("command-r-plus")
        llm.model_name = "command-r-plus"
        assert llm.supports_image_input()

    def test_supports_embeddings(self, cohere_llm):
        """Test detection of embeddings support."""
        assert cohere_llm.supports_embeddings() is True

    def test_supports_reranking(self, cohere_llm):
        """Test detection of reranking support."""
        assert cohere_llm.supports_reranking() is True

    @patch("lluminary.models.base.LLM.__init__")
    def test_get_context_window(self, mock_base_init):
        """Test getting context window size for different models."""
        # Skip calling the parent class init
        mock_base_init.return_value = None

        # Test various models
        for model_name, expected_window in [
            ("command", 4096),
            ("command-light", 4096),
            ("command-r", 128000),
            ("command-r-plus", 128000),
        ]:
            llm = CohereLLM(model_name)
            llm.model_name = model_name
            assert llm.get_context_window() == expected_window


class TestCohereImageProcessing:
    """Test image processing for Cohere provider."""

    @patch("PIL.Image.open")
    def test_process_image_file_jpeg(self, mock_image_open, cohere_llm):
        """Test processing a JPEG image."""
        # Create mock image
        mock_img = MagicMock()
        mock_img.format = "JPEG"
        mock_image_open.return_value.__enter__.return_value = mock_img

        # Mock open for reading the file
        mock_open_data = b"fake-jpeg-data"
        with patch("builtins.open", mock_open := MagicMock()):
            mock_open.return_value.__enter__.return_value.read.return_value = (
                mock_open_data
            )

            # Mock base64 encoding
            with patch("base64.b64encode") as mock_b64encode:
                mock_b64encode.return_value = b"base64-encoded-data"

                # Process image
                result = cohere_llm._process_image_file("/path/to/image.jpg")

                # Verify result
                assert result == "base64-encoded-data"
                mock_image_open.assert_called_once_with("/path/to/image.jpg")
                mock_open.assert_called_once_with("/path/to/image.jpg", "rb")
                mock_b64encode.assert_called_once_with(mock_open_data)

    @patch("PIL.Image.open")
    def test_process_image_file_png(self, mock_image_open, cohere_llm):
        """Test processing a PNG image (conversion to JPEG)."""
        # Create mock image
        mock_img = MagicMock()
        mock_img.format = "PNG"
        mock_img.mode = "RGB"  # Not RGBA
        mock_image_open.return_value.__enter__.return_value = mock_img

        # Mock BytesIO and image saving
        with patch("lluminary.models.providers.cohere.BytesIO") as mock_bytesio:
            mock_buffer = MagicMock()
            mock_bytesio.return_value = mock_buffer
            mock_buffer.read.return_value = b"converted-jpeg-data"

            # Mock base64 encoding
            with patch("base64.b64encode") as mock_b64encode:
                mock_b64encode.return_value = b"base64-encoded-data"

                # Process image
                result = cohere_llm._process_image_file("/path/to/image.png")

                # Verify result
                assert result == "base64-encoded-data"
                mock_image_open.assert_called_once_with("/path/to/image.png")
                mock_img.save.assert_called_once_with(
                    mock_buffer, format="JPEG", quality=90
                )
                mock_b64encode.assert_called_once_with(b"converted-jpeg-data")

    @patch("PIL.Image.open")
    def test_process_image_file_rgba(self, mock_image_open, cohere_llm):
        """Test processing an image with alpha channel."""
        # Create mock image
        mock_img = MagicMock()
        mock_img.format = "PNG"
        mock_img.mode = "RGBA"  # Has alpha channel
        mock_image_open.return_value.__enter__.return_value = mock_img

        # Mock BytesIO and image saving
        with patch("lluminary.models.providers.cohere.BytesIO") as mock_bytesio:
            mock_buffer = MagicMock()
            mock_bytesio.return_value = mock_buffer
            mock_buffer.read.return_value = b"converted-jpeg-data"

            # Mock base64 encoding
            with patch("base64.b64encode") as mock_b64encode:
                mock_b64encode.return_value = b"base64-encoded-data"

                # Process image
                result = cohere_llm._process_image_file("/path/to/image.png")

                # Verify result
                assert result == "base64-encoded-data"
                mock_image_open.assert_called_once_with("/path/to/image.png")
                # Check that convert was called to remove alpha channel
                mock_img.convert.assert_called_once_with("RGB")
                mock_img.convert.return_value.save.assert_called_once_with(
                    mock_buffer, format="JPEG", quality=90
                )
                mock_b64encode.assert_called_once_with(b"converted-jpeg-data")

    @patch("PIL.Image.open")
    def test_process_image_file_error(self, mock_image_open, cohere_llm):
        """Test error handling during image processing."""
        # Mock image open to raise exception
        mock_image_open.side_effect = Exception("Image processing error")

        # Process image should return None on error
        result = cohere_llm._process_image_file("/path/to/image.jpg")
        assert result is None


class TestCohereEmbeddings:
    """Test the embedding functionality."""

    @patch("cohere.Client")
    def test_embed(self, mock_client, cohere_llm):
        """Test embeddings with Cohere."""
        # Set up mock client
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Configure mock response
        embed_response = MagicMock()
        embed_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        client_instance.embed.return_value = embed_response

        # Set the config
        cohere_llm.config = {"api_key": "test-key"}

        # Call embed method
        texts = ["This is the first text", "This is the second text"]
        embeddings, usage = cohere_llm.embed(texts)

        # Verify the results
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert "total_tokens" in usage
        assert "total_cost" in usage
        assert "model" in usage

        # Verify client was called with correct parameters
        client_instance.embed.assert_called_once()
        call_args = client_instance.embed.call_args[1]
        assert call_args["texts"] == texts
        assert call_args["model"] == "embed-english-v3.0"  # Default model

    @patch("cohere.Client")
    def test_embed_with_custom_model(self, mock_client, cohere_llm):
        """Test embeddings with custom model."""
        # Set up mock client
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Configure mock response
        embed_response = MagicMock()
        embed_response.embeddings = [[0.1, 0.2, 0.3]]
        client_instance.embed.return_value = embed_response

        # Set the config
        cohere_llm.config = {"api_key": "test-key"}

        # Call embed method with custom model
        texts = ["Test text"]
        custom_model = "embed-multilingual-v3.0"
        embeddings, usage = cohere_llm.embed(texts, model=custom_model)

        # Verify client was called with correct parameters
        call_args = client_instance.embed.call_args[1]
        assert call_args["model"] == custom_model

    @patch("cohere.Client")
    def test_embed_with_invalid_model(self, mock_client, cohere_llm):
        """Test embeddings with invalid model."""
        # Set up mock client
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Set the config
        cohere_llm.config = {"api_key": "test-key"}

        # Call embed method with invalid model
        texts = ["Test text"]
        invalid_model = "invalid-model"

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            cohere_llm.embed(texts, model=invalid_model)

        assert "Embedding model" in str(exc_info.value)
        assert "not supported" in str(exc_info.value)

    @patch("cohere.Client")
    def test_embed_with_empty_texts(self, mock_client, cohere_llm):
        """Test embeddings with empty input."""
        # Call embed method with empty texts
        texts = []
        embeddings, usage = cohere_llm.embed(texts)

        # Should return empty results without calling client
        assert embeddings == []
        assert usage["total_tokens"] == 0
        assert usage["total_cost"] == 0.0
        assert usage["model"] is None
        mock_client.assert_not_called()

    @patch("cohere.Client")
    def test_embed_error_handling(self, mock_client, cohere_llm):
        """Test error handling during embedding."""
        # Set up mock client to raise exception
        client_instance = MagicMock()
        client_instance.embed.side_effect = Exception("Embedding error")
        mock_client.return_value = client_instance

        # Set the config
        cohere_llm.config = {"api_key": "test-key"}

        # Call embed method
        texts = ["Test text"]

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            cohere_llm.embed(texts)

        assert "Error getting embeddings from Cohere" in str(exc_info.value)


class TestCohereReranking:
    """Test the reranking functionality."""

    @patch("cohere.Client")
    def test_rerank(self, mock_client, cohere_llm):
        """Test reranking with Cohere."""
        # Set up mock client
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Configure mock response
        rerank_results = [
            MagicMock(index=1, relevance_score=0.9, document="second doc"),
            MagicMock(index=0, relevance_score=0.7, document="first doc"),
            MagicMock(index=2, relevance_score=0.5, document="third doc"),
        ]
        rerank_response = MagicMock()
        rerank_response.results = rerank_results
        client_instance.rerank.return_value = rerank_response

        # Set the config
        cohere_llm.config = {"api_key": "test-key"}

        # Call rerank method
        query = "example query"
        documents = ["first doc", "second doc", "third doc"]
        result = cohere_llm.rerank(query, documents)

        # Verify the results
        assert len(result["ranked_documents"]) == 3
        assert result["ranked_documents"][0] == "second doc"  # Highest score
        assert result["indices"] == [1, 0, 2]
        assert len(result["scores"]) == 3
        assert result["scores"][0] == 0.9
        assert "total_tokens" in result["usage"]
        assert "total_cost" in result["usage"]

        # Verify client was called with correct parameters
        client_instance.rerank.assert_called_once()
        call_args = client_instance.rerank.call_args[1]
        assert call_args["query"] == query
        assert call_args["documents"] == documents
        assert call_args["model"] == "rerank-english-v3.0"  # Default model

    @patch("cohere.Client")
    def test_rerank_with_top_n(self, mock_client, cohere_llm):
        """Test reranking with top_n parameter."""
        # Set up mock client
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Configure mock response
        rerank_results = [
            MagicMock(index=1, relevance_score=0.9, document="second doc"),
            MagicMock(index=0, relevance_score=0.7, document="first doc"),
        ]
        rerank_response = MagicMock()
        rerank_response.results = rerank_results
        client_instance.rerank.return_value = rerank_response

        # Set the config
        cohere_llm.config = {"api_key": "test-key"}

        # Call rerank method with top_n
        query = "example query"
        documents = ["first doc", "second doc", "third doc"]
        result = cohere_llm.rerank(query, documents, top_n=2)

        # Verify the results
        assert len(result["ranked_documents"]) == 2
        assert result["indices"] == [1, 0]

        # Verify client was called with correct parameters
        call_args = client_instance.rerank.call_args[1]
        assert call_args["top_n"] == 2
        assert call_args["return_documents"] is True

    @patch("cohere.Client")
    def test_rerank_without_scores(self, mock_client, cohere_llm):
        """Test reranking without returning scores."""
        # Set up mock client
        client_instance = MagicMock()
        mock_client.return_value = client_instance

        # Configure mock response
        rerank_results = [
            MagicMock(index=1, relevance_score=0.9, document="second doc"),
            MagicMock(index=0, relevance_score=0.7, document="first doc"),
        ]
        rerank_response = MagicMock()
        rerank_response.results = rerank_results
        client_instance.rerank.return_value = rerank_response

        # Set the config
        cohere_llm.config = {"api_key": "test-key"}

        # Call rerank method with return_scores=False
        query = "example query"
        documents = ["first doc", "second doc"]
        result = cohere_llm.rerank(query, documents, return_scores=False)

        # Verify the results
        assert result["scores"] is None

    @patch("cohere.Client")
    def test_rerank_with_empty_documents(self, mock_client, cohere_llm):
        """Test reranking with empty documents list."""
        # Call rerank method with empty documents
        query = "example query"
        documents = []
        result = cohere_llm.rerank(query, documents)

        # Should return empty results without calling client
        assert result["ranked_documents"] == []
        assert result["indices"] == []
        assert result["scores"] == []
        assert result["usage"]["total_tokens"] == 0
        assert result["usage"]["total_cost"] == 0.0
        mock_client.assert_not_called()

    @patch("cohere.Client")
    def test_rerank_error_handling(self, mock_client, cohere_llm):
        """Test error handling during reranking."""
        # Set up mock client to raise exception
        client_instance = MagicMock()
        client_instance.rerank.side_effect = Exception("Reranking error")
        mock_client.return_value = client_instance

        # Set the config
        cohere_llm.config = {"api_key": "test-key"}

        # Call rerank method
        query = "example query"
        documents = ["first doc", "second doc"]

        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            cohere_llm.rerank(query, documents)

        assert "Error reranking documents with Cohere" in str(exc_info.value)


class TestCohereClassification:
    """Test the classification functionality."""

    def test_classification(self, cohere_llm):
        """Test classification with Cohere."""
        # Define test categories
        categories = {
            "sports": "Content related to sports and athletics",
            "technology": "Content related to computers, software, and hardware",
            "entertainment": "Content related to movies, music, and pop culture",
        }

        # Test messages
        messages = [
            {"message_type": "human", "message": "The new iPhone was just released"}
        ]

        # Mock the generate method to return a classification result
        with patch.object(CohereLLM, "generate") as mock_generate:
            # Set up the mock to return a predefined classification result
            mock_generate.return_value = (
                ["technology"],
                {
                    "read_tokens": 15,
                    "write_tokens": 5,
                    "total_tokens": 20,
                    "total_cost": 0.0001,
                },
                None,
            )

            # Call the classify method
            result, usage = cohere_llm.classify(
                messages=messages, categories=categories
            )

            # Verify the result
            assert "technology" in result
            assert isinstance(usage, dict)
            assert usage["total_tokens"] == 20
            assert usage["total_cost"] == 0.0001

            # Verify generate was called with appropriate parameters
            call_args = mock_generate.call_args[1]
            assert "system_prompt" in call_args
            assert "categories" in call_args["system_prompt"]
            assert "sports" in call_args["system_prompt"]
            assert "technology" in call_args["system_prompt"]
            assert "entertainment" in call_args["system_prompt"]


@pytest.mark.skip("Requires real API access")
class TestCohereIntegration:
    """Real API integration tests for Cohere (skipped by default)."""

    def test_real_generate(self):
        """Test actual generation with Cohere API."""
        # This test requires a real API key
        if "COHERE_API_KEY" not in os.environ:
            pytest.skip("No COHERE_API_KEY in environment")

        # Create real instance
        llm = CohereLLM("command")

        # Generate a response
        response, usage, _ = llm.generate(
            event_id="test-integration",
            system_prompt="You are a helpful assistant.",
            messages=[
                {
                    "message_type": "human",
                    "message": "Explain what an LLM is in one sentence.",
                    "image_paths": [],
                    "image_urls": [],
                }
            ],
            max_tokens=50,
        )

        # Check response
        assert isinstance(response, str)
        assert len(response) > 0
        assert "language" in response.lower() or "llm" in response.lower()

        # Check usage
        assert usage["read_tokens"] > 0
        assert usage["write_tokens"] > 0
        assert usage["total_tokens"] > 0
        assert usage["total_cost"] > 0

    def test_real_embeddings(self):
        """Test actual embeddings with Cohere API."""
        # This test requires a real API key
        if "COHERE_API_KEY" not in os.environ:
            pytest.skip("No COHERE_API_KEY in environment")

        # Create real instance
        llm = CohereLLM("command")

        # Generate embeddings
        texts = ["This is a test for embeddings", "And this is another test"]
        embeddings, usage = llm.embed(texts)

        # Check embeddings
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0  # Should have embedding dimensions

        # Check usage
        assert usage["total_tokens"] > 0
        assert usage["total_cost"] > 0
        assert usage["model"] in CohereLLM.EMBEDDING_MODELS
