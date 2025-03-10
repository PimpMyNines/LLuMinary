"""
Basic test suite for Google Gemini provider.
"""

from unittest.mock import MagicMock, patch

import pytest

# Skip all tests in this file
pytestmark = pytest.mark.skip("Skip Google provider tests")

# First, patch the google.genai import
patch_genai = patch("google.genai", new=MagicMock())
patch_genai.start()

# Patch PIL.Image too
patch_pil = patch("PIL.Image", new=MagicMock())
patch_pil.start()

# Patch requests
patch_requests = patch("requests", new=MagicMock())
patch_requests.start()

# Now import GoogleLLM
from lluminary.models.providers.google import GoogleLLM

# Stop the patches after import
patch_genai.stop()
patch_pil.stop()
patch_requests.stop()


def test_basic_initialization():
    """Test basic initialization of GoogleLLM."""
    # Use a context manager to patch LLM.auth method to avoid actual auth
    with patch.object(GoogleLLM, "auth", return_value=None):
        # Create a GoogleLLM instance
        llm = GoogleLLM("gemini-2.0-flash")

        # Check model name
        assert llm.model_name == "gemini-2.0-flash"

        # Check methods exist
        assert hasattr(llm, "supports_image_input")
        assert callable(llm.supports_image_input)
        assert hasattr(llm, "get_supported_models")
        assert callable(llm.get_supported_models)


def test_get_supported_models():
    """Test the get_supported_models method."""
    # Use a context manager to patch LLM.auth method to avoid actual auth
    with patch.object(GoogleLLM, "auth", return_value=None):
        # Create a GoogleLLM instance
        llm = GoogleLLM("gemini-2.0-flash")

        # Get supported models
        models = llm.get_supported_models()

        # Check that it returns a list of models
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gemini-2.0-flash" in models


def test_supports_image_input():
    """Test the supports_image_input method."""
    # Use a context manager to patch LLM.auth method to avoid actual auth
    with patch.object(GoogleLLM, "auth", return_value=None):
        # Create a GoogleLLM instance
        llm = GoogleLLM("gemini-2.0-flash")

        # Check image support
        assert llm.supports_image_input() is True


def test_auth():
    """Test the auth method."""
    # Patch get_secret to return a mock API key
    with patch(
        "src.lluminary.models.utils.get_secret", return_value={"api_key": "test-key"}
    ):
        # Patch google.genai.Client
        with patch("google.genai.Client", return_value=MagicMock()) as mock_client:
            # Create a GoogleLLM instance
            llm = GoogleLLM("gemini-2.0-flash")

            # Call auth
            llm.auth()

            # Verify client was initialized with the API key
            mock_client.assert_called_once_with(api_key="test-key")
            assert llm.client is not None


def test_process_image():
    """Test the _process_image method."""
    with patch.object(GoogleLLM, "auth", return_value=None):
        # Create a GoogleLLM instance
        llm = GoogleLLM("gemini-2.0-flash")

        # Mock PIL.Image.open to return a mock image
        mock_image = MagicMock()
        with patch("PIL.Image.open", return_value=mock_image) as mock_open:
            # Test processing a local image
            result = llm._process_image("/path/to/image.jpg")

            # Verify Image.open was called with the path
            mock_open.assert_called_once_with("/path/to/image.jpg")
            assert result == mock_image

        # Test processing a URL
        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"

        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("PIL.Image.open", return_value=mock_image) as mock_open:
                result = llm._process_image(
                    "https://example.com/image.jpg", is_url=True
                )

                # Verify requests.get was called with the URL
                mock_get.assert_called_once_with(
                    "https://example.com/image.jpg", timeout=10
                )
                assert result == mock_image


def test_format_messages_for_model():
    """Test the _format_messages_for_model method."""
    with patch.object(GoogleLLM, "auth", return_value=None):
        # Create a GoogleLLM instance
        llm = GoogleLLM("gemini-2.0-flash")

        # Create test messages
        test_messages = [{"message_type": "human", "message": "Hello"}]

        # Mock the google.genai.types.Content and Part classes
        content_mock = MagicMock()
        part_mock = MagicMock()

        with patch("google.genai.types.Content", return_value=content_mock):
            with patch("google.genai.types.Part") as part_type_mock:
                part_type_mock.from_text.return_value = part_mock

                # Call the method
                formatted = llm._format_messages_for_model(test_messages)

                # Check that Content was created
                assert formatted[0] == content_mock

                # Check that the role was set correctly
                assert content_mock.role == "user"

                # Check that a Part was created from the text
                part_type_mock.from_text.assert_called_once_with(text="Hello")

                # Check that the part was added to the content
                assert content_mock.parts == [part_mock]


def test_raw_generate():
    """Test the _raw_generate method."""
    with patch.object(GoogleLLM, "auth", return_value=None):
        # Create a GoogleLLM instance
        llm = GoogleLLM("gemini-2.0-flash")

        # Set up mock client
        mock_client = MagicMock()
        llm.client = mock_client

        # Create mock response
        mock_response = MagicMock()
        mock_response.text = "Generated text"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15

        # Set up client to return the mock response
        mock_client.models.generate_content.return_value = mock_response

        # Mock _format_messages_for_model to return a simple list
        with patch.object(
            llm, "_format_messages_for_model", return_value=["formatted_message"]
        ):
            # Call _raw_generate
            response, usage = llm._raw_generate(
                event_id="test_event",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "Hello"}],
                max_tokens=1000,
                temp=0.7,
            )

            # Check response
            assert response == "Generated text"

            # Check usage
            assert usage["read_tokens"] == 10
            assert usage["write_tokens"] == 5
            assert usage["total_tokens"] == 15
            assert "total_cost" in usage

            # Verify client was called correctly
            mock_client.models.generate_content.assert_called_once()
            call_args = mock_client.models.generate_content.call_args
            assert call_args[1]["model"] == "gemini-2.0-flash"
            assert call_args[1]["contents"] == ["formatted_message"]

            # Check config parameters
            config = call_args[1]["config"]
            assert config.max_output_tokens == 1000
            assert config.temperature == 0.7
            assert config.system_instruction == "You are a helpful assistant"
