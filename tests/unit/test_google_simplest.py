"""
Simplified tests for Google Provider implementation.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.google import GoogleLLM


def test_import_google_llm():
    """Test that GoogleLLM can be imported."""
    assert GoogleLLM is not None


def test_google_class_attributes():
    """Test Google provider class attributes directly."""
    # Check model lists directly from the class
    assert "gemini-2.0-flash" in GoogleLLM.SUPPORTED_MODELS
    assert GoogleLLM.SUPPORTS_IMAGES is True
    assert "gemini-2.0-flash" in GoogleLLM.CONTEXT_WINDOW

    # Check cost models
    assert "gemini-2.0-flash" in GoogleLLM.COST_PER_MODEL
    assert "read_token" in GoogleLLM.COST_PER_MODEL["gemini-2.0-flash"]
    assert "write_token" in GoogleLLM.COST_PER_MODEL["gemini-2.0-flash"]
    assert "image_cost" in GoogleLLM.COST_PER_MODEL["gemini-2.0-flash"]

    # Check thinking model identification
    assert "gemini-2.0-flash-thinking-exp-01-21" in GoogleLLM.SUPPORTED_MODELS

    # Check context window sizes
    assert GoogleLLM.CONTEXT_WINDOW["gemini-2.0-flash"] > 0


def test_google_instance_creation_with_try_catch():
    """Test creating a GoogleLLM instance with try-except to catch any errors."""
    try:
        with patch.object(GoogleLLM, "auth"):
            llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")
            print(f"LLM created: {llm}, model_name={llm.model_name}")
            assert llm.model_name == "gemini-2.0-flash"
            assert "api_key" in llm.config
            assert llm.config["api_key"] == "test-key"
    except Exception as e:
        pytest.fail(f"GoogleLLM instance creation failed with error: {e}")


@patch.object(GoogleLLM, "auth")
def test_google_instance_creation(mock_auth):
    """Test creating a GoogleLLM instance directly."""
    # Create instance
    llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")
    assert llm.model_name == "gemini-2.0-flash"
    assert "api_key" in llm.config
    assert llm.config["api_key"] == "test-key"


@patch.object(GoogleLLM, "auth")
def test_message_formatting(mock_auth):
    """Test message formatting for Google provider."""
    llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

    # Create a simple message to test formatting
    message = {"message_type": "human", "message": "Hello, assistant!"}

    # Mock the Google types.Content and Part classes
    with patch("google.genai.types.Content") as mock_content_class, patch(
        "google.genai.types.Part"
    ) as mock_part:

        # Setup mock content instance
        mock_content = MagicMock()
        mock_content_class.return_value = mock_content

        # Setup mock part for text
        mock_text_part = MagicMock()
        mock_part.from_text.return_value = mock_text_part

        # Format the message
        formatted = llm._format_messages_for_model([message])

        # Verify Content was created
        assert mock_content_class.called

        # Verify role was set correctly
        assert mock_content.role == "user"

        # Verify Part.from_text was called with correct text
        mock_part.from_text.assert_called_with(text="Hello, assistant!")


@patch.object(GoogleLLM, "auth")
def test_simple_generation(mock_auth):
    """Test generation with mocked response."""
    # Create the LLM instance
    llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

    # Create test message
    messages = [{"message_type": "human", "message": "Hello, assistant!"}]

    # Create mock response
    mock_response = MagicMock()
    mock_response.text = "Hello, human!"
    mock_response.candidates = [MagicMock()]
    mock_response.usage_metadata.prompt_token_count = 5
    mock_response.usage_metadata.candidates_token_count = 3
    mock_response.usage_metadata.total_token_count = 8

    # Setup client and mock response
    llm.client = MagicMock()
    models_mock = MagicMock()
    llm.client.models = models_mock
    models_mock.generate_content.return_value = mock_response

    # Mock the _format_messages_for_model method
    with patch.object(llm, "_format_messages_for_model") as mock_format:
        mock_format.return_value = [MagicMock()]

        # Call generate method
        response, usage, _ = llm.generate(
            event_id="test_event",
            system_prompt="You are a helpful assistant",
            messages=messages,
            max_tokens=100,
        )

        # Verify the model was called
        assert models_mock.generate_content.called

        # Verify response
        assert response == "Hello, human!"
        assert usage["read_tokens"] == 5
        assert usage["write_tokens"] == 3
        assert usage["total_tokens"] == 8
        assert "total_cost" in usage
        assert usage["total_cost"] > 0


def test_thinking_model_support():
    """Test thinking model support in Google LLM."""
    # Define the thinking model we want to test
    thinking_model = "gemini-2.0-flash-thinking-exp-01-21"

    # Create instances with mock auth
    with patch.object(GoogleLLM, "auth"):
        # Test with special model name that requires different API version
        thinking_llm = GoogleLLM(thinking_model, api_key="test-key")
        assert thinking_llm.model_name == thinking_model

        # Create the regular model
        regular_llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")
        assert regular_llm.model_name == "gemini-2.0-flash"


@patch.object(GoogleLLM, "auth")
def test_image_processing_url(mock_auth):
    """Test image processing methods for URLs."""
    llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

    # Mock requests.get for URL processing
    with patch("requests.get") as mock_get, patch("PIL.Image.open") as mock_image_open:
        # Setup mock response for URL processing
        mock_response = MagicMock()
        mock_response.content = b"mock_image_data"
        mock_get.return_value = mock_response

        # Setup mock PIL image
        mock_pil_image = MagicMock()
        mock_image_open.return_value = mock_pil_image

        # Test URL processing
        image = llm._process_image("https://example.com/image.jpg", is_url=True)

        # Verify URL fetch was called
        mock_get.assert_called_with("https://example.com/image.jpg", timeout=10)
        assert mock_image_open.called
        assert image == mock_pil_image


@patch.object(GoogleLLM, "auth")
def test_image_processing_file(mock_auth):
    """Test image processing methods for local files."""
    llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

    # Mock PIL.Image.open for local file processing
    with patch("PIL.Image.open") as mock_image_open:
        # Setup mock PIL image
        mock_pil_image = MagicMock()
        mock_image_open.return_value = mock_pil_image

        # Test local file processing
        image = llm._process_image("/path/to/image.jpg")

        # Verify Image.open was called with correct path
        mock_image_open.assert_called_with("/path/to/image.jpg")
        assert image == mock_pil_image


@patch.object(GoogleLLM, "auth")
def test_functions_support(mock_auth):
    """Test functions/tools calling support."""
    llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

    # Define simple test function
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: Sunny"

    # Setup client and mock response
    mock_response = MagicMock()
    mock_response.text = "I'll check the weather"
    mock_response.candidates = [MagicMock()]
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    llm.client = MagicMock()
    models_mock = MagicMock()
    llm.client.models = models_mock
    models_mock.generate_content.return_value = mock_response

    # Test with raw_generate patched (since this is what's called by generate method)
    with patch.object(llm, "_raw_generate") as mock_raw_generate:
        mock_raw_generate.return_value = (
            mock_response.text,
            {
                "read_tokens": 10,
                "write_tokens": 5,
                "total_tokens": 15,
                "read_cost": 0.02,
                "write_cost": 0.01,
                "total_cost": 0.03,
            },
        )

        # Call generate with functions (proper parameter name for base LLM class)
        response, usage, _ = llm.generate(
            event_id="test_event",
            system_prompt="You are a helpful assistant",
            messages=[
                {"message_type": "human", "message": "What's the weather in New York?"}
            ],
            functions=[get_weather],
        )

        # Verify raw_generate was called
        assert mock_raw_generate.called

        # Verify response
        assert response == "I'll check the weather"
        assert "read_tokens" in usage
        assert "write_tokens" in usage
        assert "total_cost" in usage


@patch.object(GoogleLLM, "auth")
def test_error_handling(mock_auth):
    """Test error handling in the provider."""
    # Create the LLM instance
    llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

    # Setup client to raise an exception
    llm.client = MagicMock()
    models_mock = MagicMock()
    llm.client.models = models_mock
    models_mock.generate_content.side_effect = Exception("API Error")

    # Mock message formatting to avoid errors
    with patch.object(llm, "_format_messages_for_model"):
        # Test API error handling
        with pytest.raises(Exception) as exc_info:
            llm.generate(
                event_id="test_event",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "Hello"}],
            )

        # Verify error message contains the original error
        assert "API Error" in str(exc_info.value)
