"""
Simplified tests for the Google Gemini provider.
"""

from unittest.mock import MagicMock, patch

import pytest
from lluminary.models.providers.google import GoogleLLM


def test_google_model_lists():
    """Test that model lists contain expected models."""
    # Check model lists
    assert "gemini-2.0-flash" in GoogleLLM.SUPPORTED_MODELS
    assert "gemini-2.0-pro-exp-02-05" in GoogleLLM.SUPPORTED_MODELS
    assert "gemini-2.0-flash-thinking-exp-01-21" in GoogleLLM.SUPPORTED_MODELS

    # Check corresponding context windows and costs
    for model in GoogleLLM.SUPPORTED_MODELS:
        assert model in GoogleLLM.CONTEXT_WINDOW
        assert model in GoogleLLM.COST_PER_MODEL


def test_google_constructor():
    """Test the constructor of GoogleLLM."""
    # Skip auth during initialization
    with patch.object(GoogleLLM, "auth"):
        # Create a GoogleLLM instance
        llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

        # Verify model name and config
        assert llm.model_name == "gemini-2.0-flash"
        assert llm.config["api_key"] == "test-key"
        assert llm.client is None  # Client not initialized yet


@patch("lluminary.models.providers.google.get_secret")
@patch("google.genai.Client")
def test_google_auth(mock_client, mock_get_secret):
    """Test authentication with Google."""
    # Configure mocks
    mock_get_secret.return_value = {"api_key": "test-api-key"}
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance

    # Create and authenticate LLM
    llm = GoogleLLM("gemini-2.0-flash")
    llm.auth()

    # Verify get_secret was called correctly
    mock_get_secret.assert_called_once_with("google_api_key", required_keys=["api_key"])

    # Verify client was initialized correctly
    mock_client.assert_called_once_with(api_key="test-api-key")
    assert llm.client is not None


@patch("lluminary.models.providers.google.get_secret")
@patch("google.genai.Client")
def test_google_auth_thinking_model(mock_client, mock_get_secret):
    """Test authentication with a thinking model."""
    # Configure mocks
    mock_get_secret.return_value = {"api_key": "test-api-key"}
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance

    # Create and authenticate LLM with thinking model
    llm = GoogleLLM("gemini-2.0-flash-thinking-exp-01-21")
    llm.auth()

    # Verify client was initialized with alpha API version
    mock_client.assert_called_once_with(
        api_key="test-api-key", http_options={"api_version": "v1alpha"}
    )


@patch("lluminary.models.providers.google.get_secret")
def test_google_auth_error(mock_get_secret):
    """Test authentication error handling."""
    # Make get_secret raise an exception
    mock_get_secret.side_effect = Exception("Failed to get secret")

    # Create LLM
    llm = GoogleLLM("gemini-2.0-flash")

    # Authentication should raise an exception
    with pytest.raises(Exception) as exc_info:
        llm.auth()

    assert "Google authentication failed" in str(exc_info.value)


@patch.object(GoogleLLM, "auth")
def test_google_image_support(mock_auth):
    """Test image support detection."""
    llm = GoogleLLM("gemini-2.0-flash")
    assert llm.supports_image_input() is True
    assert llm.supports_image_input() == GoogleLLM.SUPPORTS_IMAGES


@patch.object(GoogleLLM, "auth")
def test_google_supported_models(mock_auth):
    """Test getting supported models."""
    llm = GoogleLLM("gemini-2.0-flash")
    models = llm.get_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "gemini-2.0-flash" in models


@patch.object(GoogleLLM, "auth")
def test_google_get_model_costs(mock_auth):
    """Test getting model costs."""
    llm = GoogleLLM("gemini-2.0-flash")
    costs = llm.get_model_costs()
    assert "read_token" in costs
    assert "write_token" in costs
    assert "image_cost" in costs
    assert costs["read_token"] > 0
    assert costs["write_token"] > 0
    assert costs["image_cost"] > 0


@patch("PIL.Image.open")
@patch.object(GoogleLLM, "auth")
def test_process_local_image(mock_auth, mock_image_open):
    """Test processing a local image."""
    # Configure mock
    mock_image = MagicMock()
    mock_image_open.return_value = mock_image

    # Create LLM and process image
    llm = GoogleLLM("gemini-2.0-flash")
    result = llm._process_image("/path/to/image.jpg")

    # Verify image loading
    mock_image_open.assert_called_once_with("/path/to/image.jpg")
    assert result == mock_image


@patch("requests.get")
@patch("PIL.Image.open")
@patch.object(GoogleLLM, "auth")
def test_process_url_image(mock_auth, mock_image_open, mock_requests_get):
    """Test processing an image from URL."""
    # Configure mocks
    mock_response = MagicMock()
    mock_response.content = b"image_data"
    mock_requests_get.return_value = mock_response

    mock_image = MagicMock()
    mock_image_open.return_value = mock_image

    # Create LLM and process image URL
    llm = GoogleLLM("gemini-2.0-flash")
    result = llm._process_image("https://example.com/image.jpg", is_url=True)

    # Verify HTTP request and image loading
    mock_requests_get.assert_called_once_with(
        "https://example.com/image.jpg", timeout=10
    )
    assert mock_image_open.call_count == 1
    assert result == mock_image


@patch("google.genai.types.Content")
@patch("google.genai.types.Part")
@patch.object(GoogleLLM, "auth")
def test_format_messages(mock_auth, mock_part, mock_content):
    """Test message formatting."""
    # Configure mocks
    mock_content_instance = MagicMock()
    mock_content.return_value = mock_content_instance

    mock_text_part = MagicMock()
    mock_part.from_text.return_value = mock_text_part

    # Create message to format
    message = {
        "message_type": "human",
        "message": "Hello",
        "image_paths": [],
        "image_urls": [],
    }

    # Create LLM and format message
    llm = GoogleLLM("gemini-2.0-flash")
    formatted = llm._format_messages_for_model([message])

    # Verify formatting
    assert len(formatted) == 1
    assert formatted[0] == mock_content_instance
    assert formatted[0].role == "user"
    mock_part.from_text.assert_called_once_with(text="Hello")
    assert mock_content_instance.parts == [mock_text_part]


@patch.object(GoogleLLM, "_format_messages_for_model")
@patch.object(GoogleLLM, "auth")
def test_raw_generate(mock_auth, mock_format_messages):
    """Test raw generation."""
    # Configure mocks
    formatted_messages = [MagicMock()]
    mock_format_messages.return_value = formatted_messages

    mock_response = MagicMock()
    mock_response.text = "Generated text"
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    # Create LLM
    llm = GoogleLLM("gemini-2.0-flash")
    llm.client = MagicMock()
    llm.client.models.generate_content.return_value = mock_response

    # Generate text
    response, usage = llm._raw_generate(
        event_id="test123",
        system_prompt="You are a helpful assistant",
        messages=[{"message_type": "human", "message": "Hello"}],
        max_tokens=100,
        temp=0.5,
    )

    # Verify response and usage
    assert response == "Generated text"
    assert usage["read_tokens"] == 10
    assert usage["write_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert usage["event_id"] == "test123"
    assert "total_cost" in usage


@patch("google.generativeai.GenerativeModel")
@patch.object(GoogleLLM, "_format_messages_for_model")
@patch.object(GoogleLLM, "auth")
def test_stream_generate(mock_auth, mock_format_messages, mock_generative_model):
    """Test streaming generation."""
    # Configure mocks
    formatted_messages = [MagicMock()]
    mock_format_messages.return_value = formatted_messages

    mock_model = MagicMock()
    mock_generative_model.return_value = mock_model

    mock_chunk1 = MagicMock(text="Hello")
    mock_chunk2 = MagicMock(text=" world")
    mock_chunk3 = MagicMock(candidates=[MagicMock(content=MagicMock(parts=[]))])

    mock_model.generate_content.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]

    # Create LLM
    llm = GoogleLLM("gemini-2.0-flash")

    # Stream generate
    chunks = []
    for chunk, usage in llm.stream_generate(
        event_id="test123",
        system_prompt="You are a helpful assistant",
        messages=[{"message_type": "human", "message": "Hello"}],
    ):
        chunks.append((chunk, usage))

    # Verify chunks
    assert len(chunks) == 3
    assert chunks[0][0] == "Hello"
    assert chunks[1][0] == " world"
    assert chunks[2][0] == ""  # Final empty chunk

    # Verify last chunk has complete flag
    assert chunks[-1][1]["is_complete"] is True
    assert "total_cost" in chunks[-1][1]
