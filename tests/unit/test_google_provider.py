"""
Unit tests for the Google Gemini provider implementation.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.google import GoogleLLM


@pytest.fixture
def mock_google_client():
    """Mock Google genai client."""
    with patch("google.genai.Client") as mock_client:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = "Mock response from Google"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15

        # Setup mock client methods
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = mock_response
        mock_client.return_value = mock_client_instance

        yield mock_client


@pytest.fixture
def google_llm(mock_google_client):
    """Fixture to create an instance of GoogleLLM with mocked dependencies."""
    with patch(
        "src.lluminary.utils.get_secret",
        return_value={"api_key": "test_api_key"},
    ):
        llm = GoogleLLM("gemini-2.0-flash")

        # Force authentication
        llm.auth()

        # Ensure config exists
        if not hasattr(llm, "config"):
            llm.config = {}

        # Add essential config properties
        llm.config["api_key"] = "test_api_key"
        llm.config["client"] = mock_google_client.return_value

        yield llm


def test_supported_model_lists():
    """Test that the model lists are properly configured."""
    # Make sure appropriate lists are populated
    assert len(GoogleLLM.SUPPORTED_MODELS) > 0

    # Check that core models are included
    assert "gemini-2.0-flash" in GoogleLLM.SUPPORTED_MODELS

    # Add special model list handling (they might not exist in all versions)
    thinking_models = getattr(GoogleLLM, "THINKING_MODELS", [])
    # If it exists, test it has the right type
    if thinking_models:
        assert isinstance(thinking_models, list)

    # Verify model list relationships
    # Google models should all have costs and context windows
    for model_name in GoogleLLM.SUPPORTED_MODELS:
        assert model_name in GoogleLLM.CONTEXT_WINDOW
        assert model_name in GoogleLLM.COST_PER_MODEL

        # Verify cost structure is correct
        model_costs = GoogleLLM.COST_PER_MODEL[model_name]
        assert "read_token" in model_costs
        assert "write_token" in model_costs
        assert "image_cost" in model_costs

        # Verify values are of the expected type
        assert isinstance(model_costs["read_token"], (int, float))
        assert isinstance(model_costs["write_token"], (int, float))
        assert isinstance(model_costs["image_cost"], (int, float))

        # Verify context window is a number
        assert isinstance(GoogleLLM.CONTEXT_WINDOW[model_name], int)


def test_google_instance_creation():
    """Test creating a GoogleLLM instance directly."""
    # Create instance with mock auth
    with patch.object(GoogleLLM, "auth"):
        llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")
        # Just test if the instance exists
        assert llm is not None
        assert llm.model_name == "gemini-2.0-flash"
        assert llm.api_key == "test-key"

        # Test validate_model
        assert llm.validate_model("gemini-2.0-flash") is True
        assert llm.validate_model("invalid-model") is False

        # Check image support
        assert llm.SUPPORTS_IMAGES is True


def test_auth(mock_google_client):
    """Test authentication process."""
    with patch(
        "src.lluminary.utils.get_secret",
        return_value={"api_key": "test_api_key"},
    ):
        llm = GoogleLLM("gemini-2.0-flash")
        llm.auth()

        # Verify client was initialized with the API key
        mock_google_client.assert_called_once_with(api_key="test_api_key")

    # Test authentication with thinking model (different API version)
    with patch(
        "src.lluminary.utils.get_secret",
        return_value={"api_key": "test_api_key"},
    ):
        llm = GoogleLLM("gemini-2.0-flash-thinking-exp-01-21")
        llm.auth()

        # Should be called with API version parameter
        mock_google_client.assert_called_with(
            api_key="test_api_key", http_options={"api_version": "v1alpha"}
        )


@patch("requests.get")
@patch("PIL.Image.open")
def test_process_image(mock_image_open, mock_requests_get, google_llm):
    """Test image processing for both file paths and URLs."""

    # Setup mock response for URL
    mock_response = MagicMock()
    mock_response.content = b"fake_image_data"
    mock_requests_get.return_value = mock_response

    # Setup mock PIL image
    mock_pil_image = MagicMock()
    mock_image_open.return_value = mock_pil_image

    # Test processing a local image file
    image = google_llm._process_image("/path/to/local/image.jpg")
    mock_image_open.assert_called_with("/path/to/local/image.jpg")
    assert image == mock_pil_image

    # Test processing an image URL
    image = google_llm._process_image("https://example.com/image.jpg", is_url=True)
    mock_requests_get.assert_called_with("https://example.com/image.jpg", timeout=10)
    assert mock_image_open.call_count == 2
    assert image == mock_pil_image

    # Test error handling for invalid URL
    mock_requests_get.side_effect = Exception("Failed to fetch image")
    with pytest.raises(LLMMistake) as exc:
        google_llm._process_image("https://example.com/bad_image.jpg", is_url=True)

    # Verify LLMMistake exception details
    exception = exc.value
    assert isinstance(exception, LLMMistake)
    assert "Failed to process image URL" in str(exception)
    assert exception.error_type == "image_url_error"
    assert exception.provider == "GoogleLLM"
    assert "source" in exception.details
    assert "original_error" in exception.details

    # Test error handling for invalid file path
    mock_requests_get.side_effect = None
    mock_image_open.side_effect = FileNotFoundError("No such file")
    with pytest.raises(LLMMistake) as exc:
        google_llm._process_image("/path/to/nonexistent/image.jpg")

    # Verify LLMMistake exception details for file error
    exception = exc.value
    assert isinstance(exception, LLMMistake)
    assert "Failed to process image file" in str(exception)
    assert exception.error_type == "image_processing_error"
    assert exception.provider == "GoogleLLM"
    assert "source" in exception.details
    assert "original_error" in exception.details


def test_format_messages_for_model(google_llm):
    """Test message formatting for Google's API."""
    # Create message types to test
    messages = [
        # Human message with text only
        {"message_type": "human", "message": "Hello, assistant!"},
        # AI response message
        {"message_type": "ai", "message": "How can I help you?"},
        # Tool message
        {
            "message_type": "tool",
            "message": "Tool response",
            "tool_use": {"name": "weather", "input": {"location": "New York"}},
            "tool_result": {"tool_id": "weather", "success": True, "result": "Sunny"},
        },
    ]

    # Mock the image processing method to avoid actual processing
    with patch.object(google_llm, "_process_image", return_value=MagicMock()):
        formatted = google_llm._format_messages_for_model(messages)

        # Verify human message
        assert formatted[0].role == "user"
        assert formatted[0].parts[0].text == "Hello, assistant!"

        # Verify AI message
        assert formatted[1].role == "model"
        assert formatted[1].parts[0].text == "How can I help you?"

        # Verify tool message
        assert formatted[2].role == "tool"
        assert formatted[2].parts[0].text == "Tool response"

    # Test message with images
    message_with_images = [
        {
            "message_type": "human",
            "message": "What's in this image?",
            "image_paths": ["/path/to/image.jpg"],
            "image_urls": ["https://example.com/image.jpg"],
        }
    ]

    # Mock image processing to return a mock image
    with patch.object(google_llm, "_process_image", return_value=MagicMock()):
        formatted = google_llm._format_messages_for_model(message_with_images)

        # Verify message format
        assert formatted[0].role == "user"
        assert formatted[0].parts[0].text == "What's in this image?"
        # Should have two more parts for the two images
        assert len(formatted[0].parts) == 3


def test_raw_generate(google_llm, mock_google_client):
    """Test raw generation functionality."""
    # Create test messages
    messages = [{"message_type": "human", "message": "Tell me about quantum computing"}]

    # Test with default parameters
    response, usage = google_llm._raw_generate(
        event_id="test_event",
        system_prompt="You are a helpful assistant",
        messages=messages,
        max_tokens=1000,
        temp=0.7,
    )

    # Verify response and usage stats
    assert response == "Mock response from Google"
    assert usage["read_tokens"] == 10
    assert usage["write_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert "total_cost" in usage

    # Check client was called with correct parameters
    client_instance = mock_google_client.return_value
    client_instance.models.generate_content.assert_called_once()

    # Verify system prompt was used
    call_args = client_instance.models.generate_content.call_args[1]
    assert call_args["model"] == "gemini-2.0-flash"
    assert call_args["config"].system_instruction == "You are a helpful assistant"
    assert call_args["config"].max_output_tokens == 1000
    assert call_args["config"].temperature == 0.7


def test_raw_generate_with_tools(google_llm, mock_google_client):
    """Test generation with tools enabled."""
    # Create test messages
    messages = [{"message_type": "human", "message": "What's the weather in New York?"}]

    # Define mock tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get the weather in a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]

    # Test with tools
    response, usage = google_llm._raw_generate(
        event_id="test_event",
        system_prompt="You are a helpful assistant",
        messages=messages,
        tools=tools,
    )

    # Verify tool configuration was included
    client_instance = mock_google_client.return_value
    call_args = client_instance.models.generate_content.call_args[1]
    assert call_args["config"].tools == tools
    assert call_args["config"].automatic_function_calling == {"disable": True}


def test_image_handling(google_llm, mock_google_client):
    """Test the image handling in generation requests."""
    # Create message with images
    messages = [
        {
            "message_type": "human",
            "message": "What's in this image?",
            "image_paths": ["/path/to/image.jpg"],
            "image_urls": ["https://example.com/image.jpg"],
        }
    ]

    # Mock image processing
    with patch.object(google_llm, "_process_image", return_value=MagicMock()):
        response, usage = google_llm._raw_generate(
            event_id="test_event", system_prompt="", messages=messages
        )

        # Verify image count in usage stats
        assert usage["images"] == 2
        assert "image_cost" in usage
        assert usage["total_cost"] > 0


def test_error_handling(google_llm, mock_google_client):
    """Test error handling during generation."""
    # Import relevant exception classes
    from lluminary.exceptions import RateLimitError, ServiceUnavailableError

    # Define error cases to test
    error_cases = [
        # Rate limit error
        {
            "exception": Exception("Rate limit exceeded"),
            "expected_exception": RateLimitError,
            "expected_contains": "rate limit exceeded",
        },
        # Authentication error
        {
            "exception": Exception("Invalid API key provided"),
            "expected_exception": LLMMistake,
            "expected_contains": "Google API generation failed",
        },
        # Service unavailable error
        {
            "exception": Exception("Service temporarily unavailable"),
            "expected_exception": ServiceUnavailableError,
            "expected_contains": "service unavailable",
        },
        # Server error
        {
            "exception": Exception("Server error occurred"),
            "expected_exception": LLMMistake,
            "expected_contains": "Google API generation failed",
        },
    ]

    # Test each error case
    client_instance = mock_google_client.return_value
    for error_case in error_cases:
        # Set up the error
        client_instance.models.generate_content.side_effect = error_case["exception"]

        # Create test message
        messages = [{"message_type": "human", "message": "Generate an error"}]

        # Call generate and expect an exception
        with pytest.raises(error_case["expected_exception"]) as exc_info:
            google_llm._raw_generate(
                event_id="test_event", system_prompt="", messages=messages
            )

        # Verify exception type and message
        exception = exc_info.value
        assert isinstance(exception, error_case["expected_exception"])
        assert error_case["expected_contains"] in str(exception).lower()
        assert "Google" in str(exception)

        # For LLMMistake, check error_type and provider fields
        if isinstance(exception, LLMMistake):
            assert exception.provider == "GoogleLLM"
            assert exception.error_type == "api_error"
            assert "original_error" in exception.details

        # For RateLimitError, check retry_after
        if isinstance(exception, RateLimitError):
            assert exception.provider == "GoogleLLM"
            assert exception.retry_after is not None
            assert exception.retry_after > 0

        # Reset for next test
        client_instance.models.generate_content.side_effect = None
        client_instance.models.generate_content.reset_mock()

    # Test invalid model error
    with pytest.raises(ValueError) as exc_info:
        # Create a new instance with an invalid model
        with patch.object(GoogleLLM, "auth"):
            GoogleLLM("non-existent-model", api_key="test-key")

    assert "not supported" in str(exc_info.value).lower()
    assert "non-existent-model" in str(exc_info.value)


def test_supports_image_input(google_llm):
    """Test the supports_image_input method."""
    # Just verify it returns the class attribute
    assert google_llm.supports_image_input() == google_llm.SUPPORTS_IMAGES


def test_get_supported_models(google_llm):
    """Test the get_supported_models method."""
    models = google_llm.get_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "gemini-2.0-flash" in models


@patch("google.generativeai.GenerativeModel")
def test_stream_generate(mock_generative_model, google_llm):
    """Test the streaming functionality."""
    # Create mock streaming response
    mock_model = MagicMock()
    mock_stream_response = [
        MagicMock(text="Hello"),
        MagicMock(text=" world"),
        MagicMock(text="!"),
        MagicMock(text="", candidates=[MagicMock(content=MagicMock(parts=[]))]),
    ]
    mock_model.generate_content.return_value = mock_stream_response
    mock_generative_model.return_value = mock_model

    # Create test message
    messages = [{"message_type": "human", "message": "Stream response"}]

    # Create callback to capture chunks
    chunks = []

    def callback(chunk, usage):
        chunks.append((chunk, usage))

    # Test streaming
    result_chunks = []
    for chunk, usage in google_llm.stream_generate(
        event_id="test_event",
        system_prompt="You are a helpful assistant",
        messages=messages,
        callback=callback,
    ):
        result_chunks.append((chunk, usage))

    # Verify streaming results
    assert len(result_chunks) == 4
    assert result_chunks[0][0] == "Hello"
    assert result_chunks[1][0] == " world"
    assert result_chunks[2][0] == "!"
    assert result_chunks[3][0] == ""  # Final empty chunk

    # Verify callback was called
    assert len(chunks) == 4

    # Verify final usage contains cost information
    final_usage = result_chunks[-1][1]
    assert "total_cost" in final_usage
    assert final_usage["is_complete"] is True


@patch("google.generativeai.GenerativeModel")
def test_stream_with_function_calls(mock_generative_model, google_llm):
    """Test streaming with function calls."""
    # Create mock model
    mock_model = MagicMock()

    # Create mock response with function call
    mock_candidate = MagicMock()
    mock_part = MagicMock()
    mock_function_call = MagicMock()
    mock_function_call.name = "get_weather"
    mock_function_call.args = {"location": "New York"}
    mock_part.function_call = mock_function_call

    mock_candidate.content.parts = [mock_part]
    mock_final_chunk = MagicMock(candidates=[mock_candidate])

    # Create stream sequence with function call at the end
    mock_stream = [
        MagicMock(text="I'll check"),
        MagicMock(text=" the weather"),
        mock_final_chunk,
    ]
    mock_model.generate_content.return_value = mock_stream
    mock_generative_model.return_value = mock_model

    # Define test functions
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: Sunny"

    # Test streaming with function call
    result_chunks = []
    for chunk, usage in google_llm.stream_generate(
        event_id="test_event",
        system_prompt="",
        messages=[
            {"message_type": "human", "message": "What's the weather in New York?"}
        ],
        functions=[get_weather],
    ):
        result_chunks.append((chunk, usage))

    # Verify function was detected in the final chunk
    final_usage = result_chunks[-1][1]
    assert "tool_use" in final_usage
    assert len(final_usage["tool_use"]) > 0


def test_model_costs(google_llm):
    """Test getting cost information for models."""
    # Get costs for current model
    costs = google_llm.get_model_costs()
    assert "read_token" in costs
    assert "write_token" in costs
    assert "image_cost" in costs

    # Verify each model has cost info
    for model in google_llm.SUPPORTED_MODELS:
        # Temporarily change the model name
        original_model = google_llm.model_name
        google_llm.model_name = model

        costs = google_llm.get_model_costs()
        assert "read_token" in costs
        assert "write_token" in costs
        assert "image_cost" in costs

        # Reset model name
        google_llm.model_name = original_model


def test_with_missing_client(google_llm):
    """Test behavior when client isn't initialized."""
    # Force client to None
    google_llm.client = None

    # Create test message
    messages = [{"message_type": "human", "message": "Hello"}]

    # Mock auth to verify it's called when client is None
    with patch.object(google_llm, "auth") as mock_auth:
        google_llm._raw_generate(
            event_id="test_event", system_prompt="", messages=messages
        )

        # Verify auth was called
        mock_auth.assert_called_once()


def test_classification(google_llm):
    """Test Google classification support."""
    # Define test categories and messages
    categories = {
        "question": "Content asking for information",
        "command": "Content requesting action",
        "statement": "Content stating facts or opinions",
    }

    messages = [{"message_type": "human", "message": "What time is it?"}]

    # Mock the generate method
    with patch.object(google_llm, "generate") as mock_generate:
        # Set up mock to return classification result
        mock_generate.return_value = (
            ["question"],
            {
                "read_tokens": 15,
                "write_tokens": 5,
                "total_tokens": 20,
                "total_cost": 0.0002,
            },
            None,
        )

        # Perform classification
        result, usage = google_llm.classify(messages=messages, categories=categories)

        # Verify result
        assert "question" in result
        assert usage["total_tokens"] == 20
        assert usage["total_cost"] == 0.0002

        # Verify generate was called with classification prompt
        call_args = mock_generate.call_args[1]
        assert "system_prompt" in call_args
        assert "categories" in call_args["system_prompt"]

        # Test with examples
        examples = [
            {
                "user_input": "What is the capital of France?",
                "doc_str": "This asks for information about France's capital.",
                "selection": "question",
            },
            {
                "user_input": "Please close the door.",
                "doc_str": "This is telling someone to do something with the door.",
                "selection": "command",
            },
        ]

        # Reset mock
        mock_generate.reset_mock()

        # Call classify with examples
        google_llm.classify(messages=messages, categories=categories, examples=examples)

        # Verify examples were incorporated into prompt
        call_args = mock_generate.call_args[1]
        system_prompt = call_args.get("system_prompt", "")
        assert "capital of France" in system_prompt
        assert "close the door" in system_prompt

        # Test with multiple categories selection
        mock_generate.reset_mock()
        mock_generate.return_value = (
            ["question", "statement"],
            {
                "read_tokens": 15,
                "write_tokens": 5,
                "total_tokens": 20,
                "total_cost": 0.0002,
            },
            None,
        )

        # Call with allow_multiple=True
        result, usage = google_llm.classify(
            messages=messages, categories=categories, allow_multiple=True
        )

        # Verify multiple categories are returned
        assert "question" in result
        assert "statement" in result
        assert len(result) == 2
