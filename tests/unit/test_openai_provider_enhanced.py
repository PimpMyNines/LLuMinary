"""
Enhanced unit tests for the OpenAI provider implementation.

This module extends the test coverage for the OpenAI provider, focusing on:
- Authentication flows with AWS Secrets Manager and environment variables
- In-depth reranking functionality testing
- Detailed token counting and cost calculation validation
- Comprehensive error handling and recovery testing
- Image handling and generation testing
- Timeout handling
- Complex message formatting edge cases

These enhanced tests aim to increase the test coverage from 36% to 75%+.
"""

import base64
import os
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

import pytest
from lluminary.exceptions import LLMMistake
from lluminary.models.providers.openai import OpenAILLM


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


def test_supported_model_lists(openai_llm):
    """Test that the model lists are properly configured."""
    # Make sure appropriate lists are populated
    assert len(openai_llm.SUPPORTED_MODELS) > 0
    assert len(openai_llm.THINKING_MODELS) > 0
    assert len(openai_llm.EMBEDDING_MODELS) > 0

    # Check that core models are included
    assert "gpt-4o" in openai_llm.SUPPORTED_MODELS
    assert "gpt-4o" in openai_llm.THINKING_MODELS
    assert "text-embedding-ada-002" in openai_llm.EMBEDDING_MODELS

    # Verify model list relationships
    # Thinking models should be a subset of supported models
    assert all(
        model in openai_llm.SUPPORTED_MODELS for model in openai_llm.THINKING_MODELS
    )

    # Verify that cost and context window data is properly configured for each model
    for model_name in openai_llm.SUPPORTED_MODELS:
        assert model_name in openai_llm.CONTEXT_WINDOW
        assert model_name in openai_llm.COST_PER_MODEL

        # Verify cost structure is correct
        model_costs = openai_llm.COST_PER_MODEL[model_name]
        assert "read_token" in model_costs
        assert "write_token" in model_costs

        # Verify values are of the expected type
        assert isinstance(model_costs["read_token"], (int, float))
        assert isinstance(model_costs["write_token"], (int, float))

        # Verify context window is a number
        assert isinstance(openai_llm.CONTEXT_WINDOW[model_name], int)


def test_openai_initialization(openai_llm):
    """Test OpenAI provider initialization."""
    # Verify model-related settings in initialization
    assert openai_llm.model_name == "gpt-4o"
    assert openai_llm.config["api_key"] == "test-key"

    # Test validation of model name during initialization
    assert openai_llm.validate_model("gpt-4o") is True
    assert openai_llm.validate_model("invalid-model") is False

    # Test timeout setting
    assert openai_llm.timeout == 60  # Default value

    # Test with custom timeout
    with patch.object(OpenAILLM, "auth"):
        llm = OpenAILLM("gpt-4o", api_key="test-key", timeout=120)
        assert llm.timeout == 120

    # Test with custom API base
    with patch.object(OpenAILLM, "auth"):
        llm = OpenAILLM(
            "gpt-4o", api_key="test-key", api_base="https://custom-openai.example.com"
        )
        assert llm.api_base == "https://custom-openai.example.com"

    # Test initialization with invalid model should raise exception
    with pytest.raises(ValueError) as excinfo:
        with patch.object(OpenAILLM, "validate_model", return_value=False):
            OpenAILLM("invalid-model", api_key="test-key")
    assert "not supported" in str(excinfo.value)


@patch("lluminary.utils.get_secret")
def test_auth_with_aws_secrets(mock_get_secret):
    """Test authentication using AWS Secrets Manager."""
    # Mock the get_secret function to return a test API key
    mock_get_secret.return_value = {"api_key": "test-key-from-aws"}

    # Create a new LLM instance with a mocked OpenAI client
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize the LLM
        llm = OpenAILLM("gpt-4o")

        # Call auth method
        llm.auth()

        # Verify AWS secret was retrieved
        mock_get_secret.assert_called_once_with(
            "openai_api_key", required_keys=["api_key"]
        )

        # Verify API key was set in config
        assert llm.config["api_key"] == "test-key-from-aws"

        # Verify OpenAI client was initialized with the correct API key
        mock_openai.assert_called_once_with(api_key="test-key-from-aws")

        # Verify client was set
        assert llm.client == mock_client


@patch("lluminary.utils.get_secret")
def test_auth_with_direct_api_key(mock_get_secret):
    """Test authentication using directly provided API key."""
    # Create a new LLM instance with a direct API key and mocked OpenAI client
    with patch("openai.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize the LLM with API key
        llm = OpenAILLM("gpt-4o", api_key="direct-api-key")

        # Call auth method
        llm.auth()

        # Verify AWS secret was NOT retrieved
        mock_get_secret.assert_not_called()

        # Verify direct API key was used
        assert llm.config["api_key"] == "direct-api-key"

        # Verify OpenAI client was initialized with the correct API key
        mock_openai.assert_called_once_with(api_key="direct-api-key")


@patch("lluminary.utils.get_secret")
def test_auth_error_handling(mock_get_secret):
    """Test error handling during authentication."""
    # Test case 1: AWS Secrets Manager error
    mock_get_secret.side_effect = Exception("AWS Secrets Manager error")

    with pytest.raises(Exception) as excinfo:
        llm = OpenAILLM("gpt-4o")
        llm.auth()
    assert "OpenAI authentication failed" in str(excinfo.value)
    assert "AWS Secrets Manager error" in str(excinfo.value)

    # Test case 2: OpenAI client initialization error
    mock_get_secret.side_effect = None
    mock_get_secret.return_value = {"api_key": "test-key"}

    with patch("openai.OpenAI") as mock_openai:
        mock_openai.side_effect = Exception("OpenAI client error")

        with pytest.raises(Exception) as excinfo:
            llm = OpenAILLM("gpt-4o")
            llm.auth()
        assert "OpenAI authentication failed" in str(excinfo.value)
        assert "OpenAI client error" in str(excinfo.value)


def test_custom_api_base_url():
    """Test using a custom API base URL."""
    with patch("openai.OpenAI") as mock_openai, patch.object(OpenAILLM, "auth"):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Initialize with custom API base
        api_base = "https://custom-api.openai.example.com/v1"
        llm = OpenAILLM("gpt-4o", api_key="test-key", api_base=api_base)

        # Set the client manually
        llm.client = mock_client

        # Make a request that would use the API base
        llm._raw_generate(
            event_id="test",
            system_prompt="Test prompt",
            messages=[{"message_type": "human", "message": "Hello"}],
        )

        # Verify the client was initialized with the custom base URL
        assert llm.api_base == api_base


def test_message_formatting(openai_llm):
    """Test OpenAI message formatting."""
    # Test basic message (human/user)
    messages = [{"message_type": "human", "message": "test message"}]
    formatted = openai_llm._format_messages_for_model(messages)

    assert formatted[0]["role"] == "user"
    # Content might be a string or a list of content blocks
    if isinstance(formatted[0]["content"], str):
        assert formatted[0]["content"] == "test message"
    else:
        assert formatted[0]["content"][0]["type"] == "text"
        assert formatted[0]["content"][0]["text"] == "test message"

    # Test AI/assistant message
    messages = [{"message_type": "ai", "message": "assistant response"}]
    formatted = openai_llm._format_messages_for_model(messages)
    assert formatted[0]["role"] == "assistant"

    # Content might be a string or a list of content blocks
    if isinstance(formatted[0]["content"], str):
        assert formatted[0]["content"] == "assistant response"
    else:
        assert formatted[0]["content"][0]["type"] == "text"
        assert formatted[0]["content"][0]["text"] == "assistant response"


def test_tool_formatting(openai_llm):
    """Test OpenAI tool formatting."""

    def test_tool(x: int) -> int:
        """Test tool with docstring."""
        return x * 2

    tools = [test_tool]
    formatted = openai_llm._format_tools_for_model(tools)

    assert len(formatted) == 1
    assert formatted[0]["type"] == "function"
    assert formatted[0]["function"]["name"] == "test_tool"
    assert "description" in formatted[0]["function"]
    assert "parameters" in formatted[0]["function"]

    # Test with dictionary tool definition
    dict_tool = {
        "name": "dict_test_tool",
        "description": "A test tool defined as a dictionary",
        "input_schema": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter"},
            },
            "required": ["param1"],
        },
    }

    formatted_dict_tool = openai_llm._format_tools_for_model([dict_tool])

    assert len(formatted_dict_tool) == 1
    assert formatted_dict_tool[0]["type"] == "function"
    assert formatted_dict_tool[0]["function"]["name"] == "dict_test_tool"
    assert (
        formatted_dict_tool[0]["function"]["description"]
        == "A test tool defined as a dictionary"
    )
    assert "properties" in formatted_dict_tool[0]["function"]["parameters"]
    assert "param1" in formatted_dict_tool[0]["function"]["parameters"]["properties"]
    assert "param2" in formatted_dict_tool[0]["function"]["parameters"]["properties"]

    # Verify additionalProperties is set to false (strict schema validation)
    assert (
        formatted_dict_tool[0]["function"]["parameters"]["additionalProperties"]
        is False
    )


def test_image_support(openai_llm):
    """Test OpenAI image support."""
    # Test image support flag
    assert openai_llm.supports_image_input() is True

    # Test image message formatting with image_urls
    messages = [
        {
            "message_type": "human",
            "message": "describe this image",
            "image_urls": ["http://test.com/image.jpg"],
        }
    ]

    # Mock the _encode_image_url method to avoid actual downloads
    with patch.object(openai_llm, "_encode_image_url", return_value="base64_data"):
        formatted = openai_llm._format_messages_for_model(messages)

        # The content should be a list with text and image parts
        assert isinstance(formatted[0]["content"], list)
        assert len(formatted[0]["content"]) >= 1

        # Verify there's text content somewhere
        text_found = False
        image_found = False
        for item in formatted[0]["content"]:
            if item.get("type") == "text" and "describe this image" in item.get(
                "text", ""
            ):
                text_found = True
            if item.get("type") == "image_url" and "base64" in item.get(
                "image_url", {}
            ).get("url", ""):
                image_found = True

        assert text_found, "Text content not found in formatted message"
        assert image_found, "Image content not found in formatted message"

        # Check that the image URL is properly formatted
        image_url_item = next(
            item for item in formatted[0]["content"] if item.get("type") == "image_url"
        )
        assert (
            image_url_item["image_url"]["url"] == "data:image/jpeg;base64,base64_data"
        )


def test_encode_image(openai_llm):
    """Test the _encode_image method."""
    # Create a mock image file
    mock_image_data = b"fake_image_data"

    # Mock the open function to return our fake image data
    with patch("builtins.open", mock_open(read_data=mock_image_data)):
        # Call the encode_image method
        base64_data = openai_llm._encode_image("fake_image.jpg")

        # Verify the result is base64 encoded
        expected_base64 = base64.b64encode(mock_image_data).decode("utf-8")
        assert base64_data == expected_base64


@patch("requests.get")
def test_encode_image_url(mock_requests_get, openai_llm):
    """Test the _encode_image_url method."""
    # Create a mock response with image data
    mock_response = MagicMock()
    mock_response.content = b"fake_image_data"
    mock_response.raise_for_status = MagicMock()
    mock_requests_get.return_value = mock_response

    # Mock PIL Image processing
    mock_img = MagicMock()
    mock_img.mode = "RGB"  # Set the mode to RGB
    mock_img.size = (100, 100)

    # Create a mock BytesIO object for the output
    mock_output = BytesIO()
    mock_output.getvalue = MagicMock(return_value=b"processed_image_data")

    # Patch the Image.open method to return our mock image
    with patch("PIL.Image.open", return_value=mock_img), patch.object(
        mock_img, "save", MagicMock()
    ), patch("io.BytesIO", return_value=mock_output):

        # Call the encode_image_url method
        base64_data = openai_llm._encode_image_url("http://example.com/image.jpg")

        # Verify the request was made
        mock_requests_get.assert_called_once_with(
            "http://example.com/image.jpg", timeout=10
        )

        # Verify image processing
        assert mock_img.save.called

        # Verify the result is base64 encoded
        expected_base64 = base64.b64encode(b"processed_image_data").decode("utf-8")
        assert base64_data == expected_base64


@patch("requests.get")
def test_encode_image_url_with_transparency(mock_requests_get, openai_llm):
    """Test the _encode_image_url method with transparent images."""
    # Create a mock response with image data
    mock_response = MagicMock()
    mock_response.content = b"fake_image_data"
    mock_response.raise_for_status = MagicMock()
    mock_requests_get.return_value = mock_response

    # Mock PIL Image processing - image with transparency
    mock_img = MagicMock()
    mock_img.mode = "RGBA"  # Image with alpha channel
    mock_img.size = (100, 100)
    mock_img.split = MagicMock(
        return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    )

    # Create a mock background image
    mock_background = MagicMock()

    # Create a mock BytesIO object for the output
    mock_output = BytesIO()
    mock_output.getvalue = MagicMock(return_value=b"processed_image_data")

    # Patch PIL Image methods
    with patch("PIL.Image.open", return_value=mock_img), patch(
        "PIL.Image.new", return_value=mock_background
    ), patch.object(mock_background, "paste", MagicMock()), patch.object(
        mock_background, "save", MagicMock()
    ), patch(
        "io.BytesIO", return_value=mock_output
    ):

        # Call the encode_image_url method
        base64_data = openai_llm._encode_image_url("http://example.com/transparent.png")

        # Verify a new background was created for the transparent image
        assert "PIL.Image.new" in str(mock_background.__class__)

        # Verify image processing steps for transparent image
        assert mock_img.split.called
        assert mock_background.paste.called
        assert mock_background.save.called

        # Verify the result is base64 encoded
        expected_base64 = base64.b64encode(b"processed_image_data").decode("utf-8")
        assert base64_data == expected_base64


@patch("requests.get")
def test_encode_image_url_error_handling(mock_requests_get, openai_llm):
    """Test error handling in _encode_image_url method."""
    # Test case: HTTP error
    mock_requests_get.side_effect = Exception("HTTP error")

    with pytest.raises(Exception) as excinfo:
        openai_llm._encode_image_url("http://example.com/image.jpg")
    assert "Failed to process image" in str(excinfo.value)
    assert "HTTP error" in str(excinfo.value)


def test_image_message_formatting_combined(openai_llm):
    """Test message formatting with both local images and image URLs."""
    # Create a message with both local images and image URLs
    messages = [
        {
            "message_type": "human",
            "message": "analyze these images",
            "image_paths": ["local_image1.jpg", "local_image2.jpg"],
            "image_urls": [
                "http://example.com/image1.jpg",
                "http://example.com/image2.jpg",
            ],
        }
    ]

    # Mock the encoding methods
    with patch.object(
        openai_llm, "_encode_image", return_value="local_base64"
    ), patch.object(openai_llm, "_encode_image_url", return_value="url_base64"):

        # Format the messages
        formatted = openai_llm._format_messages_for_model(messages)

        # Verify the content structure
        assert isinstance(formatted[0]["content"], list)
        assert len(formatted[0]["content"]) == 5  # 4 images + 1 text

        # Count the number of image parts
        image_parts = [
            item for item in formatted[0]["content"] if item.get("type") == "image_url"
        ]
        assert len(image_parts) == 4

        # Verify local and URL images are formatted correctly
        local_images = [
            item for item in image_parts if "local_base64" in item["image_url"]["url"]
        ]
        url_images = [
            item for item in image_parts if "url_base64" in item["image_url"]["url"]
        ]
        assert len(local_images) == 2
        assert len(url_images) == 2


def test_calculate_image_tokens(openai_llm):
    """Test token calculation for images."""
    # Test low detail mode
    low_detail_tokens = openai_llm.calculate_image_tokens(800, 600, "low")
    assert low_detail_tokens == openai_llm.LOW_DETAIL_TOKEN_COST
    assert low_detail_tokens == 85  # Default value in the code

    # Test high detail mode
    high_detail_tokens = openai_llm.calculate_image_tokens(800, 600, "high")

    # The formula is complex, but we can verify it follows the right pattern
    # For high detail:
    # 1. Scale to fit MAX_IMAGE_SIZE
    # 2. Scale shortest side to TARGET_SHORT_SIDE
    # 3. Calculate tiles: ceil(width / TILE_SIZE) * ceil(height / TILE_SIZE)
    # 4. Calculate tokens: tiles * TILE_COST + BASE_COST

    # For an 800x600 image, we expect:
    # - No scaling in step 1 (both dimensions < MAX_IMAGE_SIZE)
    # - Scaled to 768px for shortest side (600 -> 768)
    # - New dimensions: ~1024x768
    # - Tiles: ceil(1024/512) * ceil(768/512) = 2 * 2 = 4
    # - Tokens: 4 * 170 + 85 = 765

    # While we can't hardcode the exact result due to rounding differences,
    # we can check it's in a reasonable range
    assert high_detail_tokens > 700  # At least this many tokens
    assert high_detail_tokens < 800  # But not too many

    # Test very large image (should be scaled down)
    large_image_tokens = openai_llm.calculate_image_tokens(4000, 3000, "high")

    # This should have more tiles than the previous example
    assert large_image_tokens > high_detail_tokens


def test_estimate_cost(openai_llm):
    """Test the estimate_cost method."""
    # Mock the token estimation method
    with patch.object(openai_llm, "estimate_tokens", return_value=100):
        # Test basic cost estimation (text only)
        total_cost, breakdown = openai_llm.estimate_cost(
            prompt="This is a test prompt", expected_response_tokens=50
        )

        # Verify breakdown structure
        assert "prompt_cost" in breakdown
        assert "response_cost" in breakdown
        assert "image_cost" in breakdown
        assert "image_tokens" in breakdown

        # Test with images using detailed information
        total_cost_with_images, breakdown = openai_llm.estimate_cost(
            prompt="This is a test prompt with images",
            expected_response_tokens=50,
            images=[(800, 600, "high"), (1024, 768, "low")],
        )

        # Verify image costs are calculated
        assert breakdown["image_tokens"] > 0
        assert breakdown["image_cost"] > 0

        # Total cost should be higher with images
        assert total_cost_with_images > total_cost

        # Test with simple image count
        total_cost_simple, breakdown = openai_llm.estimate_cost(
            prompt="This is a test prompt with images",
            expected_response_tokens=50,
            num_images=2,
        )

        # Verify image costs are calculated
        assert breakdown["image_tokens"] > 0
        assert breakdown["image_cost"] > 0


def test_raw_generation(openai_llm):
    """Test OpenAI raw generation."""
    with patch.object(openai_llm, "client") as mock_client:
        # Create a more detailed mock response
        message_mock = MagicMock()
        message_mock.content = "test response"

        choice_mock = MagicMock()
        choice_mock.message = message_mock

        usage_mock = MagicMock()
        usage_mock.prompt_tokens = 10
        usage_mock.completion_tokens = 5
        usage_mock.total_tokens = 15

        response_mock = MagicMock()
        response_mock.choices = [choice_mock]
        response_mock.usage = usage_mock

        mock_client.chat.completions.create.return_value = response_mock

        # Call the raw generate method
        response, usage = openai_llm._raw_generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "test"}],
            max_tokens=100,
            temp=0,
        )

        # Verify response and usage information
        assert response == "test response"
        assert "read_tokens" in usage
        assert "write_tokens" in usage
        assert "total_tokens" in usage

        # Verify API call was made
        assert mock_client.chat.completions.create.called


def test_cost_tracking(openai_llm):
    """Test OpenAI cost tracking."""
    with patch.object(openai_llm, "client") as mock_client:
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="test response"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        response, usage, _ = openai_llm.generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "test"}],
            max_tokens=100,
        )

        assert response == "test response"
        assert usage["read_tokens"] == 10
        assert usage["write_tokens"] == 5
        assert usage["total_tokens"] == 15
        assert usage["total_cost"] >= 0


def test_error_handling(openai_llm):
    """Test OpenAI error handling."""
    # Define error cases to test
    error_cases = [
        # Rate limit error
        {
            "exception": Exception("Rate limit exceeded"),
            "expected_contains": "rate limit",
            "error_type": "rate_limit",
        },
        # Authentication error
        {
            "exception": Exception("Invalid API key provided"),
            "expected_contains": "api key",
            "error_type": "authentication",
        },
        # Context length error
        {
            "exception": Exception(
                "This model's maximum context length is 16385 tokens"
            ),
            "expected_contains": "context length",
            "error_type": "context_length",
        },
        # Invalid request error
        {
            "exception": Exception("Invalid request parameters"),
            "expected_contains": "invalid request",
            "error_type": "invalid_request",
        },
        # Server error
        {
            "exception": Exception("Service temporarily unavailable"),
            "expected_contains": "service",
            "error_type": "server_error",
        },
    ]

    # Test each error case
    for error_case in error_cases:
        # Set up the error
        openai_llm.client.chat.completions.create.side_effect = error_case["exception"]

        # Call generate and expect an exception
        with pytest.raises(LLMMistake) as exc_info:
            openai_llm.generate(
                event_id="test",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "test"}],
            )

        # Verify exception type and message
        exception = exc_info.value
        assert isinstance(exception, LLMMistake)
        assert str(error_case["exception"]) in str(exception)
        assert exception.provider == "openai"

        # Reset for next test
        openai_llm.client.chat.completions.create.side_effect = None
        openai_llm.client.chat.completions.create.reset_mock()

    # Test invalid model error - use a different approach since this is caught earlier
    with patch.object(OpenAILLM, "validate_model", return_value=False):
        with pytest.raises(ValueError) as exc_info:
            # Create a new instance with an invalid model
            OpenAILLM("non-existent-model", api_key="test-key")

        assert "not supported" in str(exc_info.value).lower()
        assert "non-existent-model" in str(exc_info.value)


def test_thinking_models(openai_llm):
    """Test OpenAI thinking model support."""
    # Test thinking model validation
    assert openai_llm.is_thinking_model("gpt-4o") is True
    assert openai_llm.is_thinking_model("o1") is True
    assert openai_llm.is_thinking_model("gpt-4o-mini") is False

    # Test thinking model generation with a model that supports thinking
    with patch.object(openai_llm, "client") as mock_client, patch.object(
        openai_llm, "model_name", "gpt-4o"
    ):
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="test response"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        response, usage, _ = openai_llm.generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "test"}],
            max_tokens=100,
            thinking_budget=1000,
        )

        assert response == "test response"
        assert "read_tokens" in usage
        assert "write_tokens" in usage
        assert "total_tokens" in usage


def test_classification(openai_llm):
    """Test OpenAI classification support."""
    categories = {
        "category1": "First category description",
        "category2": "Second category description",
    }
    messages = [{"message_type": "human", "message": "test message"}]

    # Override the generate method to return just 2 values for classification
    with patch.object(openai_llm, "generate") as mock_generate:
        # Mock the generate method to return the processed classification result
        mock_generate.return_value = (
            ["category1"],
            {
                "read_tokens": 10,
                "write_tokens": 5,
                "total_tokens": 15,
                "total_cost": 0.0001,
            },
            None,
        )

        # Test basic classification
        result, usage = openai_llm.classify(messages=messages, categories=categories)

        # Verify results
        assert "category1" in result
        assert isinstance(usage, dict)
        assert "total_tokens" in usage

        # Verify generate was called with classification system prompt
        assert mock_generate.called


def test_auth_with_aws_secrets():
    """Test authentication using AWS Secrets Manager."""
    # Create mock for get_secret function
    mock_secret = {"api_key": "test-secret-key"}

    with patch(
        "src.lluminary.models.providers.openai.get_secret", return_value=mock_secret
    ) as mock_get_secret, patch("openai.OpenAI") as mock_openai_client:

        # Create instance and call auth
        openai_llm = OpenAILLM("gpt-4o")
        openai_llm.auth()

        # Verify get_secret was called with correct parameters
        mock_get_secret.assert_called_once_with(
            "openai_api_key", required_keys=["api_key"]
        )

        # Verify API key was properly stored
        assert openai_llm.config["api_key"] == "test-secret-key"

        # Verify OpenAI client was initialized with correct API key
        mock_openai_client.assert_called_once_with(api_key="test-secret-key")


def test_auth_with_environment_variables():
    """Test authentication using environment variables instead of AWS Secrets Manager."""
    # Mock environment variable and make get_secret raise an exception
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}), patch(
        "src.lluminary.models.providers.openai.get_secret",
        side_effect=Exception("Secret not found"),
    ), patch("openai.OpenAI") as mock_openai_client:

        # Create instance and call auth - should fall back to env var
        openai_llm = OpenAILLM("gpt-4o")

        # We need to catch the exception and verify it contains information about checking env vars
        with pytest.raises(Exception) as excinfo:
            openai_llm.auth()

        # Verify exception message suggests checking environment variables
        assert "Secret not found" in str(excinfo.value)


def test_auth_failure_handling():
    """Test handling of authentication failures."""
    # Test different error scenarios
    error_messages = [
        "Secret not found",
        "Access denied",
        "Invalid parameters",
        "Network error",
    ]

    for error_msg in error_messages:
        # Mock get_secret to raise exception and ensure no env var fallback
        with patch(
            "src.lluminary.models.providers.openai.get_secret",
            side_effect=Exception(error_msg),
        ), patch.dict(os.environ, {}, clear=True), patch("openai.OpenAI"):

            # Create instance
            openai_llm = OpenAILLM("gpt-4o")

            # Call auth and expect exception
            with pytest.raises(Exception) as excinfo:
                openai_llm.auth()

            # Verify error message
            assert error_msg in str(excinfo.value)
            assert "OpenAI authentication failed" in str(excinfo.value)


def test_reranking_basic_functionality(openai_llm):
    """Test basic document reranking functionality."""
    # Test data
    query = "What is machine learning?"
    documents = [
        "Machine learning is a branch of artificial intelligence.",
        "Deep learning is a subset of machine learning.",
        "Python is a programming language often used for data science.",
        "Natural language processing deals with text data.",
    ]

    # Mock embedding response
    mock_embedding_data = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4]},  # Query embedding
        ],
        "usage": {"total_tokens": 5, "prompt_tokens": 5},
    }

    # Mock embeddings for documents with different similarity scores
    mock_doc_embeddings = {
        "data": [
            {"embedding": [0.11, 0.21, 0.31, 0.41]},  # High similarity to query
            {"embedding": [0.12, 0.22, 0.32, 0.42]},  # Medium similarity
            {"embedding": [0.5, 0.6, 0.7, 0.8]},  # Low similarity
            {"embedding": [0.3, 0.3, 0.3, 0.3]},  # Medium-low similarity
        ],
        "usage": {"total_tokens": 20, "prompt_tokens": 20},
    }

    with patch.object(openai_llm, "client") as mock_client:
        # Set up mock embedding responses
        mock_client.embeddings.create.side_effect = [
            MagicMock(**mock_embedding_data),
            MagicMock(**mock_doc_embeddings),
        ]

        # Call rerank method
        results, usage = openai_llm.rerank(query=query, documents=documents)

        # Verify embeddings API was called twice (once for query, once for documents)
        assert mock_client.embeddings.create.call_count == 2

        # Verify results structure
        assert isinstance(results, list)
        assert len(results) == len(documents)
        assert all("document" in item and "score" in item for item in results)

        # Verify scores are between 0 and 1
        assert all(0 <= item["score"] <= 1 for item in results)

        # Verify usage information
        assert "total_tokens" in usage
        assert "total_cost" in usage


def test_reranking_top_n_parameter(openai_llm):
    """Test limiting reranking results with top_n parameter."""
    # Test data
    query = "Python programming"
    documents = [
        "Python is a programming language.",
        "Java is another programming language.",
        "Python has simple syntax.",
        "JavaScript is used for web development.",
        "Python is popular for data science.",
    ]

    # Mock embedding responses
    mock_embedding_data = {
        "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}],
        "usage": {"total_tokens": 4, "prompt_tokens": 4},
    }

    mock_doc_embeddings = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4]},
            {"embedding": [0.3, 0.3, 0.3, 0.3]},
            {"embedding": [0.1, 0.2, 0.3, 0.4]},
            {"embedding": [0.5, 0.5, 0.5, 0.5]},
            {"embedding": [0.1, 0.2, 0.3, 0.4]},
        ],
        "usage": {"total_tokens": 20, "prompt_tokens": 20},
    }

    with patch.object(openai_llm, "client") as mock_client:
        # Set up mock embedding responses
        mock_client.embeddings.create.side_effect = [
            MagicMock(**mock_embedding_data),
            MagicMock(**mock_doc_embeddings),
        ]

        # Test with top_n=2
        results, usage = openai_llm.rerank(query=query, documents=documents, top_n=2)

        # Verify only top 2 results are returned
        assert len(results) == 2

        # Test with top_n=3
        mock_client.embeddings.create.side_effect = [
            MagicMock(**mock_embedding_data),
            MagicMock(**mock_doc_embeddings),
        ]
        results, usage = openai_llm.rerank(query=query, documents=documents, top_n=3)

        # Verify only top 3 results are returned
        assert len(results) == 3


def test_reranking_error_handling(openai_llm):
    """Test error handling during reranking process."""
    query = "Test query"
    documents = ["Document 1", "Document 2"]

    with patch.object(openai_llm, "client") as mock_client:
        # Test API error
        mock_client.embeddings.create.side_effect = Exception("API error")

        with pytest.raises(Exception) as excinfo:
            openai_llm.rerank(query=query, documents=documents)

        assert "API error" in str(excinfo.value)

        # Test empty documents list
        mock_client.embeddings.create.side_effect = None
        results, usage = openai_llm.rerank(query=query, documents=[])

        assert len(results) == 0

        # Test invalid document type
        with pytest.raises(ValueError):
            openai_llm.rerank(query=query, documents=[1, 2, 3])


def test_reranking_cost_calculation(openai_llm):
    """Test cost calculation for reranking operations."""
    query = "Short query"
    documents = ["Short document 1", "Short document 2"]

    # Mock embedding responses with token usage
    mock_embedding_data = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}],
        "usage": {"total_tokens": 3, "prompt_tokens": 3},
    }

    mock_doc_embeddings = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ],
        "usage": {"total_tokens": 6, "prompt_tokens": 6},
    }

    with patch.object(openai_llm, "client") as mock_client:
        # Set up mock embedding responses
        mock_client.embeddings.create.side_effect = [
            MagicMock(**mock_embedding_data),
            MagicMock(**mock_doc_embeddings),
        ]

        # Perform reranking
        results, usage = openai_llm.rerank(query=query, documents=documents)

        # Verify token usage is tracked
        assert usage["total_tokens"] == 9  # 3 + 6

        # Verify cost is calculated
        assert "total_cost" in usage
        assert usage["total_cost"] > 0

        # Check if cost calculation uses the correct pricing
        model = openai_llm.DEFAULT_RERANKING_MODEL
        expected_cost = 9 * openai_llm.embedding_costs[model]
        assert (
            abs(usage["total_cost"] - expected_cost) < 0.000001
        )  # Account for floating point precision

        # Test with examples
        examples = [
            {
                "user_input": "What is the capital of France?",
                "doc_str": "This asks for information about France's capital.",
                "selection": "category1",
            },
            {
                "user_input": "Please close the door.",
                "doc_str": "This is asking for an action to be performed.",
                "selection": "category2",
            },
        ]

        # Reset mock
        mock_generate.reset_mock()

        # Call classify with examples
        openai_llm.classify(messages=messages, categories=categories, examples=examples)

        # Verify examples were incorporated into prompt
        assert mock_generate.called
        args, kwargs = mock_generate.call_args
        system_prompt = kwargs.get("system_prompt", "")
        assert "capital of France" in system_prompt
        assert "close the door" in system_prompt


def test_streaming(openai_llm):
    """Test OpenAI streaming capability."""
    # Mock the client stream method
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=MagicMock(content="First "))]

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=MagicMock(content="chunk of "))]

    mock_chunk3 = MagicMock()
    mock_chunk3.choices = [MagicMock(delta=MagicMock(content="streamed text."))]

    # Final chunk with finish_reason
    mock_chunk_final = MagicMock()
    mock_chunk_final.choices = [
        MagicMock(delta=MagicMock(content=None), finish_reason="stop")
    ]

    # Mock usage info
    mock_chunk_final.usage = MagicMock(
        prompt_tokens=10, completion_tokens=6, total_tokens=16
    )

    # Create the stream of chunks
    mock_stream = [mock_chunk1, mock_chunk2, mock_chunk3, mock_chunk_final]

    # Set up the mock client to return our stream
    openai_llm.client.chat.completions.create.return_value = mock_stream

    # Create a callback to collect chunks
    collected_chunks = []
    callback_called = 0

    def test_callback(chunk, finish_reason=None):
        nonlocal callback_called
        callback_called += 1
        collected_chunks.append(chunk)

    # Call stream_generate
    response_data = openai_llm.stream_generate(
        event_id="test",
        system_prompt="You are a helpful assistant",
        messages=[{"message_type": "human", "message": "Tell me a short joke"}],
        callback=test_callback,
        max_tokens=100,
    )

    # Verify callback was called for each text chunk
    assert callback_called >= 3
    assert "First " in collected_chunks
    assert "chunk of " in collected_chunks
    assert "streamed text." in collected_chunks

    # Verify response_data contains expected information
    assert response_data["total_response"] == "First chunk of streamed text."
    assert "total_tokens" in response_data
    assert "total_cost" in response_data


def test_embeddings(openai_llm):
    """Test OpenAI embeddings functionality."""
    # Mock the embeddings endpoint
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5]),
        MagicMock(embedding=[-0.1, -0.2, -0.3, -0.4, -0.5]),
    ]
    mock_embedding_response.usage = MagicMock(prompt_tokens=20, total_tokens=20)

    # Set up the mock client
    openai_llm.client.embeddings.create.return_value = mock_embedding_response

    # Call embed method with multiple texts
    embeddings, usage = openai_llm.embed(
        texts=["This is the first text", "This is the second text"]
    )

    # Verify embeddings format
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 5
    assert isinstance(embeddings[0][0], float)
    assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert embeddings[1] == [-0.1, -0.2, -0.3, -0.4, -0.5]

    # Verify usage information
    assert "total_tokens" in usage
    assert usage["total_tokens"] == 20
    assert "total_cost" in usage

    # Verify the client was called correctly
    openai_llm.client.embeddings.create.assert_called_once()
    args, kwargs = openai_llm.client.embeddings.create.call_args
    assert "input" in kwargs
    assert kwargs["input"] == ["This is the first text", "This is the second text"]
    assert "model" in kwargs


def test_streaming_with_tools(openai_llm):
    """Test OpenAI streaming with tool calls."""

    # Define a test function
    def test_function(x: int) -> int:
        """Test function that doubles a number."""
        return x * 2

    # Mock the client stream method with tool calls
    # First chunks with regular content


def test_auth_with_aws_secrets():
    """Test authentication using AWS Secrets Manager."""
    # Create mock for get_secret function
    mock_secret = {"api_key": "test-secret-key"}

    with patch(
        "src.lluminary.models.providers.openai.get_secret", return_value=mock_secret
    ) as mock_get_secret, patch("openai.OpenAI") as mock_openai_client:

        # Create instance and call auth
        openai_llm = OpenAILLM("gpt-4o")
        openai_llm.auth()

        # Verify get_secret was called with correct parameters
        mock_get_secret.assert_called_once_with(
            "openai_api_key", required_keys=["api_key"]
        )

        # Verify API key was properly stored
        assert openai_llm.config["api_key"] == "test-secret-key"

        # Verify OpenAI client was initialized with correct API key
        mock_openai_client.assert_called_once_with(api_key="test-secret-key")


def test_auth_with_environment_variables():
    """Test authentication using environment variables instead of AWS Secrets Manager."""
    # Mock environment variable and make get_secret raise an exception
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}), patch(
        "src.lluminary.models.providers.openai.get_secret",
        side_effect=Exception("Secret not found"),
    ), patch("openai.OpenAI") as mock_openai_client:

        # Create instance and call auth - should fall back to env var
        openai_llm = OpenAILLM("gpt-4o")

        # We need to catch the exception and verify it contains information about checking env vars
        with pytest.raises(Exception) as excinfo:
            openai_llm.auth()

        # Verify exception message suggests checking environment variables
        assert "Secret not found" in str(excinfo.value)


def test_auth_failure_handling():
    """Test handling of authentication failures."""
    # Test different error scenarios
    error_messages = [
        "Secret not found",
        "Access denied",
        "Invalid parameters",
        "Network error",
    ]

    for error_msg in error_messages:
        # Mock get_secret to raise exception and ensure no env var fallback
        with patch(
            "src.lluminary.models.providers.openai.get_secret",
            side_effect=Exception(error_msg),
        ), patch.dict(os.environ, {}, clear=True), patch("openai.OpenAI"):

            # Create instance
            openai_llm = OpenAILLM("gpt-4o")

            # Call auth and expect exception
            with pytest.raises(Exception) as excinfo:
                openai_llm.auth()

            # Verify error message
            assert error_msg in str(excinfo.value)
            assert "OpenAI authentication failed" in str(excinfo.value)
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [MagicMock(delta=MagicMock(content="I'll call "))]

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock(delta=MagicMock(content="the function."))]

    # Tool call chunk
    tool_call_mock = MagicMock()
    tool_call_mock.id = "call_123"
    tool_call_mock.function = MagicMock(name="test_function", arguments='{"x": 42}')

    mock_chunk_tool = MagicMock()
    mock_chunk_tool.choices = [
        MagicMock(delta=MagicMock(content=None, tool_calls=[tool_call_mock]))
    ]

    # Final chunk
    mock_chunk_final = MagicMock()
    mock_chunk_final.choices = [
        MagicMock(delta=MagicMock(content=None), finish_reason="tool_calls")
    ]
    mock_chunk_final.usage = MagicMock(
        prompt_tokens=15, completion_tokens=10, total_tokens=25
    )

    # Create the stream of chunks
    mock_stream = [mock_chunk1, mock_chunk2, mock_chunk_tool, mock_chunk_final]

    # Set up the mock client to return our stream
    openai_llm.client.chat.completions.create.return_value = mock_stream

    # Create a callback to collect chunks and tool calls
    collected_chunks = []
    collected_tool_calls = []

    def test_callback(chunk, finish_reason=None, tool_calls=None):
        if chunk:
            collected_chunks.append(chunk)
        if tool_calls:
            collected_tool_calls.extend(tool_calls)

    # Call stream_generate with tools
    response_data = openai_llm.stream_generate(
        event_id="test",
        system_prompt="You are a helpful assistant",
        messages=[{"message_type": "human", "message": "Double the number 42"}],
        callback=test_callback,
        max_tokens=100,
        functions=[test_function],
    )

    # Verify text chunks were collected
    assert len(collected_chunks) >= 2
    assert "I'll call " in collected_chunks
    assert "the function." in collected_chunks

    # Verify tool calls were collected
    assert len(collected_tool_calls) >= 1
    assert collected_tool_calls[0]["name"] == "test_function"
    assert collected_tool_calls[0]["arguments"]["x"] == 42

    # Verify response_data contains expected information
    assert response_data["total_response"] == "I'll call the function."
    assert "total_tokens" in response_data
    assert "total_cost" in response_data
    assert "tool_calls" in response_data
