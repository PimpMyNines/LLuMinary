"""
Unit tests for Google provider error handling.

This module tests the comprehensive error handling implementation
in the Google LLM provider, including:

1. Error mapping from Google API errors to LLMHandler custom exceptions
2. Authentication error handling
3. Rate limit detection and retry mechanisms
4. Image processing error handling
5. Content policy and moderation error handling
6. Streaming error handling
"""

import os
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import requests
from lluminary.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ContentError,
    FormatError,
    LLMMistake,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
    ToolError,
)
from lluminary.models.providers.google import GoogleLLM
from PIL import UnidentifiedImageError


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


def test_map_google_error():
    """Test mapping Google API errors to LLMHandler custom exceptions."""
    llm = GoogleLLM("gemini-2.0-flash")

    # Test cases with different error messages and expected exception types
    test_cases = [
        # Authentication errors
        {"message": "API key not valid", "expected_type": AuthenticationError},
        {
            "message": "Invalid authentication credentials",
            "expected_type": AuthenticationError,
        },
        {"message": "Permission denied", "expected_type": AuthenticationError},
        {"message": "Access denied to resource", "expected_type": AuthenticationError},
        # Rate limit errors
        {"message": "Rate limit exceeded", "expected_type": RateLimitError},
        {"message": "Quota exceeded for quota metric", "expected_type": RateLimitError},
        {"message": "Too many requests", "expected_type": RateLimitError},
        {"message": "Resource exhausted", "expected_type": RateLimitError},
        # Service unavailability
        {"message": "Service unavailable", "expected_type": ServiceUnavailableError},
        {"message": "Internal server error", "expected_type": ServiceUnavailableError},
        {"message": "Backend error", "expected_type": ServiceUnavailableError},
        {"message": "Deadline exceeded", "expected_type": ServiceUnavailableError},
        {
            "message": "503 Service Temporarily Unavailable",
            "expected_type": ServiceUnavailableError,
        },
        # Configuration errors
        {"message": "Invalid model specified", "expected_type": ConfigurationError},
        {"message": "Model not found", "expected_type": ConfigurationError},
        {"message": "Invalid parameter value", "expected_type": ConfigurationError},
        {"message": "Unsupported operation", "expected_type": ConfigurationError},
        # Content policy violations
        {"message": "Content policy violation detected", "expected_type": ContentError},
        {
            "message": "Safety settings blocked the request",
            "expected_type": ContentError,
        },
        {"message": "Harmful content detected", "expected_type": ContentError},
        {
            "message": "Content moderation flagged the input",
            "expected_type": ContentError,
        },
        # Format errors
        {"message": "Invalid format in request", "expected_type": FormatError},
        {"message": "Invalid JSON in response", "expected_type": FormatError},
        {"message": "Parse error in content", "expected_type": FormatError},
        # Tool errors
        {"message": "Error in function call", "expected_type": ToolError},
        {"message": "Tool execution failed", "expected_type": ToolError},
        # Default fallback
        {"message": "Unknown error occurred", "expected_type": LLMMistake},
    ]

    # Test each case
    for case in test_cases:
        # Create an exception with the test message
        original_error = Exception(case["message"])

        # Map the error
        mapped_error = llm._map_google_error(original_error)

        # Verify correct type mapping
        assert isinstance(
            mapped_error, case["expected_type"]
        ), f"Expected {case['message']} to map to {case['expected_type'].__name__}, got {type(mapped_error).__name__}"

        # Verify error contains original message
        assert case["message"].lower() in str(mapped_error).lower()

        # Verify provider is set
        assert mapped_error.provider == "GoogleLLM"

        # Verify details contain original error
        assert "original_error" in mapped_error.details

        # For RateLimitError, check retry_after is set
        if isinstance(mapped_error, RateLimitError):
            assert mapped_error.retry_after is not None
            assert mapped_error.retry_after > 0


def test_auth_with_env_var():
    """Test authentication using environment variables."""
    # Save original environment
    original_env = os.environ.get("GOOGLE_API_KEY")

    try:
        # Set environment variable
        os.environ["GOOGLE_API_KEY"] = "env_test_key"

        # Create GoogleLLM instance with mocked client
        with patch("google.genai.Client") as mock_client:
            llm = GoogleLLM("gemini-2.0-flash")
            llm.auth()

            # Verify client was initialized with the env var API key
            mock_client.assert_called_once_with(api_key="env_test_key")
    finally:
        # Restore original environment
        if original_env:
            os.environ["GOOGLE_API_KEY"] = original_env
        else:
            del os.environ["GOOGLE_API_KEY"]


def test_auth_failures():
    """Test authentication failure scenarios."""
    # Test with missing API key
    with patch(
        "src.lluminary.utils.get_secret", side_effect=Exception("Secret not found")
    ):
        with patch("os.environ", {}):  # Ensure no env vars
            llm = GoogleLLM("gemini-2.0-flash")

            with pytest.raises(AuthenticationError) as exc_info:
                llm.auth()

            # Verify exception details
            error = exc_info.value
            assert "Failed to retrieve Google API key" in str(error)
            assert error.provider == "GoogleLLM"
            assert (
                "secret not found"
                in str(error.details.get("original_error", "")).lower()
            )

    # Test with client initialization failure
    with patch("lluminary.utils.get_secret", return_value={"api_key": "test_key"}):
        with patch(
            "google.genai.Client", side_effect=Exception("Client initialization failed")
        ):
            llm = GoogleLLM("gemini-2.0-flash")

            with pytest.raises(AuthenticationError) as exc_info:
                llm.auth()

            # Verify exception details
            error = exc_info.value
            assert "Failed to initialize Google client" in str(error)
            assert error.provider == "GoogleLLM"
            assert (
                "client initialization failed"
                in str(error.details.get("original_error", "")).lower()
            )


def test_call_with_retry():
    """Test the retry mechanism for transient errors."""
    llm = GoogleLLM("gemini-2.0-flash")

    # Mock function that fails with rate limit error twice, then succeeds
    mock_func = MagicMock()
    mock_func.side_effect = [
        RateLimitError("Rate limit exceeded", "GoogleLLM", 1),  # First call fails
        RateLimitError("Rate limit exceeded", "GoogleLLM", 1),  # Second call fails
        "success",  # Third call succeeds
    ]

    # Use the retry mechanism
    with patch("time.sleep"):  # Prevent actual sleeping
        result = llm._call_with_retry(mock_func, max_retries=3, retry_delay=1)

    # Verify function was called multiple times
    assert mock_func.call_count == 3

    # Verify final result
    assert result == "success"

    # Reset mock and test with service unavailable error
    mock_func.reset_mock()
    mock_func.side_effect = [
        ServiceUnavailableError("Service unavailable", "GoogleLLM"),
        "success",
    ]

    # Retry with service unavailable error
    with patch("time.sleep"):
        result = llm._call_with_retry(mock_func, max_retries=3, retry_delay=1)

    # Verify result
    assert mock_func.call_count == 2
    assert result == "success"

    # Test failure after max retries
    mock_func.reset_mock()
    mock_func.side_effect = RateLimitError("Persistent rate limit", "GoogleLLM", 1)

    # Should raise after max retries
    with patch("time.sleep"):
        with pytest.raises(RateLimitError) as exc_info:
            llm._call_with_retry(mock_func, max_retries=3, retry_delay=1)

    # Verify call count reached max retries + 1
    assert mock_func.call_count == 4

    # Verify non-retryable errors are raised immediately
    mock_func.reset_mock()
    mock_func.side_effect = AuthenticationError("Auth failed", "GoogleLLM")

    with pytest.raises(AuthenticationError):
        llm._call_with_retry(mock_func, max_retries=3, retry_delay=1)

    # Verify only called once for non-retryable error
    assert mock_func.call_count == 1


@patch("PIL.Image.open")
@patch("pathlib.Path.exists")
def test_image_processing_errors(mock_path_exists, mock_image_open, google_llm):
    """Test various image processing error scenarios."""
    # Set up image verification failures
    mock_image = MagicMock()
    mock_image.verify.side_effect = Exception("Invalid image data")
    mock_image_open.return_value = mock_image

    # Test nonexistent file path
    mock_path_exists.return_value = False

    with pytest.raises(LLMMistake) as exc_info:
        google_llm._process_image("/path/to/nonexistent.jpg")

    # Verify exception details
    error = exc_info.value
    assert "Image file not found" in str(error)
    assert error.error_type == "image_processing_error"
    assert error.provider == "GoogleLLM"

    # Test corrupt image file
    mock_path_exists.return_value = True

    with pytest.raises(ContentError) as exc_info:
        google_llm._process_image("/path/to/corrupt.jpg")

    # Verify exception details
    error = exc_info.value
    assert "Invalid or unsupported image format" in str(error)
    assert error.provider == "GoogleLLM"
    assert "original_error" in error.details

    # Test URL error handling
    with patch("requests.get") as mock_get:
        # 404 Not Found
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        with pytest.raises(LLMMistake) as exc_info:
            google_llm._process_image("https://example.com/missing.jpg", is_url=True)

        error = exc_info.value
        assert "Image URL not found (404)" in str(error)
        assert error.error_type == "image_url_error"

        # 403 Forbidden
        mock_resp.status_code = 403
        mock_resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")

        with pytest.raises(LLMMistake) as exc_info:
            google_llm._process_image("https://example.com/forbidden.jpg", is_url=True)

        error = exc_info.value
        assert "Access denied" in str(error)
        assert "unauthorized" in str(error)

        # Server error
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with pytest.raises(LLMMistake) as exc_info:
            google_llm._process_image(
                "https://example.com/server-error.jpg", is_url=True
            )

        error = exc_info.value
        assert "Server error" in str(error)

        # Connection error
        mock_get.side_effect = requests.ConnectionError("Failed to connect")

        with pytest.raises(LLMMistake) as exc_info:
            google_llm._process_image(
                "https://example.com/connection-error.jpg", is_url=True
            )

        error = exc_info.value
        assert "Failed to fetch image URL" in str(error)

        # Success case but invalid image data
        mock_get.side_effect = None
        mock_resp.raise_for_status.side_effect = None
        mock_resp.content = b"not an image"

        # Mock BytesIO and PIL handling
        with patch("io.BytesIO") as mock_bytesio:
            mock_bytesio.return_value = MagicMock()
            mock_image_open.side_effect = UnidentifiedImageError(
                "Cannot identify image"
            )

            with pytest.raises(ContentError) as exc_info:
                google_llm._process_image(
                    "https://example.com/invalid-format.jpg", is_url=True
                )

            error = exc_info.value
            assert "Invalid or unsupported image format" in str(error)
            assert "Cannot identify image" in str(
                error.details.get("original_error", "")
            )


def test_raw_generate_error_handling(google_llm, mock_google_client):
    """Test error handling in the _raw_generate method."""
    # Create basic test message
    messages = [{"message_type": "human", "message": "Test message"}]

    # Mock client to raise different types of errors
    client_instance = mock_google_client.return_value

    # Test auth error handling
    with patch.object(
        google_llm, "auth", side_effect=AuthenticationError("Auth failed", "GoogleLLM")
    ):
        # Force client to None to trigger auth
        google_llm.client = None

        with pytest.raises(AuthenticationError) as exc_info:
            google_llm._raw_generate(
                event_id="test", system_prompt="", messages=messages
            )

        error = exc_info.value
        assert "Auth failed" in str(error)
        assert error.provider == "GoogleLLM"

    # Reset client
    google_llm.client = client_instance

    # Test message formatting error handling
    with patch.object(
        google_llm,
        "_format_messages_for_model",
        side_effect=Exception("Failed to format message"),
    ):
        with pytest.raises(Exception) as exc_info:
            google_llm._raw_generate(
                event_id="test", system_prompt="", messages=messages
            )

        # Should be mapped to some exception type
        assert isinstance(exc_info.value, ProviderError)

    # Test API error mapping for rate limit errors
    client_instance.models.generate_content.side_effect = Exception(
        "Rate limit exceeded"
    )

    with pytest.raises(RateLimitError) as exc_info:
        google_llm._raw_generate(event_id="test", system_prompt="", messages=messages)

    error = exc_info.value
    assert "rate limit exceeded" in str(error).lower()
    assert error.provider == "GoogleLLM"
    assert error.retry_after > 0

    # Test service unavailable error
    client_instance.models.generate_content.side_effect = Exception(
        "Service unavailable"
    )

    with pytest.raises(ServiceUnavailableError) as exc_info:
        google_llm._raw_generate(event_id="test", system_prompt="", messages=messages)

    error = exc_info.value
    assert "service unavailable" in str(error).lower()
    assert error.provider == "GoogleLLM"

    # Test configuration error
    client_instance.models.generate_content.side_effect = Exception(
        "Invalid model configuration"
    )

    with pytest.raises(LLMMistake) as exc_info:
        google_llm._raw_generate(event_id="test", system_prompt="", messages=messages)

    # Test response extraction error
    mock_response = MagicMock()
    # Make the text property raise an exception when accessed
    type(mock_response).text = PropertyMock(side_effect=Exception("Cannot access text"))
    client_instance.models.generate_content.side_effect = None
    client_instance.models.generate_content.return_value = mock_response

    with pytest.raises(ContentError) as exc_info:
        google_llm._raw_generate(event_id="test", system_prompt="", messages=messages)

    error = exc_info.value
    assert "Failed to extract text" in str(error)
    assert error.provider == "GoogleLLM"


@patch("google.generativeai.GenerativeModel")
def test_stream_generate_error_handling(mock_generative_model, google_llm):
    """Test error handling in stream_generate method."""
    # Create basic test messages
    messages = [{"message_type": "human", "message": "Stream test"}]

    # Test authentication error
    with patch.object(
        google_llm, "auth", side_effect=AuthenticationError("Auth failed", "GoogleLLM")
    ):
        # Force client to trigger auth
        google_llm.client = None

        # Attempt to stream and expect AuthenticationError
        with pytest.raises(AuthenticationError):
            list(
                google_llm.stream_generate(
                    event_id="test", system_prompt="", messages=messages
                )
            )

    # Reset client
    google_llm.client = MagicMock()

    # Test message formatting error
    with patch.object(
        google_llm,
        "_format_messages_for_model",
        side_effect=Exception("Failed to format messages"),
    ):
        with pytest.raises(Exception) as exc_info:
            list(
                google_llm.stream_generate(
                    event_id="test", system_prompt="", messages=messages
                )
            )

        # Should be mapped to appropriate exception
        assert isinstance(exc_info.value, ProviderError)

    # Test rate limit error during streaming
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("Rate limit exceeded")
    mock_generative_model.return_value = mock_model

    with pytest.raises(RateLimitError) as exc_info:
        list(
            google_llm.stream_generate(
                event_id="test", system_prompt="", messages=messages
            )
        )

    error = exc_info.value
    assert "rate limit exceeded" in str(error).lower()
    assert error.provider == "GoogleLLM"

    # Test error during streaming iteration
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = iter(
        [MagicMock(text="First chunk"), MagicMock(text="Second chunk")]
    )
    # Setup __next__ to raise after a few iterations
    mock_stream.__iter__.side_effect = [
        iter([MagicMock(text="First chunk")]),  # First iteration works
        Exception("Stream connection lost"),  # Second call to iter fails
    ]
    mock_model.generate_content.side_effect = None
    mock_model.generate_content.return_value = mock_stream

    # This is complex to test due to generator nature, just verify mapping
    with patch.object(
        google_llm,
        "_map_google_error",
        return_value=ServiceUnavailableError("Test mapped error", "GoogleLLM"),
    ) as mock_map:
        with pytest.raises(ServiceUnavailableError):
            # Force iteration through all items
            for _ in google_llm.stream_generate(
                event_id="test", system_prompt="", messages=messages
            ):
                pass


def test_retry_in_raw_generate(google_llm, mock_google_client):
    """Test that retry mechanism is used in _raw_generate."""
    # Setup client to fail with rate limit once then succeed
    client_instance = mock_google_client.return_value

    # Create test messages
    messages = [{"message_type": "human", "message": "Test retry"}]

    # Mock call_with_retry to verify it's used
    with patch.object(google_llm, "_call_with_retry") as mock_retry:
        mock_retry.return_value = MagicMock(
            text="Success after retry",
            usage_metadata=MagicMock(
                prompt_token_count=10, candidates_token_count=5, total_token_count=15
            ),
        )

        # Call _raw_generate
        response, usage = google_llm._raw_generate(
            event_id="test", system_prompt="", messages=messages
        )

        # Verify _call_with_retry was used
        assert mock_retry.called
        assert response == "Success after retry"
        assert usage["read_tokens"] == 10
        assert usage["write_tokens"] == 5
        assert usage["total_tokens"] == 15
