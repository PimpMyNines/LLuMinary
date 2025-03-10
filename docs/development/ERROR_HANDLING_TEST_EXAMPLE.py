"""Example test file for standardized error handling implementation.

This file demonstrates the recommended approach to testing error handling in
LLM provider implementations for the LLuMinary library.

Overview:
---------
This example shows how to write comprehensive tests for error handling in
LLM providers, including authentication errors, rate limiting, configuration
errors, and response validation errors.

Usage:
------
Use this file as a reference when implementing tests for error handling
in provider modules. The patterns demonstrated here can be adapted for
all LLuMinary providers.

Related Files:
-------------
- exceptions.py: Contains the exception hierarchy
- openai.py: OpenAI provider implementation
- test_openai_error_handling.py: Full test suite for OpenAI error handling
"""

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ContentError,
    FormatError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
    ToolError,
)
from lluminary.models.providers.openai import OpenAILLM


class TestOpenAIErrorHandling:
    """Tests for OpenAI provider error handling."""

    @pytest.fixture
    def mock_openai_client(self) -> Generator[MagicMock, None, None]:
        """Create a mock OpenAI client."""
        with patch("openai.OpenAI") as mock_client:
            # Mock the models.list method
            models_mock = MagicMock()
            mock_client.return_value.models = models_mock

            # Mock the chat.completions.create method
            chat_mock = MagicMock()
            mock_client.return_value.chat = chat_mock

            yield mock_client

    def test_authentication_error(self, mock_openai_client: MagicMock) -> None:
        """Test handling of authentication errors."""
        # Setup mock to raise authentication error
        import openai

        mock_openai_client.side_effect = openai.AuthenticationError("Invalid API key")

        # Test that our provider properly wraps this as an AuthenticationError
        with pytest.raises(AuthenticationError) as exc_info:
            _ = OpenAILLM(model_name="gpt-4o")

        # Verify error details
        assert "authentication failed" in str(exc_info.value)
        assert exc_info.value.provider == "openai"
        assert "Invalid API key" in str(exc_info.value)

    def test_rate_limit_error_with_retry(self, mock_openai_client: MagicMock) -> None:
        """Test handling of rate limit errors with retry logic."""
        import openai

        # Mock the chat completions create method
        chat_completions = mock_openai_client.return_value.chat.completions

        # First call raises rate limit, second call succeeds
        rate_limit_error = openai.RateLimitError("Rate limit exceeded")
        rate_limit_error.headers = {"retry-after": "2"}

        # Create a success response for the second attempt
        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "Test response"
        success_response.usage.prompt_tokens = 10
        success_response.usage.completion_tokens = 5
        success_response.usage.total_tokens = 15

        # Configure mock to fail once then succeed
        chat_completions.create.side_effect = [rate_limit_error, success_response]

        # Initialize provider
        llm = OpenAILLM(model_name="gpt-4o")

        # Test generation - should retry and succeed
        with patch("time.sleep") as mock_sleep:  # Mock sleep to avoid actual delays
            response, usage, _ = llm._raw_generate(
                event_id="test",
                system_prompt="Test prompt",
                messages=[{"message_type": "human", "message": "Hello"}],
                max_tokens=100,
            )

        # Verify retry was attempted (sleep was called)
        mock_sleep.assert_called_once()

        # Verify final response
        assert response == "Test response"
        assert usage["read_tokens"] == 10
        assert usage["write_tokens"] == 5

    def test_configuration_error(self) -> None:
        """Test handling of configuration errors."""
        # Test with invalid model name
        with pytest.raises(ConfigurationError) as exc_info:
            OpenAILLM(model_name="invalid-model-name")

        # Verify error details
        assert "not supported" in str(exc_info.value)
        assert exc_info.value.provider == "openai"

    def test_content_error(self, mock_openai_client: MagicMock) -> None:
        """Test handling of empty or invalid content."""
        # Mock a response with empty content
        response_mock = MagicMock()
        response_mock.choices = [MagicMock()]
        response_mock.choices[0].message.content = ""
        response_mock.usage.prompt_tokens = 10
        response_mock.usage.completion_tokens = 0

        mock_openai_client.return_value.chat.completions.create.return_value = (
            response_mock
        )

        # Initialize provider
        llm = OpenAILLM(model_name="gpt-4o")

        # Test generation with empty response
        with pytest.raises(ContentError) as exc_info:
            llm._raw_generate(
                event_id="test",
                system_prompt="Test prompt",
                messages=[{"message_type": "human", "message": "Hello"}],
                max_tokens=100,
            )

        # Verify error details
        assert "empty response" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"

    def test_server_error(self, mock_openai_client: MagicMock) -> None:
        """Test handling of server errors."""
        import openai

        # Mock a server error
        server_error = openai.APIError("Internal server error")
        mock_openai_client.return_value.chat.completions.create.side_effect = (
            server_error
        )

        # Initialize provider
        llm = OpenAILLM(model_name="gpt-4o")

        # Test generation with server error
        with pytest.raises(ServiceUnavailableError) as exc_info:
            llm._raw_generate(
                event_id="test",
                system_prompt="Test prompt",
                messages=[{"message_type": "human", "message": "Hello"}],
                max_tokens=100,
            )

        # Verify error details
        assert "server error" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert "Internal server error" in str(exc_info.value.details.get("error", ""))

    def test_timeout_error(self, mock_openai_client: MagicMock) -> None:
        """Test handling of timeout errors."""
        import openai

        # Mock a timeout error
        timeout_error = openai.APITimeoutError("Request timed out")
        mock_openai_client.return_value.chat.completions.create.side_effect = (
            timeout_error
        )

        # Initialize provider
        llm = OpenAILLM(model_name="gpt-4o")

        # Test generation with timeout error
        with pytest.raises(ProviderError) as exc_info:
            llm._raw_generate(
                event_id="test",
                system_prompt="Test prompt",
                messages=[{"message_type": "human", "message": "Hello"}],
                max_tokens=100,
            )

        # Verify error details
        assert "timeout" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"

    def test_image_processing_error(self) -> None:
        """Test handling of image processing errors."""
        # Initialize provider
        llm = OpenAILLM(model_name="gpt-4o")

        # Test with non-existent image path
        with pytest.raises(ProviderError) as exc_info:
            llm._encode_image("/path/to/nonexistent/image.jpg")

        # Verify error details
        assert "image" in str(exc_info.value).lower()
        assert "not found" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"

    def test_tool_error(self, mock_openai_client: MagicMock) -> None:
        """Test handling of tool usage errors."""
        # Mock a response with invalid tool calls
        response_mock = MagicMock()
        response_mock.choices = [MagicMock()]
        response_mock.choices[0].message.content = None

        # Create invalid tool call structure
        tool_call_mock = MagicMock()
        tool_call_mock.function.name = "test_function"
        tool_call_mock.function.arguments = "{invalid json"

        response_mock.choices[0].message.tool_calls = [tool_call_mock]
        response_mock.usage.prompt_tokens = 10
        response_mock.usage.completion_tokens = 5

        mock_openai_client.return_value.chat.completions.create.return_value = (
            response_mock
        )

        # Initialize provider
        llm = OpenAILLM(model_name="gpt-4o")

        # Test generation with tool call that has invalid JSON
        with pytest.raises(ToolError) as exc_info:
            llm._raw_generate(
                event_id="test",
                system_prompt="Test prompt",
                messages=[{"message_type": "human", "message": "Hello"}],
                max_tokens=100,
                tools=[{"name": "test_function"}],
            )

        # Verify error details
        assert "invalid" in str(exc_info.value).lower()
        assert "json" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"


@pytest.mark.parametrize(
    "exception_class,expected_attributes",
    [
        (AuthenticationError, {"provider": "test-provider", "message": "Auth failed"}),
        (RateLimitError, {"provider": "test-provider", "retry_after": 30}),
        (
            ConfigurationError,
            {"provider": "test-provider", "details": {"param": "value"}},
        ),
        (FormatError, {"provider": "test-provider", "error_type": "format"}),
        (ContentError, {"provider": "test-provider", "error_type": "content"}),
        (ToolError, {"provider": "test-provider", "error_type": "tool"}),
    ],
)
def test_exception_attributes(exception_class: type, expected_attributes: dict) -> None:
    """Test that exception classes have correct attributes."""
    # Create exception instance
    if exception_class == RateLimitError:
        exc = exception_class(
            message=expected_attributes.get("message", "Error"),
            provider=expected_attributes.get("provider"),
            retry_after=expected_attributes.get("retry_after"),
        )
    else:
        exc = exception_class(
            message=expected_attributes.get("message", "Error"),
            provider=expected_attributes.get("provider"),
            details=expected_attributes.get("details"),
        )

    # Verify attributes
    for key, value in expected_attributes.items():
        if key == "message":
            assert value in str(exc)
        elif key == "retry_after":
            assert exc.retry_after == value
        elif key != "details":
            assert getattr(exc, key) == value

    # Test to_dict method if it exists
    if hasattr(exc, "to_dict"):
        data = exc.to_dict()
        assert data["provider"] == expected_attributes.get("provider")
        if "error_type" in expected_attributes:
            assert data["error_type"] == expected_attributes["error_type"]
