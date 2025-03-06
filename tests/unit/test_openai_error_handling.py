"""
Tests for OpenAI provider error handling functionality.
"""
import json
import os
import pytest
from unittest.mock import MagicMock, patch

from src.llmhandler.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ContentError,
    FormatError,
    LLMHandlerError,
    ProviderError,
    RateLimitError,
    ServiceUnavailableError,
    ToolError,
)
from src.llmhandler.models.providers.openai import OpenAILLM

# Mock OpenAI exception classes for testing
class MockOpenAIAuthenticationError(Exception):
    """Mock OpenAI authentication error."""
    pass

class MockOpenAIRateLimitError(Exception):
    """Mock OpenAI rate limit error."""
    def __init__(self, message, headers=None):
        self.headers = headers or {}
        super().__init__(message)

class MockOpenAIBadRequestError(Exception):
    """Mock OpenAI bad request error."""
    pass

class MockOpenAIAPIError(Exception):
    """Mock OpenAI API error."""
    pass

class MockOpenAIAPITimeoutError(Exception):
    """Mock OpenAI API timeout error."""
    pass

class MockOpenAIAPIConnectionError(Exception):
    """Mock OpenAI API connection error."""
    pass


@pytest.fixture
def openai_mock():
    """Create a mock for the OpenAI module."""
    with patch("src.llmhandler.models.providers.openai.OpenAI") as mock_openai:
        # Set up the mock module structure
        openai_module = MagicMock()
        openai_module.AuthenticationError = MockOpenAIAuthenticationError
        openai_module.RateLimitError = MockOpenAIRateLimitError
        openai_module.BadRequestError = MockOpenAIBadRequestError 
        openai_module.APIError = MockOpenAIAPIError
        openai_module.APITimeoutError = MockOpenAIAPITimeoutError
        openai_module.APIConnectionError = MockOpenAIAPIConnectionError
        
        # Mock the client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Let the module correctly have access to these exception classes
        with patch("src.llmhandler.models.providers.openai.openai", openai_module):
            yield mock_openai, mock_client, openai_module


@pytest.fixture
def openai_llm():
    """Create an OpenAILLM instance with mock API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
        with patch("src.llmhandler.models.providers.openai.OpenAI"):
            with patch("src.llmhandler.models.providers.openai.requests"):
                llm = OpenAILLM(model_name="gpt-4o")
                yield llm


class TestOpenAIErrorMapping:
    """Test error mapping functionality."""

    def test_authentication_error(self, openai_llm, openai_mock):
        """Test mapping of authentication errors."""
        _, _, openai_module = openai_mock
        
        # Create auth error
        error = openai_module.AuthenticationError("Invalid API key")
        
        # Map the error
        mapped_error = openai_llm._map_openai_error(error)
        
        # Verify mapping
        assert isinstance(mapped_error, AuthenticationError)
        assert "authentication" in str(mapped_error).lower()
        assert mapped_error.provider == "openai"
        assert "Invalid API key" in str(mapped_error.details["error"])
    
    def test_rate_limit_error(self, openai_llm, openai_mock):
        """Test mapping of rate limit errors."""
        _, _, openai_module = openai_mock
        
        # Create rate limit error with retry-after header
        headers = {"retry-after": "30"}
        error = openai_module.RateLimitError("Rate limit exceeded", headers=headers)
        
        # Map the error 
        mapped_error = openai_llm._map_openai_error(error)
        
        # Verify mapping
        assert isinstance(mapped_error, RateLimitError)
        assert "rate limit" in str(mapped_error).lower()
        assert mapped_error.provider == "openai"
        assert mapped_error.retry_after == 30
    
    def test_context_length_error(self, openai_llm, openai_mock):
        """Test mapping of context length errors."""
        _, _, openai_module = openai_mock
        
        # Create context length error
        error = openai_module.BadRequestError("context_length_exceeded: The model's context window has been exceeded")
        
        # Map the error
        mapped_error = openai_llm._map_openai_error(error)
        
        # Verify mapping
        assert isinstance(mapped_error, ProviderError)  # Not LLMMistake since it's not recoverable
        assert "context length" in str(mapped_error).lower()
        assert mapped_error.provider == "openai"
        assert "model" in mapped_error.details
        assert "context_window" in mapped_error.details
    
    def test_content_policy_error(self, openai_llm, openai_mock):
        """Test mapping of content policy errors."""
        _, _, openai_module = openai_mock
        
        # Create content policy error
        error = openai_module.BadRequestError("content_policy_violation: Your request was rejected")
        
        # Map the error
        mapped_error = openai_llm._map_openai_error(error)
        
        # Verify mapping
        assert isinstance(mapped_error, ContentError)
        assert "content policy" in str(mapped_error).lower()
        assert mapped_error.provider == "openai"
    
    def test_server_error(self, openai_llm, openai_mock):
        """Test mapping of server errors."""
        _, _, openai_module = openai_mock
        
        # Create server error
        error = openai_module.APIError("server_error: Internal server error")
        
        # Map the error
        mapped_error = openai_llm._map_openai_error(error)
        
        # Verify mapping
        assert isinstance(mapped_error, ServiceUnavailableError)
        assert "server error" in str(mapped_error).lower()
        assert mapped_error.provider == "openai"


class TestOpenAIRetryMechanism:
    """Test retry mechanism functionality."""
    
    def test_successful_call(self, openai_llm):
        """Test successful API call without retries."""
        mock_func = MagicMock()
        mock_func.return_value = "success"
        
        # Call with retry
        result = openai_llm._call_with_retry(mock_func, arg1="value1", arg2="value2")
        
        # Verify call was made once with correct arguments
        assert result == "success"
        mock_func.assert_called_once_with(arg1="value1", arg2="value2")
    
    def test_retry_on_rate_limit(self, openai_llm, openai_mock):
        """Test retry on rate limit error."""
        _, _, openai_module = openai_mock
        
        # Create mock function that fails twice with rate limit then succeeds
        mock_func = MagicMock()
        rate_limit_error = openai_module.RateLimitError("Rate limit exceeded")
        mock_func.side_effect = [rate_limit_error, rate_limit_error, "success"]
        
        # Patch sleep to avoid actual delays
        with patch("time.sleep") as mock_sleep:
            # Call with retry
            result = openai_llm._call_with_retry(
                mock_func, 
                max_retries=3,
                initial_backoff=0.1,
                arg1="value1"
            )
        
        # Verify results
        assert result == "success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2  # Called twice for two retries
    
    def test_exhausted_retries(self, openai_llm, openai_mock):
        """Test exception when retries are exhausted."""
        _, _, openai_module = openai_mock
        
        # Create mock function that always fails with rate limit
        mock_func = MagicMock()
        rate_limit_error = openai_module.RateLimitError("Rate limit exceeded")
        mock_func.side_effect = rate_limit_error
        
        # Patch sleep to avoid actual delays
        with patch("time.sleep"):
            # Call with retry and verify it raises the mapped error
            with pytest.raises(RateLimitError) as exc_info:
                openai_llm._call_with_retry(
                    mock_func, 
                    max_retries=2,
                    initial_backoff=0.1
                )
        
        # Verify call count
        assert mock_func.call_count == 3  # Initial call + 2 retries
        assert "rate limit" in str(exc_info.value).lower()
    
    def test_non_retryable_error(self, openai_llm, openai_mock):
        """Test immediate failure on non-retryable errors."""
        _, _, openai_module = openai_mock
        
        # Create mock function that fails with authentication error
        mock_func = MagicMock()
        auth_error = openai_module.AuthenticationError("Invalid API key")
        mock_func.side_effect = auth_error
        
        # Call with retry and verify it raises immediately without retrying
        with pytest.raises(AuthenticationError) as exc_info:
            openai_llm._call_with_retry(mock_func)
        
        # Verify call count
        assert mock_func.call_count == 1  # Only called once, no retries
        assert "authentication" in str(exc_info.value).lower()


class TestOpenAIImageProcessing:
    """Test image processing error handling."""
    
    def test_nonexistent_image_file(self, openai_llm):
        """Test handling of nonexistent image files."""
        with pytest.raises(ProviderError) as exc_info:
            openai_llm._encode_image("/path/to/nonexistent/image.jpg")
        
        assert "not found" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert "image_path" in exc_info.value.details
    
    def test_image_url_download_error(self, openai_llm):
        """Test handling of image URL download errors."""
        # Mock a RequestException from requests
        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection error")
            
            with pytest.raises(ProviderError) as exc_info:
                openai_llm._encode_image_url("https://example.com/image.jpg")
        
        assert "download" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert "url" in exc_info.value.details
    
    def test_image_processing_error(self, openai_llm):
        """Test handling of image processing errors."""
        # Mock successful download but failed processing
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_get.return_value = mock_response
            
            with patch("PIL.Image.open") as mock_open:
                mock_open.side_effect = Exception("Invalid image format")
                
                with pytest.raises(FormatError) as exc_info:
                    openai_llm._encode_image_url("https://example.com/image.jpg")
        
        assert "process" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
        assert "url" in exc_info.value.details


class TestOpenAIRawGenerate:
    """Test raw_generate error handling."""
    
    def test_empty_response_error(self, openai_llm):
        """Test handling of empty response from API."""
        # Mock a response with empty choices
        mock_response = MagicMock()
        mock_response.choices = []
        
        with patch.object(openai_llm, "_call_with_retry", return_value=mock_response):
            with patch.object(openai_llm, "_format_messages_for_model", return_value=[]):
                with pytest.raises(ContentError) as exc_info:
                    openai_llm._raw_generate(
                        event_id="test",
                        system_prompt="Test prompt",
                        messages=[{"message_type": "human", "message": "Hello"}]
                    )
        
        assert "empty choices" in str(exc_info.value).lower()
        assert exc_info.value.provider == "openai"
    
    def test_empty_content_with_tool_calls(self, openai_llm):
        """Test handling of empty content with tool calls."""
        # Mock a response with null content but valid tool calls
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "test_function"
        tool_call.function.arguments = '{"param": "value"}'
        mock_response.choices[0].message.tool_calls = [tool_call]
        
        # Mock usage data
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        with patch.object(openai_llm, "_call_with_retry", return_value=mock_response):
            with patch.object(openai_llm, "_format_messages_for_model", return_value=[]):
                # This should not raise an error
                result, usage = openai_llm._raw_generate(
                    event_id="test",
                    system_prompt="Test prompt",
                    messages=[{"message_type": "human", "message": "Hello"}]
                )
                
                # Content should be empty string
                assert result == ""
                # Tool use should be captured
                assert "tool_use" in usage
                assert usage["tool_use"]["id"] == "call_123"
    
    def test_empty_content_without_tool_calls(self, openai_llm):
        """Test handling of empty content without tool calls."""
        # Mock a response with null content and no tool calls
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = None
        
        with patch.object(openai_llm, "_call_with_retry", return_value=mock_response):
            with patch.object(openai_llm, "_format_messages_for_model", return_value=[]):
                with pytest.raises(ContentError) as exc_info:
                    openai_llm._raw_generate(
                        event_id="test",
                        system_prompt="Test prompt",
                        messages=[{"message_type": "human", "message": "Hello"}]
                    )
        
        assert "empty response" in str(exc_info.value).lower()
        assert "no tool calls" in str(exc_info.value).lower()
    
    def test_tool_json_parsing_error(self, openai_llm):
        """Test handling of invalid JSON in tool arguments."""
        # Mock a response with invalid JSON in tool arguments
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Some content"
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "test_function"
        tool_call.function.arguments = '{invalid json}'  # Invalid JSON
        mock_response.choices[0].message.tool_calls = [tool_call]
        
        # Mock usage data
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        # This should still complete without error, but warn about invalid JSON
        with patch.object(openai_llm, "_call_with_retry", return_value=mock_response):
            with patch.object(openai_llm, "_format_messages_for_model", return_value=[]):
                # This should not raise an error
                result, usage = openai_llm._raw_generate(
                    event_id="test",
                    system_prompt="Test prompt",
                    messages=[{"message_type": "human", "message": "Hello"}]
                )
                
                # Should still have content
                assert result == "Some content"
                # Tool use should be captured with raw string
                assert "tool_use" in usage
                assert usage["tool_use"]["input"] == '{invalid json}'