"""
Helper module for mocking Google API components in tests.

This module provides factory functions and mock classes for simulating
Google's generative AI API responses, including content structure,
function calls, streaming, and error handling.
"""

from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

# Mock Google API Types and Content Structures


class MockPart:
    """Mock implementation of Google's Part class for message components."""

    @staticmethod
    def from_text(text: str) -> "MockPart":
        """Create a text part."""
        part = MockPart()
        part.text = text
        part.type = "text"
        return part

    @staticmethod
    def from_function_call(name: str, args: Dict[str, Any]) -> "MockPart":
        """Create a function call part."""
        part = MockPart()
        part.type = "function_call"
        part.function_call = MagicMock()
        part.function_call.name = name
        part.function_call.args = args
        return part

    @staticmethod
    def from_function_response(name: str, response: Dict[str, Any]) -> "MockPart":
        """Create a function response part."""
        part = MockPart()
        part.type = "function_response"
        part.function_response = MagicMock()
        part.function_response.name = name
        part.function_response.response = response
        return part


class MockContent:
    """Mock implementation of Google's Content class for message container."""

    def __init__(self, role: str = "user", parts: Optional[List[MockPart]] = None):
        self.role = role
        self.parts = parts or []


class MockGenerationConfig:
    """Mock for GenerationConfig with proper attribute storage."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockFunctionCall:
    """Mock for function call output in model responses."""

    def __init__(self, name: str, args: Dict[str, Any], call_id: str = "func_123"):
        self.name = name
        self.args = args
        self.id = call_id


class MockCandidate:
    """Mock for response candidates in streaming or thinking responses."""

    def __init__(self, index: int = 0, content: Optional[MockContent] = None):
        self.index = index
        self.content = content or MockContent()
        self.finish_reason = "STOP"


class MockResponse:
    """Mock response from Google's generate_content method."""

    def __init__(
        self,
        text: str = "Mock response from Google",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        function_calls: Optional[List[MockFunctionCall]] = None,
        candidates: Optional[List[MockCandidate]] = None,
    ):
        self.text = text

        # Set up usage_metadata
        self.usage_metadata = MagicMock()
        self.usage_metadata.prompt_token_count = prompt_tokens
        self.usage_metadata.candidates_token_count = completion_tokens
        self.usage_metadata.total_token_count = prompt_tokens + completion_tokens

        # Set up function calls
        self.function_calls = function_calls or []

        # Set up candidates
        self.candidates = candidates or []


class MockStreamResponse:
    """Mock for streaming responses from generate_content with stream=True."""

    def __init__(self, chunks: List[str], include_function_call: bool = False):
        """Initialize a streaming response with text chunks and optional function call."""
        self.chunks = []

        # Create text chunks
        for i, chunk in enumerate(chunks):
            mock_chunk = MagicMock()
            mock_chunk.text = chunk
            mock_chunk.candidates = []
            self.chunks.append(mock_chunk)

        # Add final empty chunk with function call if requested
        final_chunk = MagicMock()
        final_chunk.text = ""

        if include_function_call:
            # Create function call candidate
            mock_content = MockContent(role="model")
            mock_part = MockPart()
            mock_part.function_call = MockFunctionCall(
                name="test_function", args={"param1": "value1"}
            )
            mock_content.parts = [mock_part]

            mock_candidate = MockCandidate(content=mock_content)
            final_chunk.candidates = [mock_candidate]
        else:
            final_chunk.candidates = []

        self.chunks.append(final_chunk)

    def __iter__(self):
        """Make the mock response iterable."""
        return iter(self.chunks)


class MockGenerativeModel:
    """Mock for Google's GenerativeModel class."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        generation_config: Optional[Any] = None,
        client: Optional[Any] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        safety_settings: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.generation_config = generation_config
        self.client = client
        self.tools = tools
        self.safety_settings = safety_settings

        # Store initialization parameters for inspection in tests
        self.init_params = {
            "model_name": model_name,
            "generation_config": generation_config,
            "client": client,
            "tools": tools,
            "safety_settings": safety_settings,
        }

    def generate_content(
        self, contents: Any, stream: bool = False, **kwargs
    ) -> Union[MockResponse, MockStreamResponse]:
        """Mock generate_content with support for both regular and streaming responses."""
        # Store call parameters for inspection in tests
        self.last_generate_params = {"contents": contents, "stream": stream, **kwargs}

        if stream:
            # Return a streaming response
            return MockStreamResponse(
                chunks=["This ", "is ", "a ", "streaming ", "response"],
                include_function_call=bool(self.tools),
            )
        else:
            # Return a regular response
            return MockResponse(
                text="This is a regular response",
                function_calls=(
                    [MockFunctionCall(name="test_function", args={"param1": "value1"})]
                    if self.tools
                    else []
                ),
            )


# Factory Functions for Creating Mock Objects


def create_mock_client(api_version: str = "v1") -> MagicMock:
    """Create a mock Google API client with the specified API version."""
    mock_client = MagicMock()

    # Store API version for verification
    mock_client.api_version = api_version

    # Add models attribute with generate_content method
    mock_client.models = MagicMock()

    # Set up default response
    response = MockResponse()
    mock_client.models.generate_content.return_value = response

    # For thinking models
    if api_version == "v1alpha":
        mock_client.models.thinking_generate_content = MagicMock()
        mock_client.models.thinking_generate_content.return_value = response

    return mock_client


def create_image_mock() -> MagicMock:
    """Create a mock PIL Image for image processing tests."""
    mock_image = MagicMock()

    # Add verify method that does nothing
    mock_image.verify = MagicMock()

    # Return the mock image
    return mock_image


def create_path_mock(exists: bool = True, is_dir: bool = False) -> MagicMock:
    """Create a mock pathlib.Path with configurable behavior."""
    mock_path = MagicMock()

    # Configure behavior
    mock_path.exists.return_value = exists
    mock_path.is_dir.return_value = is_dir
    mock_path.absolute.return_value = "/absolute/path/to/mock"

    return mock_path


def create_mock_http_response(content: bytes = b"image_data") -> MagicMock:
    """Create a mock HTTP response for URL fetching."""
    mock_response = MagicMock()

    # Configure response properties
    mock_response.content = content
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    return mock_response


def create_mock_error_response(error_type: str) -> Exception:
    """Create a mock exception with appropriate error message based on type."""
    error_messages = {
        "rate_limit": "Rate limit exceeded for API requests",
        "authentication": "Invalid API key provided",
        "service_unavailable": "Service temporarily unavailable",
        "content_policy": "Content violates safety policy",
        "invalid_model": "Model gemini-invalid not found",
        "format_error": "Invalid JSON format in request",
        "tool_error": "Error executing function get_weather",
    }

    message = error_messages.get(error_type, f"Unknown error: {error_type}")

    # Create exception with the message
    return Exception(message)
