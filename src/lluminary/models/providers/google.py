"""
Google LLM provider implementation.

This module implements support for Google's Gemini AI models within the LLM handler framework.
It handles authentication, message formatting, image processing, and cost tracking for Google's
Gemini models.

## Google Messaging System Overview

Google's Gemini models use a structured content format that differs from other providers:

1. Messages are organized as "contents" with specific roles (user, model, tool)
2. Each content contains "parts" that can include:
   - Text
   - Images
   - Function calls (tools)
   - Function responses

### Message Conversion Process
This implementation converts the standard LLM handler message format:
```
{
    "message_type": "human"|"ai"|"tool",
    "message": "text content",
    "image_paths": ["local/path/to/image.jpg"],
    "image_urls": ["https://example.com/image.jpg"],
    "tool_use": {"name": "func_name", "input": {...}},
    "tool_result": {"tool_id": "id", "success": bool, "result": any, "error": str}
}
```

Into Google's format:
```
Content(
    role="user"|"model"|"tool",
    parts=[
        Part.from_text("text content"),
        Image.from_file("local/path/to/image.jpg"),
        Part.from_function_call(name="func_name", args={...}),
        Part.from_function_response(name="func_name", response={...})
    ]
)
```

### Key Features
- Full support for text, images, and function calling
- Handles Google's model-specific versions and configuration
- Provides token usage and cost tracking
- Compatible with experimental "thinking" models
"""

import asyncio
import datetime
import json
import logging
import pathlib
import time
import uuid
import warnings
from io import BytesIO
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    TypedDict,
)

import requests
import google
import google.genai as genai
from google.genai import types
from PIL import Image

from ...exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContentError,
    LLMFormatError,
    LLMMistake,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMToolError,
    LLMProviderError,
)
from ...utils.aws import get_secret
from ..base import LLM
from ..types import (
    BaseAuthConfig,
    BaseAPIRequest,
    BaseGenerationOptions,
    BaseMessage,
    BaseUserMessage,
    BaseAssistantMessage,
    BaseToolDefinition,
    BaseFunctionDefinition,
    BaseToolCall,
    BaseUsageStatistics,
    Provider,
    ContentType,
    RoleType,
    ToolType,
    StandardMessageInput,
    StandardMessageOutput,
    ErrorDetails,
    BaseTextContent,
    BaseImageContent,
    BaseToolUseContent,
    BaseToolResultContent,
    BaseThinkingContent,
    BaseParameters,
    BaseParameterProperty,
)

# Define aliases for compatibility with internal used names


# Type definitions for Google API structures
class GoogleTextContent(TypedDict):
    """Google-specific text content part."""
    
    type: Literal["text"]
    text: str


class GoogleImageSource(TypedDict):
    """Google-specific image source structure."""
    
    url: Optional[str]
    base64: Optional[str]
    file_path: Optional[str]
    mime_type: Optional[str]


class GoogleImageContent(TypedDict):
    """Google-specific image content part."""
    
    type: Literal["image"]
    source: Union[str, GoogleImageSource]


class GoogleFunctionCall(TypedDict):
    """Google-specific function call data."""
    
    name: str
    args: Dict[str, Any]


class GoogleFunctionResponse(TypedDict):
    """Google-specific function response data."""
    
    name: str
    response: Dict[str, Any]


class GoogleUsageMetadata(TypedDict):
    """Google-specific usage metadata."""
    
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int


class GoogleAPIResponse(TypedDict):
    """Google-specific API response structure."""
    
    text: str
    usage_metadata: GoogleUsageMetadata
    function_calls: List[GoogleFunctionCall]


class GoogleToolUse(TypedDict):
    """Google-specific tool use data."""
    
    type: Literal["tool_use"]
    name: str
    input: Dict[str, Any]
    tool_id: Optional[str]


class GoogleToolResult(TypedDict):
    """Google-specific tool result data."""
    
    type: Literal["tool_result"]
    tool_id: str
    result: Dict[str, Any]


class GoogleFunctionDict(TypedDict):
    """Google-specific function definition."""
    
    name: str
    description: str
    parameters: Dict[str, Any]
    required: Optional[bool]


class GoogleToolUseData(TypedDict):
    """Google-specific tool call data."""
    
    name: str
    input: Dict[str, Any]
    id: Optional[str]
    type: Optional[ToolType]


class GoogleStreamingUsage(TypedDict):
    """Usage information during streaming for Google models."""

    is_complete: bool
    tool_use: Optional[Dict[str, Any]]
    read_tokens: int
    write_tokens: int
    total_tokens: int
    read_cost: float
    write_cost: float
    total_cost: float
    images: int
    image_cost: float
    provider: str
    model: str
    successful: bool
    error: Optional[str]
    error_type: Optional[str]
    retry_count: int
    event_id: str
    request_timestamp: Optional[float]
    response_timestamp: Optional[float]
    latency_ms: Optional[float]


class GoogleModelCosts(TypedDict):
    """Cost structure for a Google model."""

    read_token: float
    write_token: float
    image_cost: float


class GoogleParameterProperty(TypedDict, total=False):
    """Google-specific parameter property structure."""
    
    type: str
    description: Optional[str]
    enum: Optional[List[str]]
    format: Optional[str]
    minimum: Optional[int]
    maximum: Optional[int]
    default: Optional[Any]
    items: Optional[Dict[str, Any]]
    properties: Optional[Dict[str, Any]]
    required: Optional[List[str]]


class GoogleParameters(TypedDict, total=False):
    """Google-specific parameters structure."""
    
    type: str
    properties: Dict[str, GoogleParameterProperty]
    required: List[str]
    description: Optional[str]


class GoogleFunctionDefinition(TypedDict):
    """Google-specific function definition."""
    
    name: str
    description: str
    parameters: GoogleParameters
    required: Optional[bool]


class GoogleToolDefinition(TypedDict):
    """Google-specific tool definition."""
    
    name: str
    description: str
    input_schema: GoogleParameters
    required: Optional[bool]


class GoogleToolCallData(TypedDict):
    """Data structure for tool call information."""

    id: str
    name: str
    arguments: str
    type: Literal["function"]


# Filter out specific warnings
# Suppress the automatic_function_calling warning
logging.getLogger().addFilter(
    lambda record: "automatic_function_calling" not in record.getMessage()
)

# Suppress the Pydantic serializer warning by using a filter or warnings.filterwarnings
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


class GoogleLLM(LLM):
    """
    Implementation of Google's Gemini LLM models.

    This class provides methods for authentication, message formatting, image processing,
    and interaction with Google's Generative AI API. It converts between the standardized
    LLM handler message format and Google's content/part structure.

    Attributes:
        PROVIDER_ID: Provider identifier from the Provider enum
        SUPPORTED_MODELS: List of Gemini models supported by this implementation
        CONTEXT_WINDOW: Maximum token limits for each model
        SUPPORTS_IMAGES: Whether this provider supports image inputs
        COST_PER_MODEL: Cost information per token for each model
    """
    
    # Provider identifier for type-safe provider reference
    PROVIDER_ID = Provider.GOOGLE

    SUPPORTED_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-pro-exp-02-05",
        "gemini-2.0-flash-thinking-exp-01-21",
    ]

    CONTEXT_WINDOW = {
        "gemini-2.0-flash": 128000,
        "gemini-2.0-flash-lite-preview-02-05": 128000,
        "gemini-2.0-pro-exp-02-05": 128000,
        "gemini-2.0-flash-thinking-exp-01-21": 128000,
    }

    # All Google models support image input
    SUPPORTS_IMAGES = True

    # Base costs per token for each model (placeholder values, adjust as needed)
    COST_PER_MODEL = {
        "gemini-2.0-flash": {
            "read_token": 0.0000025,
            "write_token": 0.00001,
            "image_cost": 0.001,  # Fixed cost per image
        },
        "gemini-2.0-flash-lite-preview-02-05": {
            "read_token": 0.000001,
            "write_token": 0.000004,
            "image_cost": 0.0005,
        },
        "gemini-2.0-pro-exp-02-05": {
            "read_token": 0.000003,
            "write_token": 0.000012,
            "image_cost": 0.002,
        },
        "gemini-2.0-flash-thinking-exp-01-21": {
            "read_token": 0.000004,
            "write_token": 0.000016,
            "image_cost": 0.002,
        },
    }

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize a Google LLM instance.

        Args:
            model_name: Name of the Google model to use
            **kwargs: Additional arguments passed to the base LLM class
        """
        super().__init__(model_name, **kwargs)
        self.client = None
        self.timeout = kwargs.get("timeout", 60)
        self.api_base = kwargs.get("api_base", None)
        
        # Set provider identifier for consistent provider identification
        self.provider = self.PROVIDER_ID

    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """
        Validate Google provider configuration.
        
        Args:
            config: Provider configuration dictionary
            
        Raises:
            LLMLLMConfigurationError: If configuration is invalid
        """
        # Google-specific configuration validation
        if "api_base" in config and not isinstance(config["api_base"], (str, type(None))):
            raise LLMConfigurationError(
                message="api_base must be a string or None",
                provider=Provider.GOOGLE.value,
                details={"provided_type": type(config["api_base"]).__name__}
            )
            
        if "timeout" in config and not isinstance(config["timeout"], (int, float)):
            raise LLMConfigurationError(
                message="timeout must be a number",
                provider=Provider.GOOGLE.value,
                details={"provided_type": type(config["timeout"]).__name__}
            )
        
        if "service_account_file" in config and not isinstance(config["service_account_file"], (str, type(None))):
            raise LLMConfigurationError(
                message="service_account_file must be a string or None",
                provider=Provider.GOOGLE.value,
                details={"provided_type": type(config["service_account_file"]).__name__}
            )
        
        if "project_id" in config and not isinstance(config["project_id"], (str, type(None))):
            raise LLMConfigurationError(
                message="project_id must be a string or None",
                provider=Provider.GOOGLE.value,
                details={"provided_type": type(config["project_id"]).__name__}
            )

    def _initialize_client(self) -> None:
        """Initialize the Google API client."""
        try:
            genai.configure(api_key=self.api_key)
            self._client = genai
            
            # Test connection by accessing models (will raise if auth fails)
            test = self._client.list_models()
            if not test:
                raise LLMProviderError(
                    message="Failed to connect to Google API",
                    provider=self.PROVIDER_ID.value,
                    details={"error": "No models returned"},
                )
                
        except Exception as e:
            # Map the error to an appropriate exception type
            raise self._map_google_error(e)

    def auth(self) -> None:
        """
        Initialize the Google AI API client.

        Raises:
            LLMLLMAuthenticationError: If authentication with Google API fails
            LLMLLMConfigurationError: If the API key is missing or client initialization fails
        """
        try:
            self._initialize_client()
        except Exception as e:
            raise self._map_google_error(e)

    def _process_image(self, image_source: str, is_url: bool = False) -> Image.Image:
        """
        Process an image for the Google API.

        Loads an image from a local path or URL and converts it to a format
        compatible with Google's API.

        Args:
            image_source: Path or URL to the image
            is_url: Whether the source is a URL

        Returns:
            PIL.Image.Image: Processed image for the API

        Raises:
            LLMMistake: If image processing fails due to other reasons
        """
        try:
            if is_url:
                # For URLs, fetch the image content first
                try:
                    response = requests.get(image_source, timeout=10)
                    response.raise_for_status()
                except requests.RequestException as req_error:
                    # Detailed handling of different HTTP errors
                    status_code = getattr(req_error.response, "status_code", None)
                    if status_code == 404:
                        raise LLMMistake(
                            message=f"Image URL not found (404): {image_source}",
                            error_type="image_url_error",
                            provider="GoogleLLM",
                            details={
                                "source": image_source,
                                "status_code": 404,
                                "original_error": str(req_error),
                            },
                        )
                    elif status_code in (401, 403):
                        raise LLMMistake(
                            message=f"Access denied to image URL (unauthorized): {image_source}",
                            error_type="image_url_error",
                            provider="GoogleLLM",
                            details={
                                "source": image_source,
                                "status_code": status_code,
                                "original_error": str(req_error),
                            },
                        )
                    elif status_code and status_code >= 500:
                        raise LLMMistake(
                            message=f"Server error when fetching image URL: {image_source}",
                            error_type="image_url_error",
                            provider="GoogleLLM",
                            details={
                                "source": image_source,
                                "status_code": status_code,
                                "original_error": str(req_error),
                            },
                        )
                    else:
                        # General request exception
                        raise LLMMistake(
                            message=f"Failed to fetch image URL: {image_source}",
                            error_type="image_url_error",
                            provider="GoogleLLM",
                            details={
                                "source": image_source,
                                "original_error": str(req_error),
                            },
                        )

                # Convert response content to PIL Image
                try:
                    image_data = BytesIO(response.content)
                    image = Image.open(image_data)

                    # Validate image
                    image.verify()  # Verify that it's a valid image

                    # Reopen for processing (verify closes the file)
                    image_data.seek(0)
                    return Image.open(image_data)
                except Exception as img_error:
                    # Image format or content errors
                    raise LLMContentError(
                        message=f"Invalid or unsupported image format from URL: {image_source}",
                        provider="GoogleLLM",
                        details={
                            "source": image_source,
                            "original_error": str(img_error),
                        },
                    )
            else:
                # For local paths, validate the path first
                if not pathlib.Path(image_source).exists():
                    raise LLMMistake(
                        message=f"Image file not found: {image_source}",
                        error_type="image_processing_error",
                        provider="GoogleLLM",
                        details={"source": image_source},
                    )

                try:
                    # Attempt to open and validate the image
                    image = Image.open(image_source)

                    # Verify image is valid
                    image.verify()

                    # Reopen for processing (verify closes the file)
                    return Image.open(image_source)
                except Exception as img_error:
                    # Image format or content errors
                    raise LLMContentError(
                        message=f"Invalid or unsupported image format: {image_source}",
                        provider="GoogleLLM",
                        details={
                            "source": image_source,
                            "original_error": str(img_error),
                        },
                    )

        except (LLMMistake, LLMContentError):
            # Re-raise our custom exceptions directly
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            error_type = "image_url_error" if is_url else "image_processing_error"
            raise LLMMistake(
                message=f"Failed to process image {'URL' if is_url else 'file'} {image_source}: {e!s}",
                error_type=error_type,
                provider="GoogleLLM",
                details={"source": image_source, "original_error": str(e)},
            )

    def _format_messages_for_model(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """
        Format standard messages into Google's API format.

        Args:
            messages: List of standardized message dictionaries

        Returns:
            List of formatted messages ready for Google's API
        """
        from google.genai import types

        # Convert general Dict to more specific TypedDict for internal use
        typed_messages: List[Dict[str, Any]] = []

        for message in messages:
            typed_message: Dict[str, Any] = {
                "message_type": message.get("message_type", ""),
                "message": message.get("message", ""),
                "image_paths": message.get("image_paths", []),
                "image_urls": message.get("image_urls", []),
                "tool_use": message.get("tool_use", None),
                "tool_result": message.get("tool_result", None),
                "thinking": message.get("thinking", None),
                "name": message.get("name", None),
            }
            typed_messages.append(typed_message)

        formatted_contents = []

        for message in typed_messages:
            # Create a new Content object for this message
            current_content = types.Content()

            # Convert standard message_type to Google's role system:
            # - "ai" -> "model"  (AI responses)
            # - "tool" -> "tool" (tool/function responses)
            # - any other value -> "user" (user inputs)
            if message.get("message_type") == "ai":
                current_content.role = "model"
            elif message.get("message_type") == "tool":
                current_content.role = "tool"
            else:
                current_content.role = "user"

            parts = []

            # Add text content if present
            if message.get("message"):
                parts.append(types.Part.from_text(text=message["message"]))

            # Process local image files
            image_paths = message.get("image_paths", [])
            if image_paths:
                for image_path in image_paths:
                    try:
                        image_part = self._process_image(image_path)
                        parts.append(image_part)
                    except LLMMistake:
                        # Re-raise LLMMistake exceptions
                        raise
                    except Exception as e:
                        # Convert to LLMMistake if not already

                        raise LLMMistake(
                            message=f"Failed to process image file {image_path}: {e!s}",
                            error_type="image_processing_error",
                            provider="GoogleLLM",
                            details={"path": image_path, "original_error": str(e)},
                        )

            # Process image URLs
            image_urls = message.get("image_urls", [])
            if image_urls:
                for image_url in image_urls:
                    try:
                        image_part = self._process_image(image_url, is_url=True)
                        parts.append(image_part)
                    except LLMMistake:
                        # Re-raise LLMMistake exceptions
                        raise
                    except Exception as e:
                        # Convert to LLMMistake if not already

                        raise LLMMistake(
                            message=f"Failed to process image URL {image_url}: {e!s}",
                            error_type="image_url_error",
                            provider="GoogleLLM",
                            details={"url": image_url, "original_error": str(e)},
                        )

            # Add function call information (outgoing tool use)
            tool_use = message.get("tool_use")
            if tool_use:
                # Convert tool_use to Google's function_call format
                parts.append(
                    types.Part.from_function_call(
                        name=tool_use["name"],
                        args=tool_use["input"],
                    )
                )

            # Add function response information (incoming tool results)
            tool_result = message.get("tool_result")
            if tool_result:
                name = tool_result["tool_id"]
                # Format response based on success or failure
                if tool_result.get("success", False):
                    function_response = {"result": tool_result["result"]}
                else:
                    function_response = {
                        "error": tool_result.get("error", "Unknown error")
                    }

                parts.append(
                    types.Part.from_function_response(
                        name=name, response=function_response
                    )
                )

            # Assign all parts to the current content
            current_content.parts = parts

            # Add the completed content to our formatted list
            formatted_contents.append(current_content)

        return formatted_contents

    def _map_google_error(self, error: Exception) -> Exception:
        """
        Map Google API errors to our standard exception types.

        Args:
            error: Original exception from Google API

        Returns:
            Mapped exception
        """
        error_str = str(error).lower()

        # Rate limit or quota exceeded
        if any(term in error_str for term in ["quota", "rate limit", "too many requests", "resource exhausted"]):
            return LLMRateLimitError(
                message=f"Google API rate limit exceeded: {str(error)}",
                provider=self.PROVIDER_ID.value,
                details={"error": str(error)},
            )

        # Authentication errors
        if any(term in error_str for term in ["unauthorized", "authentication", "permission", "credentials"]):
            return LLMAuthenticationError(
                message=f"Google API authentication error: {str(error)}",
                provider=self.PROVIDER_ID.value,
                details={"error": str(error)},
            )

        # Service unavailable
        if any(term in error_str for term in ["unavailable", "server error", "timeout", "connection"]):
            return LLMServiceUnavailableError(
                message=f"Google API service error: {str(error)}",
                provider=self.PROVIDER_ID.value,
                details={"error": str(error)},
            )

        # Content filtering
        if any(term in error_str for term in ["content filter", "safety", "harmful", "blocked"]):
            return LLMContentError(
                message=f"Content blocked by Google safety filters: {str(error)}",
                provider=self.PROVIDER_ID.value,
                details={"error": str(error)},
            )

        # Configuration errors
        if any(term in error_str for term in ["invalid", "parameter", "configuration"]):
            return LLMConfigurationError(
                message=f"Invalid configuration or parameters: {str(error)}",
                provider=self.PROVIDER_ID.value,
                details={"error": str(error)},
            )

        # Default to provider error
        return LLMProviderError(
            message=f"Google API error: {str(error)}",
            provider=self.PROVIDER_ID.value,
            details={"error": str(error)},
        )

    def _call_with_retry(self, func, *args, max_retries=3, retry_delay=1, **kwargs):
        """
        Execute an API call with automatic retry for transient errors.

        This method implements an exponential backoff retry mechanism for
        handling transient errors from the Google API, such as rate limiting
        or temporary service unavailability.

        Args:
            func: Function to call
            *args: Positional arguments to pass to the function
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (in seconds)
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Any: Result of the function call

        Raises:
            Exception: Re-raises the last exception after all retry attempts fail
        """
        import random
        import time

        # Using LLMRateLimitError and LLMServiceUnavailableError from top imports

        attempts = 0
        last_error = None

        while attempts <= max_retries:
            try:
                return func(*args, **kwargs)
            except (LLMRateLimitError, LLMServiceUnavailableError) as e:
                last_error = e
                attempts += 1

                if attempts > max_retries:
                    # Re-raise the exception after max retries
                    raise

                # Determine retry delay with exponential backoff and jitter
                # If the error provides a retry_after value, use that
                if isinstance(e, LLMRateLimitError) and e.retry_after:
                    delay = e.retry_after
                else:
                    # Otherwise, use exponential backoff with jitter
                    delay = min(
                        retry_delay * (2 ** (attempts - 1)), 60
                    )  # Cap at 60 seconds
                    # Add jitter to avoid thundering herd
                    delay = delay * (0.5 + random.random())

                # Wait before retrying
                time.sleep(delay)
            except Exception:
                # For all other exceptions, don't retry
                raise

        # If we reach here, all retries failed
        raise last_error

    def _raw_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        top_k: int = 200,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking_budget: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response using Google's API.

        This method handles:
        1. Authentication if needed
        2. Message formatting using Google's content structure
        3. Configuration of generation parameters
        4. Making the API request
        5. Extracting and processing the response
        6. Calculating token usage and costs
        7. Extracting any function/tool calls from the response

        Args:
            event_id: Unique identifier for this generation event
            system_prompt: System-level instructions for the model
            messages: List of messages in the standard format
            max_tokens: Maximum number of tokens to generate
            temp: Temperature for generation (0.0 = deterministic)
            top_k: Top K tokens to consider (not used in Google)
            tools: Optional list of function/tool definitions
            thinking_budget: Optional budget for thinking (not used in Google)

        Returns:
            Tuple containing:
            - Generated text response
            - Usage statistics (tokens, costs, tool use)

        Raises:
            LLMMistake: If the API call fails with a recoverable error
            LLMAuthenticationError: If authentication fails
            LLMRateLimitError: If rate limits are exceeded
            LLMServiceUnavailableError: If the service is unavailable
        """
        # Convert tools to GoogleFunctionDict if needed
        google_tools = None
        if tools is not None:
            google_tools = []
            for tool in tools:
                # Create GoogleFunctionDict from the tool
                google_tool: GoogleFunctionDict = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                }
                google_tools.append(google_tool)
        from ...exceptions import (
            LLMAuthenticationError,
            LLMConfigurationError,
            LLMMistake,
            LLMRateLimitError,
            LLMServiceUnavailableError,
        )

        try:
            # Ensure we're authenticated
            if self.client is None:
                try:
                    self.auth()
                except Exception as e:
                    # Map any authentication errors through our error mapping system
                    raise self._map_google_error(e)

            # Format messages for Google's API
            try:
                formatted_contents = self._format_messages_for_model(messages)
            except LLMMistake:
                # Re-raise LLMMistake exceptions directly
                raise
            except Exception as e:
                # Map other message formatting errors
                raise self._map_google_error(e)

            # Create generation config with appropriate parameters
            try:
                if google_tools:
                    # If tools are provided, configure them in the request
                    # Setting automatic_function_calling to disable prevents the model
                    # from automatically choosing a function without explicit instructions
                    config = types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temp,
                        tools=google_tools,
                        automatic_function_calling={"disable": True},
                    )
                else:
                    # Basic config without tools
                    config = types.GenerateContentConfig(
                        max_output_tokens=max_tokens, temperature=temp
                    )

                # Add system instruction if provided
                if system_prompt:
                    config.system_instruction = system_prompt
            except Exception as e:
                # Configuration errors
                raise LLMConfigurationError(
                    message=f"Failed to configure Google API request: {e!s}",
                    provider="GoogleLLM",
                    details={"original_error": str(e)},
                )

            # Make the API request to generate content with retry logic
            def api_call():
                return self.client.models.generate_content(
                    model=self.model_name, contents=formatted_contents, config=config
                )

            try:
                response = self._call_with_retry(api_call, max_retries=3, retry_delay=1)
            except Exception as e:
                # Map API call errors through our error mapping system
                if isinstance(
                    e,
                    (
                        LLMAuthenticationError,
                        LLMRateLimitError,
                        LLMServiceUnavailableError,
                        LLMMistake,
                    ),
                ):
                    # Re-raise error types that are already properly mapped
                    raise
                else:
                    # Map other errors
                    raise self._map_google_error(e)

            # Extract usage information from response metadata
            # Note: For some API versions, these fields may not be available
            try:
                usage_metadata = getattr(response, "usage_metadata", None) or {}
                input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = (
                    getattr(usage_metadata, "candidates_token_count", 0) or 0
                )
                total_tokens = getattr(usage_metadata, "total_token_count", None) or (
                    input_tokens + output_tokens
                )
            except Exception:
                # Default to zero counts if usage metadata extraction fails
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0

            # Extract the text response, handling potential errors gracefully
            try:
                response_text = getattr(response, "text", None) or ""
            except Exception as e:
                # Handle missing text in response as a content error

                raise LLMContentError(
                    message=f"Failed to extract text from Google API response: {e!s}",
                    provider="GoogleLLM",
                    details={"original_error": str(e)},
                )

            # Count how many images were included in the request
            num_images = sum(
                len(msg.get("image_paths", [])) + len(msg.get("image_urls", []))
                for msg in messages
            )

            # Calculate costs based on token usage and our cost model
            costs = self.get_model_costs()
            read_token_cost = costs.get("read_token", 0.0) or 0.0
            write_token_cost = costs.get("write_token", 0.0) or 0.0
            image_token_cost = costs.get("image_cost", 0.0) or 0.0

            read_cost = float(input_tokens) * read_token_cost
            write_cost = float(output_tokens) * write_token_cost
            image_cost = float(num_images) * image_token_cost

            # Prepare complete usage statistics
            usage_stats: GoogleUsageStatsDict = {
                "read_tokens": input_tokens,
                "write_tokens": output_tokens,
                "images": num_images,
                "total_tokens": total_tokens,
                "read_cost": round(read_cost, 6),
                "write_cost": round(write_cost, 6),
                "image_cost": round(image_cost, 6),
                "total_cost": round(read_cost + write_cost + image_cost, 6),
                "tool_use": None,
            }

            # Extract any function/tool calls from the response
            try:
                tool_use = getattr(response, "function_calls", None) or []
                if tool_use:
                    # Format tool use information in a standardized way
                    tool_use_data: GoogleToolCallData = {
                        "id": tool_use[0].id,
                        "name": tool_use[0].name,
                        "input": tool_use[0].args,
                    }
                    usage_stats["tool_use"] = tool_use_data
            except Exception:
                # No need to raise an exception for tool extraction failures
                # Log the error and continue
                pass

            # Convert to Dict[str, Any] for base class compatibility
            return response_text, dict(usage_stats)

        except (
            LLMAuthenticationError,
            LLMConfigurationError,
            LLMRateLimitError,
            LLMServiceUnavailableError,
            LLMMistake,
        ) as e:
            # Known exceptions, just re-raise
            raise
        except Exception as e:
            # Catch-all for any other exceptions
            # Map to appropriate exception type
            mapped_error = self._map_google_error(e)
            raise mapped_error

    def supports_image_input(self) -> bool:
        """
        Check if this model supports image inputs.

        Returns:
            bool: True if the model supports images, False otherwise
        """
        return self.SUPPORTS_IMAGES

    def get_supported_models(self) -> list[str]:
        """
        Get the list of supported model identifiers.

        Returns:
            list[str]: List of supported model identifiers
        """
        return self.SUPPORTED_MODELS

    def get_model_costs(self) -> Dict[str, Union[float, None]]:
        """
        Get the cost information for the current model.

        Returns:
            Dict[str, Union[float, None]]: Dictionary containing 'read_token', 'write_token', and 'image_cost'
        """
        if self.model_name in self.COST_PER_MODEL:
            return self.COST_PER_MODEL[self.model_name]
        else:
            # Default to a reasonably priced model if exact model is not found
            default_costs: Dict[str, Union[float, None]] = {
                "read_token": 0.000003,
                "write_token": 0.000006,
                "image_cost": 0.001,
            }
            return default_costs

    def stream_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        functions: Optional[List[Callable]] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Stream a response from the Google Gemini API, yielding chunks as they become available.

        Args:
            event_id (str): Unique identifier for this generation event
            system_prompt (str): System instructions for the model
            messages (List[Dict[str, Any]]): List of messages in the standard format
            max_tokens (int): Maximum number of tokens to generate
            temp (float): Temperature for generation
            functions (List[Callable]): List of functions the LLM can use
            callback (Callable): Optional callback function for each chunk

        Yields:
            Tuple[str, Dict[str, Any]]: Tuples of (text_chunk, partial_usage_data)

        Raises:
            LLMMistake: If the API call fails with a recoverable error
            LLMAuthenticationError: If authentication fails
            LLMRateLimitError: If rate limits are exceeded
            LLMServiceUnavailableError: If the service is unavailable
            ImportError: If the Google Generative AI package is not installed
        """
        # Convert to more specific typed messages for internal use
        typed_messages: List[Dict[str, Any]] = []
        for msg in messages:
            typed_message: Dict[str, Any] = {
                "message_type": msg.get("message_type", "human"),
                "message": msg.get("message"),
                "image_paths": msg.get("image_paths", []),
                "image_urls": msg.get("image_urls", []),
                "tool_use": msg.get("tool_use"),
                "tool_result": msg.get("tool_result"),
            }
            typed_messages.append(typed_message)
        # Import required exceptions
        from ...exceptions import (
            LLMAuthenticationError,
            LLMConfigurationError,
            LLMMistake,
            LLMRateLimitError,
            LLMServiceUnavailableError,
        )

        # Import Google Generative AI here to avoid import errors if not installed
        try:
            import google.generativeai as genai
            from google.generativeai.types import content_types, generation_types
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. Install with 'pip install google-generativeai'"
            )

        # Initialize client if needed
        if not hasattr(self, "genai"):
            try:
                self.auth()
            except Exception as e:
                # Map any authentication errors through our error mapping system
                raise self._map_google_error(e)

        # Convert messages to Google format
        try:
            # Convert typed_messages back to Dict[str, Any] for method compatibility
            messages_for_model: List[Dict[str, Any]] = [
                dict(msg) for msg in typed_messages
            ]
            formatted_messages = self._format_messages_for_model(messages_for_model)
        except LLMMistake:
            # Re-raise LLMMistake exceptions directly
            raise
        except Exception as e:
            # Map other message formatting errors
            raise self._map_google_error(e)

        # Prepare function declarations if functions are provided
        try:
            function_declarations = None
            if functions:
                function_declarations = []
                for function in functions:
                    # Parse function signature and create declaration
                    name = getattr(function, "__name__", str(function))
                    docstring = getattr(function, "__doc__", "")

                    # Simplified schema - in a real implementation, you'd parse the function signature
                    function_declarations.append(
                        {
                            "name": name,
                            "description": docstring or f"Call {name} function",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                            },
                        }
                    )
        except Exception as e:
            raise LLMToolError(
                message=f"Failed to prepare function declarations for Google streaming: {e!s}",
                provider=self.PROVIDER_ID,
                details={"original_error": str(e)},
            )

        # Count images for cost calculation
        image_count = 0
        for message in typed_messages:
            if message.get("message_type") == "human":
                image_paths = message.get("image_paths", [])
                if image_paths:
                    image_count += len(image_paths)

                image_urls = message.get("image_urls", [])
                if image_urls:
                    image_count += len(image_urls)

        # Prepare system instruction if provided
        try:
            if system_prompt:
                # For Google, we add the system prompt as a user message at the beginning
                system_content = content_types.Content(
                    role="user", parts=[content_types.Part.from_text(system_prompt)]
                )
                formatted_messages.insert(0, system_content)
        except Exception as e:
            raise LLMConfigurationError(
                message=f"Failed to configure system prompt for Google streaming: {e!s}",
                provider=self.PROVIDER_ID,
                details={"original_error": str(e)},
            )

        # Define the streaming function to be called with retries
        def stream_request():
            # Create the model
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temp,
                },
                tools=function_declarations,
                safety_settings=(
                    self.safety_settings if hasattr(self, "safety_settings") else None
                ),
            )

            # Create a streaming request
            return model.generate_content(
                formatted_messages, stream=True  # Enable streaming
            )

        try:
            # Initialize response with retry logic
            response = self._call_with_retry(
                stream_request, max_retries=3, retry_delay=1
            )

            # Initialize variables to accumulate data
            accumulated_text = ""
            accumulated_tokens = 0
            tool_call_data: Dict[str, GoogleToolCallData] = {}

            # Rough estimation of input tokens
            input_tokens = 0
            for message in typed_messages:
                # Estimate based on message content
                message_text = message.get("message")
                if message_text:
                    # Rough estimation: count words and multiply by 1.3
                    input_tokens += int(len(message_text.split()) * 1.3)

                # Add token estimates for images
                image_paths = message.get("image_paths", [])
                if image_paths:
                    input_tokens += len(image_paths) * 1000  # Rough estimate

                image_urls = message.get("image_urls", [])
                if image_urls:
                    input_tokens += len(image_urls) * 1000  # Rough estimate

            # Add system prompt tokens if present
            if system_prompt:
                input_tokens += int(len(system_prompt.split()) * 1.3)

            # Process each chunk as it arrives
            for chunk in response:
                # Check for text content
                if hasattr(chunk, "text"):
                    content = chunk.text
                    if content:
                        # For first chunk, clean up any system prompt echoing
                        if (
                            accumulated_text == ""
                            and system_prompt
                            and content.startswith(system_prompt)
                        ):
                            content = content[len(system_prompt) :].lstrip()

                        accumulated_text += content
                        # Rough token estimation: 1 token per character / 4
                        token_estimate = max(1, len(content) // 4)
                        accumulated_tokens += token_estimate

                        # Call the callback if provided
                        if callback:
                            # Create partial usage data
                            partial_usage: Dict[str, Any] = {
                                "event_id": event_id,
                                "model": self.model_name,
                                "read_tokens": input_tokens,
                                "write_tokens": accumulated_tokens,
                                "images": image_count,
                                "total_tokens": input_tokens + accumulated_tokens,
                                "is_complete": False,
                                "tool_use": tool_call_data,
                            }
                            callback(content, partial_usage)

                        # Yield the content chunk and partial usage data
                        partial_data: Dict[str, Any] = {
                            "event_id": event_id,
                            "model": self.model_name,
                            "read_tokens": input_tokens,
                            "write_tokens": accumulated_tokens,
                            "images": image_count,
                            "total_tokens": input_tokens + accumulated_tokens,
                            "is_complete": False,
                            "tool_use": tool_call_data,
                        }
                        yield content, partial_data

                # Check for function calls
                # Google API returns function calls at the end, not incrementally
                if hasattr(chunk, "candidates") and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, "content") and hasattr(
                        candidate.content, "parts"
                    ):
                        for part in candidate.content.parts:
                            if hasattr(part, "function_call"):
                                try:
                                    function_call = part.function_call
                                    tool_id = (
                                        f"func_{len(tool_call_data)}"
                                        # Generate an ID
                                    )

                                    tool_data: GoogleToolCallData = {
                                        "id": tool_id,
                                        "name": function_call.name,
                                        "arguments": str(
                                            function_call.args
                                        ),  # Convert args to string
                                        "type": "function",
                                    }
                                    tool_call_data[tool_id] = tool_data
                                except Exception:
                                    # Log but don't fail the streaming process
                                    pass

            # Calculate costs
            model_costs = self.get_model_costs()
            read_token_cost = model_costs.get("read_token", 0.0) or 0.0
            write_token_cost = model_costs.get("write_token", 0.0) or 0.0
            image_token_cost = model_costs.get("image_cost", 0.0) or 0.0

            read_cost = float(input_tokens) * read_token_cost
            write_cost = float(accumulated_tokens) * write_token_cost

            # Calculate image cost if applicable
            image_cost = 0.0
            if image_count > 0:
                image_cost = float(image_count) * image_token_cost

            total_cost = read_cost + write_cost + image_cost

            # Create final usage data
            final_usage: Dict[str, Any] = {
                "event_id": event_id,
                "model": self.model_name,
                "read_tokens": input_tokens,
                "write_tokens": accumulated_tokens,
                "images": image_count,
                "total_tokens": input_tokens + accumulated_tokens,
                "read_cost": round(read_cost, 6),
                "write_cost": round(write_cost, 6),
                "image_cost": round(image_cost, 6),
                "total_cost": round(total_cost, 6),
                "is_complete": True,
                "tool_use": tool_call_data,
            }

            # Call the callback with an empty string to signal completion
            if callback:
                callback("", final_usage)

            # Yield an empty string with the final usage data to signal completion
            yield "", final_usage

        except (
            LLMAuthenticationError,
            LLMConfigurationError,
            LLMRateLimitError,
            LLMServiceUnavailableError,
            LLMMistake,
        ) as e:
            # Known exceptions, just re-raise
            raise
        except Exception as e:
            # Map to appropriate exception type via the error mapper
            mapped_error = self._map_google_error(e)
            raise mapped_error

    def _catch_exceptions(self, func: Callable) -> Callable:
        """
        Decorator to catch exceptions from Google API calls and map them to our exception types.

        Args:
            func: Function to wrap

        Returns:
            Wrapped function
        """

        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except (
                LLMAuthenticationError,
                LLMConfigurationError,
                LLMRateLimitError,
                LLMServiceUnavailableError,
                LLMContentError,
                LLMProviderError,
                LLMToolError,
            ) as e:
                # Re-raise known exceptions
                raise e
            except Exception as e:
                # Map other exceptions
                mapped_error = self._map_google_error(e)
                raise mapped_error

        return wrapper

