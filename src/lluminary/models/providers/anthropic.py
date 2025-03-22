"""
Anthropic LLM provider implementation.

This module implements support for Anthropic's Claude AI models within the LLM handler framework.
It handles authentication, message formatting, image processing, and cost tracking for all Claude models.

## Anthropic Messaging System Overview

Anthropic's Claude models use a structured content format with specific requirements:

1. Messages are organized as "messages" with roles (user, assistant)
2. Each message contains "content" which is an array of content parts that can include:
   - Text (type: "text")
   - Images (type: "image")
   - Tool use (type: "tool_use")
   - Tool results (type: "tool_result")
   - Thinking process (type: "thinking", for supported models)

### Message Conversion Process
This implementation converts the standard LLM handler message format:
```
{
    "message_type": "human"|"ai"|"tool",
    "message": "text content",
    "image_paths": ["local/path/to/image.jpg"],
    "image_urls": ["https://example.com/image.jpg"],
    "tool_use": {"id": "id", "name": "func_name", "input": {...}},
    "tool_result": {"tool_id": "id", "success": bool, "result": any, "error": str},
    "thinking": {"thinking": "thought process", "thinking_signature": "signature"}
}
```

Into Anthropic's format:
```
{
    "role": "user"|"assistant",
    "content": [
        {
            "type": "text",
            "text": "text content"
        },
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "base64_encoded_image"
            }
        },
        {
            "type": "tool_use",
            "id": "id",
            "name": "func_name",
            "input": {...}
        },
        {
            "type": "tool_result",
            "tool_use_id": "id",
            "content": "Tool Call Successful: result" | "Tool Call Failed: error"
        },
        {
            "type": "thinking",
            "thinking": "thought process",
            "signature": "signature"
        }
    ]
}
```

### Key Features
- Full support for text, images, and function calling
- Special support for Claude 3.7 "thinking" capabilities
- JPEG conversion with proper handling of transparency
- Provides token usage and cost tracking
"""

import base64
import json
import os
import time
from io import BytesIO
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import requests
from PIL import Image

from ...exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContentError,
    LLMError,
    LLMFormatError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMThinkingError,
    LLMToolError,
)
from ...utils.aws import get_secret
from ..base import LLM
from ..types import (
    # Base content types
    BaseTextContent,
    BaseImageContent,
    BaseImageSource,
    BaseToolUseContent,
    BaseToolResultContent,
    BaseThinkingContent,
    ContentType,
    
    # Message types
    BaseMessage,
    BaseUserMessage,
    BaseAssistantMessage,
    RoleType,
    MessageList,
    
    # Tool and function types
    BaseParameterProperty,
    BaseParameters,
    BaseToolDefinition,
    BaseFunctionDefinition,
    BaseToolCall,
    BaseToolCallData,
    
    # Provider identification
    Provider,
)


# Anthropic-specific type definitions using the base types as reference
class AnthropicImageSource(TypedDict):
    """Anthropic-specific image source structure."""
    
    type: Literal["base64"]
    media_type: str
    data: str


class AnthropicTextContent(TypedDict):
    """Anthropic-specific text content part."""
    
    type: Literal["text"]
    text: str


class AnthropicImageContent(TypedDict):
    """Anthropic-specific image content part."""
    
    type: Literal["image"]
    source: AnthropicImageSource


class AnthropicToolUseContent(TypedDict):
    """Anthropic-specific tool use content part."""
    
    type: Literal["tool_use"]
    id: str  # Anthropic uses 'id' instead of 'tool_id'
    name: str
    input: Dict[str, Any]


class AnthropicToolResultContent(TypedDict):
    """Anthropic-specific tool result content part."""
    
    type: Literal["tool_result"]
    tool_use_id: str  # Anthropic uses 'tool_use_id' instead of 'tool_id'
    content: str  # Anthropic uses 'content' instead of 'result'


class AnthropicThinkingContent(TypedDict):
    """Anthropic-specific thinking content part."""
    
    type: Literal["thinking"]
    thinking: str
    signature: str  # Anthropic requires a signature for thinking content


# Union type for all Anthropic content part types
AnthropicContentPart = Union[
    AnthropicTextContent,
    AnthropicImageContent,
    AnthropicToolUseContent,
    AnthropicToolResultContent,
    AnthropicThinkingContent,
]


class AnthropicMessage(TypedDict):
    """Anthropic-specific message structure."""
    
    role: Literal["user", "assistant"]
    content: List[AnthropicContentPart]


# Tool-related Anthropic-specific type definitions
class AnthropicParameterProperties(TypedDict, total=False):
    """Anthropic-specific parameter properties."""
    
    type: str
    description: Optional[str]
    enum: Optional[List[str]]
    items: Optional[Dict[str, Any]]
    properties: Optional[Dict[str, Any]]
    required: Optional[List[str]]


class AnthropicParameters(TypedDict):
    """Anthropic-specific parameters structure."""
    
    type: str
    properties: Dict[str, AnthropicParameterProperties]
    required: Optional[List[str]]


class AnthropicFunctionDefinition(TypedDict):
    """Anthropic-specific function definition."""
    
    name: str
    description: str
    parameters: AnthropicParameters


class AnthropicTool(TypedDict):
    """Anthropic-specific tool definition."""
    
    type: Literal["function"]
    function: AnthropicFunctionDefinition


class ToolCallData(TypedDict):
    """Anthropic-specific tool call data for streaming."""
    
    id: str
    name: str
    arguments: str  # Anthropic uses 'arguments' instead of 'input'
    type: Literal["function"]


class AnthropicLLM(LLM):
    """
    Implementation of Anthropic's Claude LLM models.

    This class provides methods for authentication, message formatting, image processing,
    and interaction with Anthropic's API. It converts between the standardized
    LLM handler message format and Anthropic's message/content structure.

    Attributes:
        THINKING_MODELS: List of models supporting the "thinking" capability
        SUPPORTED_MODELS: List of Claude models supported by this implementation
        CONTEXT_WINDOW: Maximum token limits for each model
        COST_PER_MODEL: Cost information per token for each model
    """

    THINKING_MODELS = [
        "claude-3-7-sonnet-20250219",
        "claude-3-7-opus-20250213",
        # Add any new models that support thinking as they're released
    ]

    SUPPORTED_MODELS = [
        # Claude 3.5 models
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-opus-20241022",
        # Claude 3.7 models
        "claude-3-7-sonnet-20250219",
        "claude-3-7-opus-20250213",
        # Add new models as they become available
    ]

    CONTEXT_WINDOW = {
        # Claude 3.5 models
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-opus-20241022": 200000,
        # Claude 3.7 models
        "claude-3-7-sonnet-20250219": 200000,
        "claude-3-7-opus-20250213": 200000,
    }

    COST_PER_MODEL = {
        # Claude 3.5 models
        "claude-3-5-haiku-20241022": {
            "read_token": 0.000001,
            "write_token": 0.000005,
            "image_cost": None,
        },
        "claude-3-5-sonnet-20241022": {
            "read_token": 0.000003,
            "write_token": 0.000015,
            "image_cost": 0.024,
        },
        "claude-3-5-opus-20241022": {
            "read_token": 0.000015,
            "write_token": 0.000075,
            "image_cost": 0.024,
        },
        # Claude 3.7 models
        "claude-3-7-sonnet-20250219": {
            "read_token": 0.000003,
            "write_token": 0.000015,
            "image_cost": 0.024,
        },
        "claude-3-7-opus-20250213": {
            "read_token": 0.000015,
            "write_token": 0.000075,
            "image_cost": 0.024,
        },
    }

    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Initialize an Anthropic LLM instance.

        Args:
            model_name: Name of the Anthropic model to use
            **kwargs: Additional arguments passed to the base LLM class
        """
        super().__init__(model_name, **kwargs)

        # Initialize API key directly if provided in kwargs
        if "api_key" in kwargs:
            self.config["api_key"] = kwargs["api_key"]
            
        # Set provider identifier
        self.provider = Provider.ANTHROPIC
        
    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """
        Validate Anthropic provider configuration.
        
        Args:
            config: Provider configuration dictionary
            
        Raises:
            LLMConfigurationError: If configuration is invalid
        """
        # No specific validation required for initialization
        # Authentication will be handled in the auth method
        pass

    def auth(self) -> None:
        """
        Authenticate with Anthropic using API key from AWS Secrets Manager or direct parameter.

        This method:
        1. Uses API key from config if already provided in constructor
        2. Tries to get the API key from AWS Secrets Manager if not provided
        3. Falls back to environment variables

        Raises:
            LLMAuthenticationError: If authentication fails and no valid API key is found
        """
        # If API key is already set in config, we're already authenticated
        if self.config.get("api_key"):
            return

        try:
            try:
                # Try to get the API key from Secrets Manager
                secret = get_secret("anthropic_api_key", required_keys=["api_key"])
                self.config["api_key"] = secret["api_key"]
            except Exception as secret_error:
                # If Secrets Manager fails, check environment variables
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    # Re-raise using proper error type with context
                    raise LLMAuthenticationError(
                        message="Failed to get API key for Anthropic",
                        provider="anthropic",
                        details={
                            "secret_error": str(secret_error),
                            "tried_sources": [
                                "AWS Secrets Manager",
                                "environment variables",
                            ],
                        },
                    )
                self.config["api_key"] = api_key

            # Verify the API key by making a test call
            try:
                # Simple validation with a HEAD request to the models endpoint
                response = requests.head(
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": self.config["api_key"],
                        "anthropic-version": "2023-06-01",  # This is the base version
                        "anthropic-beta": "tools-2023-12-13",  # Enable latest tool features
                    },
                )

                if response.status_code == 401:
                    raise LLMAuthenticationError(
                        message="Invalid Anthropic API key",
                        provider="anthropic",
                        details={"status_code": response.status_code},
                    )
                elif response.status_code != 200:
                    raise LLMProviderError(
                        message=f"Anthropic API returned status code {response.status_code} during authentication check",
                        provider="anthropic",
                        details={"status_code": response.status_code},
                    )

            except requests.RequestException as request_error:
                # Network errors during validation
                raise LLMProviderError(
                    message="Failed to validate Anthropic API key: Network error",
                    provider="anthropic",
                    details={"error": str(request_error)},
                )

        except LLMError:
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise LLMAuthenticationError(
                message="Anthropic authentication failed",
                provider="anthropic",
                details={"error": str(e)},
            )

    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64 string.

        This method:
        1. Opens the image file
        2. Converts to RGB format (handling transparency by adding white background)
        3. Saves as JPEG with 95% quality
        4. Encodes as base64

        Args:
            image_path: Path to the image file

        Returns:
            str: Base64 encoded image string

        Raises:
            LLMProviderError: If image file doesn't exist or can't be read
            LLMFormatError: If image format is invalid or can't be processed
        """
        try:
            # Check if the file exists
            if not os.path.exists(image_path):
                raise LLMProviderError(
                    message=f"Image file not found: {image_path}",
                    provider="anthropic",
                    details={"image_path": image_path},
                )

            # Process the image
            try:
                with Image.open(image_path) as img:
                    # If the image has an alpha channel (RGBA), convert to RGB
                    if img.mode == "RGBA":
                        # Create a white background image
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        # Paste the image on the background using the alpha channel as mask
                        background.paste(img, mask=img.split()[3])
                        # Use the background image for further processing
                        processed_img = background
                    elif img.mode != "RGB":
                        # Convert any other mode to RGB
                        processed_img = img.convert("RGB")
                    else:
                        # Already in RGB mode
                        processed_img = img.copy()

                    # Save as JPEG in memory with 95% quality (Anthropic's recommendation)
                    output = BytesIO()
                    processed_img.save(output, format="JPEG", quality=95)
                    return base64.b64encode(output.getvalue()).decode("utf-8")
            except OSError as io_error:
                # File read errors
                raise LLMProviderError(
                    message=f"Failed to read image file: {image_path}",
                    provider="anthropic",
                    details={"error": str(io_error), "image_path": image_path},
                )
            except Exception as format_error:
                # Image format/processing errors
                raise LLMFormatError(
                    message=f"Failed to process image: {image_path}",
                    provider="anthropic",
                    details={"error": str(format_error), "image_path": image_path},
                )

        except LLMError:
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise LLMProviderError(
                message=f"Unexpected error processing image: {image_path}",
                provider="anthropic",
                details={"error": str(e), "image_path": image_path},
            )

    def _download_image_from_url(self, image_url: str) -> str:
        """
        Download an image from a URL and convert it to a base64-encoded JPEG.

        Args:
            image_url (str): URL of the image to download

        Returns:
            str: Base64-encoded JPEG data

        Raises:
            LLMProviderError: If image download or processing fails
        """
        try:
            # Download the image
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()  # Raise an exception for HTTP errors
            except requests.RequestException as request_error:
                raise LLMProviderError(
                    message=f"Failed to download image from {image_url}",
                    provider="anthropic",
                    details={"error": str(request_error), "url": image_url},
                )

            # Process the image
            try:
                with Image.open(BytesIO(response.content)) as img:
                    # Handle transparency (RGBA mode) by adding white background
                    if img.mode == "RGBA":
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        processed_img = background
                    elif img.mode != "RGB":
                        # Convert any other mode to RGB
                        processed_img = img.convert("RGB")
                    else:
                        # Already in RGB mode
                        processed_img = img.copy()

                    # Save as JPEG in memory with 95% quality (Anthropic's recommendation)
                    output = BytesIO()
                    processed_img.save(output, format="JPEG", quality=95)
                    return base64.b64encode(output.getvalue()).decode("utf-8")
            except Exception as e:
                # Image processing errors
                raise LLMFormatError(
                    message=f"Failed to process image from URL: {image_url}",
                    provider="anthropic",
                    details={"error": str(e), "url": image_url},
                )

        except LLMError:
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise LLMProviderError(
                message=f"Unexpected error processing image from URL: {image_url}",
                provider="anthropic",
                details={"error": str(e), "url": image_url},
            )

    def _format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert standard message format into Anthropic's format.

        This method transforms the unified message format used across providers into
        Anthropic's specific structure with roles (user/assistant) and content parts.
        Each message is converted to an object with:
        - role: either "user" or "assistant"
        - content: array of content parts (text, image, tool_use, tool_result, thinking)

        Args:
            messages: List of messages in standard format

        Returns:
            List[Dict[str, Any]]: Messages formatted for Anthropic API
            (These will conform to the AnthropicMessage structure internally)

        Raises:
            LLMFormatError: If message formatting fails due to invalid structure
            LLMProviderError: If image processing fails
        """
        try:
            formatted_messages = []

            for msg_index, msg in enumerate(messages):
                try:
                    # Check for required message_type field
                    if "message_type" not in msg:
                        raise LLMFormatError(
                            message=f"Missing required field 'message_type' in message at index {msg_index}",
                            provider="anthropic",
                            details={"message": msg},
                        )

                    # Map message type to Anthropic role:
                    # - "ai" -> "assistant" (AI responses)
                    # - any other value -> "user" (user inputs, tools)
                    role = "assistant" if msg["message_type"] == "ai" else "user"
                    content_parts = []

                    # Process image files from local paths
                    if msg.get("image_paths"):
                        for image_path in msg["image_paths"]:
                            # _encode_image will raise appropriate errors if needed
                            base64_image = self._encode_image(image_path)
                            content_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image,
                                    },
                                }
                            )

                    # Process images from URLs
                    if msg.get("image_urls"):
                        for image_url in msg["image_urls"]:
                            # _download_image_from_url will raise appropriate errors if needed
                            base64_image = self._download_image_from_url(image_url)
                            content_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": base64_image,
                                    },
                                }
                            )

                    # Add thinking information (Claude 3.7 specific)
                    if msg.get("thinking"):
                        if (
                            not isinstance(msg["thinking"], dict)
                            or "thinking" not in msg["thinking"]
                            or "thinking_signature" not in msg["thinking"]
                        ):
                            raise LLMFormatError(
                                message="Invalid 'thinking' structure in message",
                                provider="anthropic",
                                details={"thinking": msg.get("thinking")},
                            )

                        content_parts.append(
                            {
                                "type": "thinking",
                                "thinking": msg["thinking"]["thinking"],
                                "signature": msg["thinking"]["thinking_signature"],
                            }
                        )

                    # Add text content if present
                    if msg.get("message"):
                        if not isinstance(msg["message"], str):
                            raise LLMFormatError(
                                message="Message content must be a string",
                                provider="anthropic",
                                details={"message": msg["message"]},
                            )

                        content_parts.append({"type": "text", "text": msg["message"]})

                    # Add tool use information (function calls)
                    if msg.get("tool_use"):
                        if (
                            not isinstance(msg["tool_use"], dict)
                            or "id" not in msg["tool_use"]
                            or "name" not in msg["tool_use"]
                            or "input" not in msg["tool_use"]
                        ):
                            raise LLMFormatError(
                                message="Invalid 'tool_use' structure in message",
                                provider="anthropic",
                                details={"tool_use": msg.get("tool_use")},
                            )

                        content_parts.append(
                            {
                                "type": "tool_use",
                                "id": msg["tool_use"]["id"],
                                "name": msg["tool_use"]["name"],
                                "input": msg["tool_use"]["input"],
                            }
                        )

                    # Add tool result information (function responses)
                    if msg.get("tool_result"):
                        if (
                            not isinstance(msg["tool_result"], dict)
                            or "tool_id" not in msg["tool_result"]
                        ):
                            raise LLMFormatError(
                                message="Invalid 'tool_result' structure in message",
                                provider="anthropic",
                                details={"tool_result": msg.get("tool_result")},
                            )

                        # Format the result string based on success or failure
                        result = ""
                        if msg["tool_result"].get("success"):
                            if "result" not in msg["tool_result"]:
                                raise LLMFormatError(
                                    message="Missing 'result' field in successful tool_result",
                                    provider="anthropic",
                                    details={"tool_result": msg["tool_result"]},
                                )
                            result = "Tool Call Successful: " + str(
                                msg["tool_result"]["result"]
                            )
                        else:
                            if "error" not in msg["tool_result"]:
                                raise LLMFormatError(
                                    message="Missing 'error' field in failed tool_result",
                                    provider="anthropic",
                                    details={"tool_result": msg["tool_result"]},
                                )
                            result = "Tool Call Failed: " + str(
                                msg["tool_result"]["error"]
                            )

                        content_parts.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": msg["tool_result"]["tool_id"],
                                "content": result,
                            }
                        )

                    # If no content was added, add a fallback message
                    # This ensures we don't send an empty content array which would be invalid
                    if not content_parts:
                        content_parts.append(
                            {"type": "text", "text": "No content available"}
                        )

                    # Add the completed message with role and content parts
                    formatted_messages.append({"role": role, "content": content_parts})

                except LLMError:
                    # Let our custom exceptions propagate
                    raise
                except Exception as e:
                    # Wrap other errors with message context
                    raise LLMFormatError(
                        message=f"Failed to format message at index {msg_index}",
                        provider="anthropic",
                        details={"error": str(e), "message": msg},
                    )

            return formatted_messages

        except LLMError:
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise LLMFormatError(
                message="Failed to format messages for Anthropic API",
                provider="anthropic",
                details={"error": str(e)},
            )

    def _raw_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        top_k: int = 40,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking_budget: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text from the Anthropic API.

        Args:
            event_id (str): Unique identifier for this generation event
            system_prompt (str): System instructions for the model
            messages (List[Dict[str, Any]]): List of messages in the standard format
            max_tokens (int): Maximum number of tokens to generate
            temp (float): Temperature for generation
            top_k (int): Top-k sampling parameter
            tools (List[Dict[str, Any]], optional): List of tools the LLM can use
            thinking_budget (int, optional): Maximum tokens to use for thinking (Claude 3.7)

        Returns:
            Tuple[str, Dict[str, Any]]: Generated text and usage data

        Raises:
            LLMError: With appropriate error subtype based on the failure
        """
        try:
            # Format messages for Anthropic's API (will raise appropriate errors)
            try:
                formatted_messages = self._format_messages_for_model(messages)
            except LLMError:
                # Let our custom exceptions propagate
                raise
            except Exception as e:
                # Wrap other formatting errors
                raise LLMFormatError(
                    message=f"Failed to format messages for Anthropic API: {e!s}",
                    provider="anthropic",
                    details={"error": str(e)},
                )

            # Prepare request body with required parameters
            request_body = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "temperature": temp,
                "system": system_prompt,
                "messages": formatted_messages,
            }

            # Add tools if provided
            if tools:
                # Validate tools format
                if not isinstance(tools, list):
                    raise LLMConfigurationError(
                        message="Tools parameter must be a list",
                        provider="anthropic",
                        details={"tools": tools},
                    )
                request_body["tools"] = tools

            # Configure thinking if budget is provided and model supports it
            if thinking_budget:
                if self.model_name not in self.THINKING_MODELS:
                    raise LLMConfigurationError(
                        message=f"Thinking budget is only supported for these models: {self.THINKING_MODELS}",
                        provider="anthropic",
                        details={
                            "model": self.model_name,
                            "thinking_budget": thinking_budget,
                        },
                    )

                # Enable thinking with specified token budget
                request_body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                # Adjust max tokens to account for thinking budget
                if thinking_budget is not None:
                    # Ensure both are integers before adding
                    max_tokens_value = request_body.get("max_tokens", 0)
                    if isinstance(max_tokens_value, (int, float)):
                        request_body["max_tokens"] = int(max_tokens_value) + int(
                            thinking_budget
                        )
                # Set temperature to 1 for thinking models (recommended)
                request_body["temperature"] = 1

            # Make API call to Anthropic with retry logic
            try:
                response = self._http_call_with_retry(
                    method="POST",
                    url="https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.config["api_key"],
                        "anthropic-version": "2023-06-01",  # This is the base version
                        "anthropic-beta": "tools-2023-12-13",  # Enable latest tool features
                        "content-type": "application/json",
                    },
                    json=request_body,
                    max_retries=3,
                )
            except LLMError:
                # Let our custom exceptions propagate
                raise

            # Parse response JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                raise LLMFormatError(
                    message="Failed to parse JSON response from Anthropic API",
                    provider="anthropic",
                    details={
                        "error": str(e),
                        "response_text": response.text[:1000],
                    },  # First 1000 chars for context
                )

            # Validate response structure
            if "content" not in response_data:
                raise LLMFormatError(
                    message="Missing 'content' field in Anthropic API response",
                    provider="anthropic",
                    details={"response": response_data},
                )

            if "usage" not in response_data:
                raise LLMFormatError(
                    message="Missing 'usage' field in Anthropic API response",
                    provider="anthropic",
                    details={"response": response_data},
                )

            # Extract usage information
            usage = response_data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens

            # Count images in input messages to calculate image costs
            num_images = sum(
                len(msg.get("image_paths", [])) + len(msg.get("image_urls", []))
                for msg in messages
            )

            # Get model-specific cost rates
            costs = self.get_model_costs()

            # Calculate costs based on usage
            read_cost = input_tokens * costs["read_token"]
            write_cost = output_tokens * costs["write_token"]
            # Some models might not have image costs
            image_cost = num_images * (costs["image_cost"] or 0.0)

            # Prepare comprehensive usage statistics (using base type structure)
            usage_stats = {
                "read_tokens": input_tokens,
                "write_tokens": output_tokens,
                "images": num_images,
                "total_tokens": total_tokens,
                "read_cost": round(read_cost, 6),
                "write_cost": round(write_cost, 6),
                "image_cost": round(image_cost, 6),
                "total_cost": round(read_cost + write_cost + image_cost, 6),
                "model": self.model_name,
                "provider": Provider.ANTHROPIC.value,
                "successful": True,
            }

            # Extract text response and any special content
            response_text = ""
            has_content = False

            for content in response_data["content"]:
                # Combine all text content parts
                if content["type"] == "text":
                    has_content = True
                    response_text += content.get("text", "")

                # Extract thinking content if present
                if content["type"] == "thinking":
                    usage_stats["thinking"] = content["thinking"]
                    usage_stats["thinking_signature"] = content["signature"]

                # Extract tool use information if present
                if content["type"] == "tool_use":
                    has_content = True
                    usage_stats["tool_use"] = content

            # Check if we received any actual content
            if not has_content and not usage_stats.get("tool_use"):
                raise LLMContentError(
                    message="Empty response from Anthropic API (no text or tool use)",
                    provider="anthropic",
                    details={"response": response_data},
                )

            return response_text, usage_stats

        except LLMError:
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            # Map any other exceptions
            raise self._map_anthropic_error(e)

    def _map_anthropic_error(
        self, error: Exception, response: Optional[requests.Response] = None
    ) -> LLMError:
        """
        Map Anthropic-specific errors to standardized LLM error types.

        This method analyzes an exception from the Anthropic API and converts it
        to the appropriate LLM exception type with relevant context.

        Args:
            error: The original Anthropic exception
            response: Optional HTTP response object with status code and headers

        Returns:
            LLMError: The mapped exception with detailed context
        """
        error_str = str(error).lower()
        status_code = getattr(response, "status_code", None) if response else None

        # Try to parse error details from response if available
        error_details = {}
        if response and hasattr(response, "json"):
            try:
                error_data = response.json()
                if isinstance(error_data, dict):
                    error_details = error_data
            except:
                pass

        # Authentication errors
        if (
            "api key" in error_str
            or "authentication" in error_str
            or "auth" in error_str
            or status_code == 401
        ):
            return LLMAuthenticationError(
                message="Anthropic authentication failed: Invalid API key",
                provider="anthropic",
                details={"error": str(error), "response": error_details},
            )

        # Rate limiting errors
        if (
            "rate limit" in error_str
            or "too many requests" in error_str
            or status_code == 429
        ):
            # Try to extract retry-after header if available
            retry_after = None
            if response and hasattr(response, "headers"):
                retry_after_header = response.headers.get("retry-after")
                if retry_after_header and retry_after_header.isdigit():
                    retry_after = int(retry_after_header)

            return LLMRateLimitError(
                message="Anthropic rate limit exceeded",
                provider="anthropic",
                retry_after=retry_after,
                details={"error": str(error), "response": error_details},
            )

        # Context length errors
        if "context" in error_str and (
            "length" in error_str or "window" in error_str or "too long" in error_str
        ):
            return LLMProviderError(
                message="Anthropic model context length exceeded",
                provider="anthropic",
                details={
                    "error": str(error),
                    "model": self.model_name,
                    "context_window": self.get_context_window(),
                },
            )

        # Content policy violations
        if (
            "content policy" in error_str
            or "content_policy" in error_str
            or "violation" in error_str
        ):
            return LLMContentError(
                message="Anthropic content policy violation",
                provider="anthropic",
                details={"error": str(error), "response": error_details},
            )

        # Parameter validation errors
        if "invalid" in error_str and (
            "parameter" in error_str or "argument" in error_str
        ):
            return LLMConfigurationError(
                message="Invalid parameter for Anthropic API",
                provider="anthropic",
                details={"error": str(error), "response": error_details},
            )

        # Server errors (5xx)
        if (
            status_code
            and str(status_code).startswith("5")
            or "server" in error_str
            or "service" in error_str
        ):
            return LLMServiceUnavailableError(
                message="Anthropic API server error",
                provider="anthropic",
                details={
                    "error": str(error),
                    "status_code": status_code,
                    "response": error_details,
                },
            )

        # Tool/function related errors
        if "function" in error_str or "tool" in error_str:
            return LLMToolError(
                message="Anthropic tool/function error",
                provider="anthropic",
                details={"error": str(error), "response": error_details},
            )

        # Thinking related errors
        if "thinking" in error_str or "budget" in error_str:
            return LLMThinkingError(
                message="Anthropic thinking error",
                provider="anthropic",
                details={"error": str(error), "response": error_details},
            )

        # Default fallback
        return LLMProviderError(
            message=f"Anthropic API error: {error!s}",
            provider="anthropic",
            details={
                "error": str(error),
                "status_code": status_code,
                "response": error_details,
            },
        )

    def get_supported_models(self) -> list[str]:
        """
        Get the list of supported model identifiers.

        Returns:
            list[str]: List of supported model identifiers
        """
        return self.SUPPORTED_MODELS

    def validate_model(self, model_name: str) -> bool:
        """
        Check if a model name is supported by this provider.

        Args:
            model_name: The model name to validate

        Returns:
            bool: True if the model is supported, False otherwise
        """
        return model_name in self.SUPPORTED_MODELS

    def _call_with_retry(
        self,
        func: Callable[..., Any],
        *args: Any,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        retryable_errors: Optional[List[type[Exception]]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call a function with exponential backoff retry logic.

        This implementation of _call_with_retry matches the base LLM class signature
        and handles retrying function calls that may fail due to temporary issues.

        Args:
            func: Function to call
            *args: Arguments to pass to the function
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Factor to increase backoff time on each retry
            retryable_errors: List of exception types that should trigger a retry
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The return value from the function

        Raises:
            The last exception encountered after all retries are exhausted
        """
        # Set default retryable errors if not provided
        if retryable_errors is None:
            retryable_errors = [
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
            ]

        last_exception = None
        current_backoff = initial_backoff

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except tuple(retryable_errors) as e:
                last_exception = e
                if attempt < max_retries:
                    time.sleep(current_backoff)
                    current_backoff *= backoff_factor
                else:
                    raise last_exception

    def _http_call_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        retry_status_codes: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> requests.Response:
        """
        Make an HTTP request with retry logic for specific status codes.

        This provider-specific implementation handles retry logic for HTTP requests,
        using exponential backoff for specified status codes.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Factor to increase backoff time on each retry
            retry_status_codes: List of HTTP status codes that should trigger a retry
            **kwargs: Additional arguments to pass to requests

        Returns:
            requests.Response: The HTTP response

        Raises:
            requests.RequestException: The last exception encountered after all retries
        """
        if retry_status_codes is None:
            retry_status_codes = [429, 500, 502, 503, 504]

        retry_count = 0
        backoff_time = initial_backoff
        last_response = None

        while retry_count <= max_retries:  # <= because first attempt is not a retry
            try:
                # Make the request
                response = requests.request(method, url, **kwargs)
                last_response = response

                # If status code indicates success, return the response
                if response.status_code < 400:
                    return response

                # If status code is in retry_status_codes, retry
                if (
                    response.status_code in retry_status_codes
                    and retry_count < max_retries
                ):
                    retry_count += 1

                    # Get retry-after if available (for rate limits)
                    retry_after = None
                    if (
                        response.status_code == 429
                        and "retry-after" in response.headers
                    ):
                        retry_after_header = response.headers.get("retry-after")
                        if retry_after_header and retry_after_header.isdigit():
                            retry_after = float(retry_after_header)

                    # Use retry-after if available, otherwise use exponential backoff
                    wait_time = retry_after if retry_after else backoff_time

                    # Sleep before retry
                    time.sleep(wait_time)

                    # Increase backoff for next retry
                    backoff_time *= backoff_factor
                    continue

                # If we get here, it's an error we don't want to retry or we've exhausted retries
                error_message = f"Anthropic API error ({response.status_code})"
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict) and "error" in error_data:
                        error_message = f"{error_message}: {error_data['error']}"
                except:
                    pass

                # Map the error to appropriate type and raise
                raise self._map_anthropic_error(Exception(error_message), response)

            except requests.RequestException as e:
                # Network errors might be retryable
                if retry_count < max_retries:
                    retry_count += 1
                    time.sleep(backoff_time)
                    backoff_time *= backoff_factor
                    continue

                # If we've exhausted retries, map and raise
                raise self._map_anthropic_error(e)

        # Should never get here, but just in case
        if last_response:
            raise self._map_anthropic_error(
                Exception(f"API call failed after {max_retries} retries"), last_response
            )

        # Generic fallback
        raise LLMProviderError(
            message=f"API call failed after {max_retries} retries", provider="anthropic"
        )

    def _convert_function_to_tool(self, function: Callable) -> Dict[str, Any]:
        """
        Convert a function to the tool format required by Anthropic's API.

        Args:
            function: The callable function to convert

        Returns:
            Dict[str, Any]: Tool definition in Anthropic's format
            (This will conform to the AnthropicTool structure internally)
        """
        # Get the base tool definition
        base_tool = super()._convert_function_to_tool(function)

        # Format specifically for Anthropic's API using the unified type structure
        # But still maintain compatibility with Anthropic's expected format
        anthropic_tool: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": base_tool["name"],
                "description": base_tool.get("description", ""),
                "parameters": {"type": "object", "properties": {}},
            },
        }

        # Convert input schema to parameters
        if "input_schema" in base_tool and "properties" in base_tool["input_schema"]:
            # Use dict() to ensure we have a dictionary
            properties_dict = dict(base_tool["input_schema"].get("properties", {}))
            anthropic_tool["function"]["parameters"]["properties"] = properties_dict

            # Add required parameters if present
            if "required" in base_tool["input_schema"]:
                # Use list() to ensure we have a list
                required_params = list(base_tool["input_schema"].get("required", []))
                anthropic_tool["function"]["parameters"]["required"] = required_params

        return anthropic_tool

    def _convert_functions_to_tools(
        self, functions: List[Callable]
    ) -> List[Dict[str, Any]]:
        """Convert functions to Anthropic tool format

        Args:
            functions: List of callable functions to convert

        Returns:
            List[Dict[str, Any]]: List of tool definitions in Anthropic format
            (These will conform to the AnthropicTool structure internally)
        """
        tools: List[Dict[str, Any]] = []
        for function in functions:
            tool = self._convert_function_to_tool(function)
            tools.append(tool)
        return tools

    def stream_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        functions: Optional[List[Callable[..., Any]]] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Stream a response from the Anthropic API, yielding chunks as they become available.

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
            LLMError: With appropriate error subtype based on the failure
        """
        try:
            # Import Anthropic here to avoid import errors if not installed
            try:
                import anthropic
            except ImportError:
                raise LLMProviderError(
                    message="Anthropic package not installed",
                    provider="anthropic",
                    details={
                        "required_package": "anthropic",
                        "install_command": "pip install anthropic",
                    },
                )

            # Initialize client if needed
            if not hasattr(self, "client"):
                # Authentication will raise proper errors if it fails
                self.auth()
                # Create the client
                self.client = anthropic.Anthropic(api_key=self.config["api_key"])

            # Convert messages to Anthropic format
            try:
                formatted_messages = self._format_messages_for_model(messages)
            except LLMError:
                # Let our custom exceptions propagate
                raise
            except Exception as e:
                # Wrap other formatting errors
                raise LLMFormatError(
                    message=f"Failed to format messages for Anthropic streaming: {e!s}",
                    provider="anthropic",
                    details={"error": str(e)},
                )

            # Prepare tools if functions are provided
            tools = None
            if functions:
                try:
                    tools = []
                    for function in functions:
                        # Convert function to tool using the utility method
                        tools.append(self._convert_function_to_tool(function))
                except Exception as e:
                    raise LLMConfigurationError(
                        message="Failed to convert functions to tools for Anthropic",
                        provider="anthropic",
                        details={"error": str(e)},
                    )

            # Count images for cost calculation
            image_count = 0
            for message in messages:
                if message.get("message_type") == "human":
                    image_count += len(message.get("image_paths", []))
                    image_count += len(message.get("image_urls", []))

            try:
                # Import Anthropic types for proper casting
                import anthropic

                # Cast types to satisfy Anthropic's API typing requirements
                # First ensure we have the correct object structure
                api_messages = []
                for msg in formatted_messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        api_messages.append(msg)

                # Create API parameters dict with proper typing
                api_params: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": api_messages,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                    "stream": True,  # Enable streaming
                }

                # Only add system if it exists and is not empty
                if system_prompt:
                    api_params["system"] = system_prompt

                # Only add tools if they exist and are properly formatted
                if tools and isinstance(tools, list):
                    api_params["tools"] = tools

                # Create a streaming request
                # Note: We can't fully type-check the Anthropic client API call
                # as it's an external dependency with incomplete type stubs
                response = self.client.messages.create(**api_params)
            except anthropic.APIError as api_error:
                # Map Anthropic API errors to our custom types
                status_code = getattr(api_error, "status_code", None)
                if hasattr(api_error, "response") and hasattr(
                    api_error.response, "json"
                ):
                    try:
                        error_data = api_error.response.json()
                        raise self._map_anthropic_error(api_error, api_error.response)
                    except (ValueError, AttributeError):
                        pass
                # Use basic error mapping if we can't get more details
                raise self._map_anthropic_error(api_error)
            except Exception as e:
                # Map other errors
                raise self._map_anthropic_error(e)

            # Initialize variables to accumulate data
            accumulated_text = ""
            accumulated_tokens = 0
            tool_call_data: Dict[str, ToolCallData] = {}

            # Estimate input tokens (a more accurate count would require using Anthropic's tokenizer)
            # This is a simplified estimation
            input_tokens = 0
            for message in formatted_messages:
                for content_part in message["content"]:
                    if content_part.get("type") == "text":
                        # Rough estimation: count words and multiply by 1.3
                        text = content_part.get("text", "")
                        input_tokens += int(len(text.split()) * 1.3)
                    elif content_part.get("type") == "image":
                        # Anthropic charges more for images
                        input_tokens += 1000  # Rough estimate

            # Add system prompt tokens if present
            if system_prompt:
                input_tokens += int(len(system_prompt.split()) * 1.3)

            # Process each chunk as it arrives
            try:
                for chunk in response:
                    # Check for content delta
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                        content = chunk.delta.text
                        if content:
                            accumulated_text += content
                            # Rough token estimation: 1 token per character / 4
                            token_estimate = max(1, len(content) // 4)
                            accumulated_tokens += token_estimate

                            # Call the callback if provided
                            if callback is not None:
                                # Create partial usage data following the standardized structure
                                partial_usage = {
                                    "event_id": event_id,
                                    "model": self.model_name,
                                    "read_tokens": input_tokens,
                                    "write_tokens": accumulated_tokens,
                                    "images": image_count,
                                    "total_tokens": input_tokens + accumulated_tokens,
                                    "is_complete": False,
                                    "tool_use": tool_call_data,
                                    "provider": Provider.ANTHROPIC.value,
                                    "successful": True,
                                }
                                callback(content, partial_usage)

                            # Yield the content chunk and partial usage data using standardized structure
                            yield content, {
                                "event_id": event_id,
                                "model": self.model_name,
                                "read_tokens": input_tokens,
                                "write_tokens": accumulated_tokens,
                                "images": image_count,
                                "total_tokens": input_tokens + accumulated_tokens,
                                "is_complete": False,
                                "tool_use": tool_call_data,
                                "provider": Provider.ANTHROPIC.value,
                                "successful": True,
                            }

                    # Check for tool calls in the delta
                    if hasattr(chunk, "delta") and hasattr(chunk.delta, "tool_use"):
                        tool_use = chunk.delta.tool_use
                        if tool_use:
                            tool_id = tool_use.id

                            # Initialize or update tool call data
                            if tool_id not in tool_call_data:
                                # Get name and input from the tool use, defaulting to empty strings
                                tool_name = (
                                    tool_use.name if hasattr(tool_use, "name") else ""
                                )
                                tool_input = (
                                    tool_use.input if hasattr(tool_use, "input") else ""
                                )

                                # Create a properly typed tool call data entry
                                tool_call_data[tool_id] = {
                                    "id": tool_id,
                                    "name": tool_name,
                                    "arguments": str(tool_input),  # Ensure string type
                                    "type": "function",
                                }
                            elif hasattr(tool_use, "input") and tool_use.input:
                                # Append to existing arguments if this is a partial update
                                # Using the dictionary access syntax since we're modifying an existing entry
                                existing_args = tool_call_data[tool_id]["arguments"]
                                tool_call_data[tool_id]["arguments"] = (
                                    existing_args + str(tool_use.input)
                                )
            except anthropic.APIError as api_error:
                # Handle streaming errors differently - map but preserve what we've got so far
                mapped_error = self._map_anthropic_error(api_error)
                # Only add details if the error type supports it
                if hasattr(mapped_error, "details"):
                    if (
                        not hasattr(mapped_error, "details")
                        or mapped_error.details is None
                    ):
                        mapped_error.details = {}
                    mapped_error.details["partial_text"] = accumulated_text
                    mapped_error.details["tool_calls"] = tool_call_data
                raise mapped_error
            except Exception as stream_error:
                # Map other streaming errors
                raise self._map_anthropic_error(stream_error)

            # Calculate costs
            model_costs = self.get_model_costs()

            # Get cost values with safe defaults
            read_token_cost = float(model_costs.get("read_token", 0.0) or 0.0)
            write_token_cost = float(model_costs.get("write_token", 0.0) or 0.0)

            # Calculate costs with explicit type conversions
            read_cost = float(input_tokens) * read_token_cost
            write_cost = float(accumulated_tokens) * write_token_cost

            # Calculate image cost if applicable
            image_cost = 0.0
            if image_count > 0 and "image_cost" in model_costs:
                # Get image cost with safe default and ensure it's a float
                image_token_cost = float(model_costs.get("image_cost", 0.0) or 0.0)
                image_cost = float(image_count) * image_token_cost

            total_cost = read_cost + write_cost + image_cost

            # Create final usage data following the standardized structure
            final_usage = {
                "event_id": event_id,
                "model": self.model_name,
                "read_tokens": input_tokens,
                "write_tokens": accumulated_tokens,
                "images": image_count,
                "total_tokens": input_tokens + accumulated_tokens,
                "read_cost": read_cost,
                "write_cost": write_cost,
                "image_cost": image_cost,
                "total_cost": total_cost,
                "is_complete": True,
                "tool_use": tool_call_data,
                "provider": Provider.ANTHROPIC.value,
                "successful": True,
                "latency_ms": None,  # Would be calculated if timing is implemented
            }

            # Call the callback with an empty string to signal completion
            if callback is not None:
                callback("", final_usage)

            # Yield an empty string with the final usage data to signal completion
            yield "", final_usage

        except LLMError:
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            # Map other streaming errors
            raise self._map_anthropic_error(e)
