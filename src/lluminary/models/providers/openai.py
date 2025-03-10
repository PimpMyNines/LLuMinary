"""
OpenAI LLM provider implementation.

This module implements support for OpenAI's models within the LLM handler framework.
It handles authentication, message formatting, image processing, and cost tracking for
all OpenAI models including GPT-4o and "o" series models.

## OpenAI Messaging System Overview

OpenAI's models use a structured content format with specific requirements:

1. Messages are organized as an array with roles (user, assistant, system, tool)
2. Content can be either:
   - Simple text strings (for basic messages)
   - Array of content parts (for messages with images or multiple parts)
   - Content parts include text (type: "text") and images (type: "image_url")
3. Function/tool calls are handled via special properties:
   - Assistant messages can include "tool_calls" array
   - Tool responses use a "tool_call_id" and "content" property

### Message Conversion Process
This implementation converts the standard LLM handler message format:
```
{
    "message_type": "human"|"ai"|"tool"|"system"|"developer",
    "message": "text content",
    "image_paths": ["local/path/to/image.jpg"],
    "image_urls": ["https://example.com/image.jpg"],
    "tool_use": {"id": "id", "name": "func_name", "input": {...}},
    "tool_result": {"tool_id": "id", "success": bool, "result": any, "error": str}
}
```

Into OpenAI's format:
```
{
    "role": "user"|"assistant"|"system"|"tool",
    "content": "text content" | [
        {
            "type": "text",
            "text": "text content"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,base64_encoded_image"
            }
        }
    ],
    "tool_calls": [  # Only for assistant messages with tool use
        {
            "type": "function",
            "id": "id",
            "function": {
                "name": "func_name",
                "arguments": "json_string"
            }
        }
    ],
    "tool_call_id": "id",  # Only for tool response messages
}
```

### Key Features
- Full support for text, images, and function calling
- Special support for reasoning models (o1, o3-mini)
- Advanced image token calculation and cost estimation
- Smart image processing and scaling
- Support for both simple text and multipart content
"""

import base64
import json
import os
import time
from io import BytesIO
from math import ceil
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

import requests
from openai import OpenAI
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
    LLMToolError,
    LLMValidationError,
)
from ...utils.aws import get_secret
from ..base import LLM


# Type definitions for OpenAI API structures
class OpenAITextContent(TypedDict):
    """Type definition for OpenAI text content part."""

    type: Literal["text"]
    text: str


class OpenAIImageUrl(TypedDict):
    """Type definition for OpenAI image URL."""

    url: str


class OpenAIImageContent(TypedDict):
    """Type definition for OpenAI image content part."""

    type: Literal["image_url"]
    image_url: OpenAIImageUrl


# Union type for content parts
OpenAIContentPart = Union[OpenAITextContent, OpenAIImageContent]


class OpenAIFunction(TypedDict):
    """Type definition for OpenAI function in tool calls."""

    name: str
    arguments: str


class OpenAIToolCall(TypedDict):
    """Type definition for OpenAI tool call."""

    type: Literal["function"]
    id: str
    function: OpenAIFunction


class OpenAIUserMessage(TypedDict):
    """Type definition for OpenAI user message."""

    role: Literal["user"]
    content: Union[str, List[OpenAIContentPart]]


class OpenAIAssistantMessage(TypedDict):
    """Type definition for OpenAI assistant message."""

    role: Literal["assistant"]
    content: Union[str, List[OpenAIContentPart], None]
    tool_calls: Optional[List[OpenAIToolCall]]


class OpenAISystemMessage(TypedDict):
    """Type definition for OpenAI system message."""

    role: Literal["system"]
    content: Union[str, List[OpenAIContentPart]]


class OpenAIToolMessage(TypedDict):
    """Type definition for OpenAI tool message."""

    role: Literal["tool"]
    content: str
    tool_call_id: str


# Union type for all message types
OpenAIMessage = Union[
    OpenAIUserMessage, OpenAIAssistantMessage, OpenAISystemMessage, OpenAIToolMessage
]


# Function/tool related type definitions
class OpenAIParameterProperty(TypedDict, total=False):
    """Type definition for a parameter property in a function schema."""

    type: str
    description: Optional[str]
    enum: Optional[List[str]]
    items: Optional[Dict[str, Any]]
    properties: Optional[Dict[str, Any]]
    required: Optional[List[str]]


class OpenAIParameters(TypedDict, total=False):
    """Type definition for parameters in a function schema."""

    type: str
    properties: Dict[str, OpenAIParameterProperty]
    required: Optional[List[str]]
    additionalProperties: bool


class OpenAIFunctionDefinition(TypedDict):
    """Type definition for a function definition in a tool."""

    name: str
    description: str
    parameters: OpenAIParameters
    strict: Optional[bool]


class OpenAITool(TypedDict):
    """Type definition for an OpenAI tool."""

    type: Literal["function"]
    function: OpenAIFunctionDefinition


# Streaming response type definitions
class OpenAIStreamingToolCall(TypedDict):
    """Type definition for tool call data in streaming responses."""

    id: str
    type: Literal["function"]
    function: OpenAIFunction


class OpenAILLM(LLM):
    """
    Implementation of OpenAI's LLM models.

    This class provides methods for authentication, message formatting, image processing,
    and interaction with OpenAI's API. It converts between the standardized
    LLM handler message format and OpenAI's specific structure.

    Attributes:
        SUPPORTED_MODELS: List of OpenAI models supported by this implementation
        REASONING_MODELS: List of models that support enhanced reasoning capabilities
        CONTEXT_WINDOW: Maximum token limits for each model
        SUPPORTS_IMAGES: Whether image inputs are supported (true for all OpenAI models)
        COST_PER_MODEL: Cost information per token for each model
        THINKING_MODELS: List of models that support explicit thinking capabilities
    """

    # Define THINKING_MODELS for API compatibility with other providers

    SUPPORTED_MODELS = [
        "gpt-4.5-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o3-mini",
    ]

    REASONING_MODELS = [
        "o1",
        "o3-mini",
    ]

    CONTEXT_WINDOW = {
        "gpt-4.5-preview": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "o1": 200000,
        "o3-mini": 200000,
    }

    # All OpenAI models support image input
    SUPPORTS_IMAGES = True

    # Base costs per token for each model
    COST_PER_MODEL = {
        "gpt-4.5-preview": {
            "read_token": 0.0000750,
            "write_token": 0.00015,
            "image_cost": None,  # Calculated per image based on size/detail
        },
        "gpt-4o": {
            "read_token": 0.0000025,
            "write_token": 0.00001,
            "image_cost": None,  # Calculated per image based on size/detail
        },
        "gpt-4o-mini": {
            "read_token": 0.00000015,
            "write_token": 0.0000006,
            "image_cost": None,  # Calculated per image based on size/detail
        },
        "o1": {
            "read_token": 0.000015,
            "write_token": 0.00006,
            "image_cost": None,  # Calculated per image based on size/detail
        },
        "o3-mini": {
            "read_token": 0.0000011,
            "write_token": 0.0000044,
            "image_cost": None,  # Calculated per image based on size/detail
        },
    }

    # Image processing constants
    LOW_DETAIL_TOKEN_COST = 85
    HIGH_DETAIL_TILE_COST = 170
    HIGH_DETAIL_BASE_COST = 85
    MAX_IMAGE_SIZE = 2048
    TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    # List of models supporting "thinking" capability
    THINKING_MODELS = ["gpt-4o", "o1", "o3-mini"]

    # Add models that support embeddings
    EMBEDDING_MODELS = [
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]

    # Add models that support reranking
    RERANKING_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]

    # Default embedding model to use if none specified
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    # Default reranking model to use if none specified
    DEFAULT_RERANKING_MODEL = "text-embedding-3-small"

    # Add reranking costs after embedding_costs
    embedding_costs = {
        "text-embedding-ada-002": 0.0001,
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
    }

    # Costs for reranking operations
    reranking_costs = {
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
    }

    # Costs for image generation by model and size
    IMAGE_GENERATION_COSTS = {
        "dall-e-3": {"1024x1024": 0.040, "1024x1792": 0.080, "1792x1024": 0.080},
        "dall-e-2": {"1024x1024": 0.020, "512x512": 0.018, "256x256": 0.016},
    }

    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Initialize an OpenAI LLM instance.

        Args:
            model_name: Name of the OpenAI model to use
            **kwargs: Additional arguments passed to the base LLM class
        """
        # Set these attributes before calling super().__init__ which calls auth()
        self.api_base = kwargs.get("api_base", None)
        self.timeout = kwargs.get("timeout", 60)

        # Store embedding costs for different models
        self.embedding_costs = {
            "text-embedding-ada-002": 0.0001,  # $0.10 per million tokens
            "text-embedding-3-small": 0.00002,  # $0.02 per million tokens
            "text-embedding-3-large": 0.00013,  # $0.13 per million tokens
        }

        # Call super().__init__ after setting required attributes
        super().__init__(model_name, **kwargs)

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "openai"

    def auth(self) -> None:
        """
        Authenticate with OpenAI using API key from AWS Secrets Manager or environment variables.

        This method retrieves the OpenAI API key from:
        1. The API key passed during initialization (if provided)
        2. AWS Secrets Manager (if available)
        3. Environment variable OPENAI_API_KEY (as fallback)

        It then initializes the official OpenAI Python client.

        Raises:
            AuthenticationError: If authentication fails and no valid API key is found
        """
        try:
            # If API key was already provided during initialization, use it
            if self.config.get("api_key"):
                api_key = self.config["api_key"]
            else:
                try:
                    # Try to get the API key from Secrets Manager
                    secret = get_secret("openai_api_key", required_keys=["api_key"])
                    api_key = secret["api_key"]
                except Exception as secret_error:
                    # If Secrets Manager fails, check environment variables
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        # Re-raise using proper error type with context
                        raise LLMAuthenticationError(
                            message="Failed to get API key for OpenAI",
                            provider="openai",
                            details={
                                "secret_error": str(secret_error),
                                "tried_sources": [
                                    "AWS Secrets Manager",
                                    "environment variables",
                                ],
                            },
                        )

            # Store the API key in config and initialize the client
            self.config["api_key"] = api_key

            try:
                self.client = OpenAI(api_key=api_key, base_url=self.api_base)
                # Verify the credentials by making a minimal API call
                self.client.models.list()
            except Exception as api_error:
                # Handle specific OpenAI errors
                if (
                    "invalid_api_key" in str(api_error).lower()
                    or "authentication" in str(api_error).lower()
                ):
                    raise LLMAuthenticationError(
                        message="Invalid OpenAI API key",
                        provider="openai",
                        details={"error": str(api_error)},
                    )
                elif (
                    "access" in str(api_error).lower()
                    or "permission" in str(api_error).lower()
                ):
                    raise LLMAuthenticationError(
                        message="Insufficient permissions with OpenAI API key",
                        provider="openai",
                        details={"error": str(api_error)},
                    )
                else:
                    # General initialization error
                    raise LLMProviderError(
                        message="Failed to initialize OpenAI client",
                        provider="openai",
                        details={"error": str(api_error)},
                    )

        except LLMError:
            # Pass through our custom exceptions
            raise
        except Exception as e:
            # Catch all other exceptions
            raise LLMProviderError(
                message="OpenAI authentication failed",
                provider="openai",
                details={"error": str(e)},
            )

    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64 string.

        This method simply reads the binary content of an image file
        and encodes it as base64 without altering the image format.

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
                    provider="openai",
                    details={"image_path": image_path},
                )

            # Try to read and encode the file
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        except LLMProviderError:
            # Re-raise provider error
            raise
        except FileNotFoundError as e:
            # For cases where path exists check was bypassed somehow
            raise LLMProviderError(
                message=f"Image file not found: {image_path}",
                provider="openai",
                details={"error": str(e), "image_path": image_path},
            )
        except PermissionError as e:
            raise LLMProviderError(
                message=f"Permission denied when reading image file: {image_path}",
                provider="openai",
                details={"error": str(e), "image_path": image_path},
            )
        except Exception as e:
            raise LLMFormatError(
                message=f"Failed to encode image {image_path}",
                provider="openai",
                details={"error": str(e), "image_path": image_path},
            )

    def _encode_image_url(self, image_url: str) -> str:
        """
        Download and encode an image from URL.

        This method:
        1. Fetches the image from the given URL
        2. Converts to RGB format (handling transparency if needed)
        3. Saves as JPEG format
        4. Encodes as base64

        Args:
            image_url: URL of the image to download

        Returns:
            str: Base64 encoded image string

        Raises:
            LLMProviderError: If download fails or network error occurs
            LLMFormatError: If image format is invalid or can't be processed
        """
        try:
            # Download the image with timeout
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                # Use a specific error type for network issues
                raise LLMProviderError(
                    message=f"Failed to download image from URL: {image_url}",
                    provider="openai",
                    details={
                        "error": str(e),
                        "url": image_url,
                        "status_code": (
                            getattr(response, "status_code", None)
                            if "response" in locals()
                            else None
                        ),
                    },
                )

            # Process the image
            try:
                # Convert to JPEG format
                img_original = Image.open(BytesIO(response.content))
                # Handle transparency (RGBA or LA modes) by adding white background
                if img_original.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img_original.size, (255, 255, 255))
                    background.paste(img_original, mask=img_original.split()[-1])
                    img_processed = background
                elif img_original.mode != "RGB":
                    img_processed = img_original.convert("RGB")
                else:
                    img_processed = img_original

                # Save as JPEG in memory
                output = BytesIO()
                img_processed.save(output, format="JPEG", quality=95)
                return base64.b64encode(output.getvalue()).decode("utf-8")

            except Exception as e:
                # Image processing errors
                raise LLMFormatError(
                    message=f"Failed to process image from URL: {image_url}",
                    provider="openai",
                    details={"error": str(e), "url": image_url},
                )

        except (LLMProviderError, LLMFormatError):
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise LLMProviderError(
                message=f"Unexpected error processing image from URL: {image_url}",
                provider="openai",
                details={"error": str(e), "url": image_url},
            )

    def _format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format messages for OpenAI's API.

        This method transforms the unified message format used across providers into
        OpenAI's specific message structure. Key transformations include:
        1. Converting message_type to OpenAI role (user, assistant, system, tool)
        2. Handling image attachments as content parts
        3. Processing function/tool calls for assistant messages
        4. Managing tool responses for tool messages

        Args:
            messages: List of messages in standard format

        Returns:
            List[OpenAIMessage]: Messages formatted for OpenAI API

        Raises:
            LLMFormatError: If message formatting fails, particularly during image processing
        """
        formatted_messages: List[Dict[str, Any]] = []
        for message in messages:
            # To handle the dual type nature, we'll maintain a flag for type
            is_content_string = False
            content_parts: List[OpenAIContentPart] = []
            content_text = ""
            tool_calls: List[OpenAIToolCall] = []

            # Process local image files
            for image_path in message.get("image_paths", []):
                try:
                    # Convert image to base64 and format as image_url
                    image_base64 = self._encode_image(image_path)
                    image_content: OpenAIImageContent = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    }
                    content_parts.append(image_content)
                except Exception as e:
                    raise LLMFormatError(
                        message=f"Failed to process image file {image_path}",
                        provider="openai",
                        details={"error": str(e), "file_path": image_path},
                    )

            # Process image URLs
            for image_url in message.get("image_urls", []):
                try:
                    # Download, convert to base64, and format as image_url
                    image_base64 = self._encode_image_url(image_url)
                    url_image_content: OpenAIImageContent = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    }
                    content_parts.append(url_image_content)
                except Exception as e:
                    raise LLMFormatError(
                        message=f"Failed to process image URL {image_url}",
                        provider="openai",
                        details={"error": str(e), "url": image_url},
                    )

            # Add text content, handling different content formats
            if len(content_parts) > 0:
                # If we already have image content parts, add text as another part
                if message.get("message"):
                    text_content: OpenAITextContent = {
                        "type": "text",
                        "text": message["message"],
                    }
                    content_parts.append(text_content)
            else:
                # Simple case: just text, no need for content parts array
                if message.get("message"):
                    content_text = message["message"]
                    is_content_string = True
                else:
                    # Empty message, default to empty string
                    content_text = ""
                    is_content_string = True

            # Map message types to OpenAI roles
            message_type = message["message_type"]

            # Handle tool result messages (special case)
            if message_type == "tool_result":
                tool_id = message["tool_result"]["tool_id"]

                # Format result message based on success/failure
                result_text = ""
                if message["tool_result"].get("success"):
                    result_text = (
                        f"Tool Call Successful: {message['tool_result']['result']}"
                    )
                else:
                    result_text = f"Tool Call Failed: {message['tool_result']['error']}"

                # Create tool message as Dict[str, Any] to match parent class expectations
                tool_message: Dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_text,
                }
                formatted_messages.append(tool_message)
                continue  # Skip to next message

            # Process tool use for assistant messages
            if message_type == "ai" and message.get("tool_use"):
                tool_use = message["tool_use"]
                tool_call: OpenAIToolCall = {
                    "type": "function",
                    "id": tool_use["id"],
                    "function": {
                        "name": tool_use["name"],
                        "arguments": json.dumps(tool_use["input"]),
                    },
                }
                tool_calls.append(tool_call)

            # Create appropriate message based on role
            if message_type == "human":
                # Create the user message with the appropriate content type
                user_message: Dict[str, Any] = {
                    "role": "user",
                    "content": content_text if is_content_string else content_parts,
                }
                formatted_messages.append(user_message)

            elif message_type == "ai":
                # Create assistant message with the appropriate content type
                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": (
                        content_text
                        if is_content_string
                        else (content_parts if content_parts else None)
                    ),
                }

                # Add tool calls if present
                if tool_calls:
                    assistant_message["tool_calls"] = tool_calls

                formatted_messages.append(assistant_message)

            elif message_type in [
                "system",
                "developer",
            ]:  # Map developer to system for compatibility
                # Create system message with the appropriate content type
                system_message: Dict[str, Any] = {
                    "role": "system",
                    "content": content_text if is_content_string else content_parts,
                }
                formatted_messages.append(system_message)

        return formatted_messages

    def _format_tools_for_model(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert standard tool format into OpenAI's format.

        OpenAI requires tools to be formatted in their function calling format,
        with a specific structure including function name, description, and parameters.

        Args:
            tools: List of tools in standard format or Python callables

        Returns:
            List[OpenAITool]: Tools formatted for OpenAI API

        Raises:
            LLMFormatError: If tool processing fails
        """
        formatted_tools: List[Dict[str, Any]] = []

        for tool in tools:
            try:
                # Handle Python function objects
                if callable(tool):
                    # Convert callable to tool format using base method
                    base_tool = self._convert_function_to_tool(tool)

                    # Extract name and parameters from the base tool
                    tool_name = base_tool["name"]
                    tool_description = base_tool.get("description", "")
                    tool_params = base_tool.get("input_schema", {})

                    # Ensure additionalProperties is set to False
                    if "additionalProperties" not in tool_params:
                        tool_params["additionalProperties"] = False

                    # Create function definition for callable tools
                    callable_function_def: Dict[str, Any] = {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": tool_params,
                        "strict": True,
                    }

                    # Create OpenAI tool with this function definition
                    callable_openai_tool: Dict[str, Any] = {
                        "type": "function",
                        "function": callable_function_def,
                    }

                    # Set the common openai_tool variable that we'll add to the list
                    openai_tool = callable_openai_tool

                else:
                    # Handle dictionary specification
                    if not isinstance(tool, dict):
                        raise LLMFormatError(
                            message="Tool must be a callable or a dictionary",
                            provider="openai",
                            details={"tool_type": type(tool).__name__},
                        )

                    # Extract required fields
                    tool_name = tool.get("name")
                    if not tool_name:
                        raise LLMFormatError(
                            message="Tool dictionary must have a 'name' field",
                            provider="openai",
                            details={"tool": str(tool)},
                        )

                    # Get parameters with defaults
                    tool_description = tool.get("description", "")
                    tool_params = tool.get("input_schema", {})

                    # Ensure additionalProperties is set to False
                    if (
                        isinstance(tool_params, dict)
                        and "additionalProperties" not in tool_params
                    ):
                        tool_params["additionalProperties"] = False

                    # Create function definition for dictionary tools
                    dict_function_def: Dict[str, Any] = {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": tool_params,
                        "strict": True,
                    }

                    # Create OpenAI tool with this function definition
                    dict_openai_tool: Dict[str, Any] = {
                        "type": "function",
                        "function": dict_function_def,
                    }

                    # Set the common openai_tool variable that we'll add to the list
                    openai_tool = dict_openai_tool

                # Add the formatted tool to the list
                # Use the tool we created regardless of which branch (callable or dict) we came from
                formatted_tools.append(openai_tool)

            except LLMError:
                # Let our custom exceptions propagate
                raise
            except Exception as e:
                # Wrap other errors with proper context
                raise LLMFormatError(
                    message=f"Failed to format tool for OpenAI: {e}",
                    provider="openai",
                    details={"error": str(e), "tool": str(tool)},
                )

        # Return the list as List[Dict[str, Any]] to satisfy parent class expectations
        return formatted_tools

    # Custom Helper Functions (Overwrites LLM Parent Functions)
    def supports_image_input(self) -> bool:
        """
        Check if the current model supports image input.

        All OpenAI models in this implementation support images.

        Returns:
            bool: True for all OpenAI models
        """
        return self.SUPPORTS_IMAGES

    def calculate_image_tokens(
        self, width: int, height: int, detail: str = "high"
    ) -> int:
        """
        Calculate the number of tokens required for an image based on its dimensions and detail level.

        OpenAI has a specific token charging model for images that depends on:
        1. Detail level (low or high)
        2. Image dimensions (which determine number of tiles)

        This method implements OpenAI's token calculation algorithm for images.

        Args:
            width: Original image width in pixels
            height: Original image height in pixels
            detail: Either "high" or "low" detail mode

        Returns:
            int: Number of tokens required for the image
        """
        if detail.lower() == "low":
            return self.LOW_DETAIL_TOKEN_COST

        # High detail mode calculations
        # Step 1: Scale to fit within MAX_IMAGE_SIZE x MAX_IMAGE_SIZE
        scale = min(self.MAX_IMAGE_SIZE / max(width, height), 1.0)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 2: Scale so shortest side is TARGET_SHORT_SIDE
        short_side = min(scaled_width, scaled_height)
        if short_side > self.TARGET_SHORT_SIDE:
            scale = self.TARGET_SHORT_SIDE / short_side
            scaled_width = int(scaled_width * scale)
            scaled_height = int(scaled_height * scale)

        # Step 3: Calculate number of 512px tiles needed
        tiles_width = ceil(scaled_width / self.TILE_SIZE)
        tiles_height = ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_width * tiles_height

        # Step 4: Calculate total tokens
        return (total_tiles * self.HIGH_DETAIL_TILE_COST) + self.HIGH_DETAIL_BASE_COST

    def estimate_cost(
        self,
        prompt: str,
        expected_response_tokens: Optional[int] = None,
        images: Optional[List[Tuple[int, int, str]]] = None,
        num_images: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate the cost for a generation based on input and expected output.

        This method provides a detailed cost breakdown for text, images, and response.
        It's particularly useful for OpenAI models where image costs can vary significantly
        based on dimensions and detail level.

        Args:
            prompt: The input prompt text
            expected_response_tokens: Expected response length in tokens
            images: List of (width, height, detail) tuples for each image
            num_images: Simple count of images (used if images parameter not provided)

        Returns:
            Tuple containing:
            - Total estimated cost
            - Detailed cost breakdown dictionary
        """
        costs = self.get_model_costs()
        prompt_tokens = self.estimate_tokens(prompt)
        response_tokens = expected_response_tokens or prompt_tokens

        # Calculate text costs
        read_cost = costs.get("read_token", 0)
        write_cost = costs.get("write_token", 0)

        # Handle potential None values
        if read_cost is None:
            read_cost = 0
        if write_cost is None:
            write_cost = 0

        prompt_cost = prompt_tokens * read_cost
        response_cost = response_tokens * write_cost

        # Calculate image costs
        image_tokens = 0
        if images:
            # Use detailed image information if provided
            for width, height, detail in images:
                image_tokens += self.calculate_image_tokens(width, height, detail)
        else:
            # Fall back to simple count with default settings
            image_count = num_images or 0
            for _ in range(image_count):
                image_tokens += self.calculate_image_tokens(
                    1024, 1024, "high"
                )  # Default to high quality 1024x1024

        # Get read token cost, defaulting to 0 if None
        image_token_cost = costs.get("read_token", 0)
        if image_token_cost is None:
            image_token_cost = 0

        image_cost = (
            image_tokens * image_token_cost
        )  # Images are charged at input token rate

        cost_breakdown = {
            "prompt_cost": round(prompt_cost, 6),
            "response_cost": round(response_cost, 6),
            "image_cost": round(image_cost, 6),
            "image_tokens": image_tokens,
        }
        total_cost = sum(
            value for key, value in cost_breakdown.items() if key != "image_tokens"
        )

        return round(total_cost, 6), cost_breakdown

    # OpenAI Specific Implementation Classes
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
        Generate a response using OpenAI's API.

        This method handles:
        1. Message formatting using OpenAI's message structure
        2. Adding system prompt as a system message
        3. Configuring special parameters for reasoning models
        4. Making the API request via the OpenAI client
        5. Extracting and processing the response
        6. Calculating token usage and costs
        7. Extracting any function/tool calls from the response

        Args:
            event_id: Unique identifier for this generation event
            system_prompt: System-level instructions for the model
            messages: List of messages in the standard format
            max_tokens: Maximum number of tokens to generate
            temp: Temperature for generation (not used for some models)
            tools: Optional list of function/tool definitions

        Returns:
            Tuple containing:
            - Generated text response
            - Usage statistics (tokens, costs, tool use)

        Raises:
            LLMError: With appropriate error type based on the issue
        """
        try:
            # Format messages for OpenAI's API
            try:
                formatted_messages = self._format_messages_for_model(messages)
            except LLMError:
                # Let our custom exceptions propagate
                raise
            except Exception as e:
                # Wrap other formatting errors
                raise LLMFormatError(
                    message=f"Failed to format messages for OpenAI: {e!s}",
                    provider="openai",
                    details={"error": str(e)},
                )

            # Add system message at the start if provided
            if system_prompt:
                formatted_messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                )

            # Prepare API call parameters based on model type and tools
            try:
                # Create API parameters with common values first
                # Explicitly type annotate for mypy
                create_params: Dict[str, Any] = {
                    "model": self.model_name,
                    "messages": formatted_messages,
                }

                # Add parameters based on model type
                if self.model_name in self.REASONING_MODELS:
                    # Special handling for reasoning models (o1, o3-mini)
                    create_params["reasoning_effort"] = (
                        "high"  # Enable enhanced reasoning
                    )
                else:
                    # Standard handling for non-reasoning models
                    create_params["max_completion_tokens"] = max_tokens

                # Add tools if provided
                if tools:
                    formatted_tools = self._format_tools_for_model(tools)
                    create_params["tools"] = formatted_tools

                # Set temperature if not using default
                if temp != 0.0:
                    create_params["temperature"] = temp

                # Import OpenAI types for properly typed API calls
                from openai.types.chat import ChatCompletionMessageParam
                from openai.types.chat.chat_completion_tool_param import (
                    ChatCompletionToolParam,
                )

                # Our messages are already properly typed with OpenAIMessage,
                # but OpenAI's API expects ChatCompletionMessageParam
                # We'll use the cast operator to satisfy the type checker
                api_messages = cast(
                    List[ChatCompletionMessageParam], formatted_messages
                )
                create_params["messages"] = api_messages

                # Handle tools parameter similar to messages
                if "tools" in create_params:
                    api_tools = cast(List[ChatCompletionToolParam], formatted_tools)
                    create_params["tools"] = api_tools

                # Make the API call with retry logic
                response = self._call_with_retry(
                    self.client.chat.completions.create, **create_params
                )

            except LLMError:
                # Let our custom exceptions propagate
                raise
            except Exception as e:
                # Map other API errors
                raise self._map_openai_error(e)

            # Process the response
            try:
                # Extract response text
                if not response.choices:
                    raise LLMContentError(
                        message="OpenAI returned empty choices array",
                        provider="openai",
                        details={"response": str(response)},
                    )

                response_text = response.choices[0].message.content

                # Handle potential null response (happens with tool calls)
                if response_text is None:
                    # Check if there are tool calls
                    tool_calls = response.choices[0].message.tool_calls
                    if tool_calls:
                        # Return empty string as the tool calls are handled separately
                        response_text = ""
                    else:
                        # Empty response without tool calls is an error
                        raise LLMContentError(
                            message="OpenAI returned empty response with no tool calls",
                            provider="openai",
                            details={"response": str(response)},
                        )
            except LLMError:
                # Let our custom exceptions propagate
                raise
            except Exception as e:
                # Wrap other response parsing errors
                raise LLMFormatError(
                    message=f"Failed to parse response from OpenAI: {e!s}",
                    provider="openai",
                    details={"error": str(e), "response": str(response)},
                )

            # Calculate usage statistics
            try:
                # Get usage information
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens

                # Count images in input messages
                num_images = sum(
                    len(msg.get("image_paths", [])) + len(msg.get("image_urls", []))
                    for msg in messages
                )

                # Calculate costs based on token usage
                costs = self.get_model_costs()

                # Calculate individual costs with proper null checking
                read_token_cost = float(costs.get("read_token", 0.0) or 0.0)
                write_token_cost = float(costs.get("write_token", 0.0) or 0.0)
                image_token_cost = float(costs.get("image_cost", 0.0) or 0.0)

                # Calculate costs safely
                read_cost = float(input_tokens) * read_token_cost
                write_cost = float(output_tokens) * write_token_cost
                # Note: Real image cost is more complex and depends on size/detail
                image_cost = float(num_images) * image_token_cost

                # Prepare comprehensive usage statistics
                usage_stats = {
                    "read_tokens": input_tokens,
                    "write_tokens": output_tokens,
                    "images": num_images,
                    "total_tokens": total_tokens,
                    "read_cost": round(read_cost, 6),
                    "write_cost": round(write_cost, 6),
                    "image_cost": round(image_cost, 6),
                    "total_cost": round(read_cost + write_cost + image_cost, 6),
                }
            except Exception:
                # Non-fatal: use defaults if usage calculation fails
                usage_stats = {
                    "read_tokens": len(str(formatted_messages)) // 4,
                    # Approximate
                    "write_tokens": len(response_text) // 4,  # Approximate
                    "total_tokens": (len(str(formatted_messages)) + len(response_text))
                    // 4,
                    "images": 0,
                    "read_cost": 0,
                    "write_cost": 0,
                    "image_cost": 0,
                    "total_cost": 0,
                }

            # Extract function/tool calls from the response
            try:
                tool_use = response.choices[0].message.tool_calls
                if tool_use:
                    # Try to parse arguments as JSON, fallback to raw string if needed
                    try:
                        tool_input = json.loads(tool_use[0].function.arguments)
                    except json.JSONDecodeError as json_error:
                        # Specific error for invalid JSON in tool arguments
                        if "tool_use" in usage_stats:
                            raise LLMToolError(
                                message="Invalid JSON in tool arguments",
                                provider="openai",
                                details={
                                    "error": str(json_error),
                                    "arguments": tool_use[0].function.arguments,
                                },
                            )
                        tool_input = tool_use[0].function.arguments

                    # Add tool use information to usage stats
                    usage_stats["tool_use"] = {
                        "id": tool_use[0].id,
                        "name": tool_use[0].function.name,
                        "input": tool_input,
                    }
            except LLMError:
                # Re-raise tool errors
                raise
            except Exception:
                # Ignore other tool extraction errors (non-critical)
                pass

            return response_text, usage_stats

        except LLMError:
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise LLMProviderError(
                message="Unexpected error in OpenAI generation",
                provider="openai",
                details={"error": str(e)},
            )

    def get_supported_models(self) -> list[str]:
        """
        Get the list of supported model identifiers.

        Returns:
            list[str]: List of supported model identifiers
        """
        return self.SUPPORTED_MODELS

    def embed(
        self, texts: List[str], model: Optional[str] = None, batch_size: int = 100
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Get embeddings for a list of texts using OpenAI's embedding models.

        Args:
            texts (List[str]): List of texts to embed
            model (Optional[str]): Specific embedding model to use
            batch_size (int): Number of texts to process in each batch

        Returns:
            Tuple[List[List[float]], Dict[str, Any]]: Tuple containing:
                - List of embedding vectors (one per input text)
                - Usage statistics

        Raises:
            ValueError: If embedding fails
        """
        # Import OpenAI here to avoid import errors if not installed
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with 'pip install openai'"
            )

        # Use specified model or default
        embedding_model = model or self.DEFAULT_EMBEDDING_MODEL

        # Validate that the model is supported
        if embedding_model not in self.EMBEDDING_MODELS:
            raise ValueError(
                f"Embedding model {embedding_model} not supported. Supported models: {self.EMBEDDING_MODELS}"
            )

        # Initialize client if needed
        if not hasattr(self, "client"):
            self.client = OpenAI(api_key=self.config["api_key"], base_url=self.api_base)

        # Create batches to avoid rate limits and large requests
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        all_embeddings = []
        total_tokens = 0

        try:
            for batch in batches:
                response = self.client.embeddings.create(
                    model=embedding_model, input=batch, encoding_format="float"
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                total_tokens += response.usage.total_tokens

                # Optional: add a small delay between batches to avoid rate limits
                # import time
                # time.sleep(0.1)

        except Exception as e:
            raise ValueError(f"Error getting embeddings from OpenAI: {e!s}")

        # Calculate cost
        cost_per_token = self.embedding_costs.get(
            embedding_model, 0.0001
        )  # Default if unknown
        total_cost = total_tokens * cost_per_token

        # Return embeddings and usage data
        usage = {"tokens": total_tokens, "cost": total_cost, "model": embedding_model}

        return all_embeddings, usage

    def is_thinking_model(self, model_name: str) -> bool:
        """
        Check if a model supports thinking/reasoning capabilities.

        Args:
            model_name: The model name to check

        Returns:
            bool: True if the model supports thinking capabilities, False otherwise
        """
        return model_name in self.THINKING_MODELS

    def _map_openai_error(self, error: Exception) -> LLMError:
        """
        Map OpenAI-specific errors to standardized LLM error types.

        This method analyzes an exception from the OpenAI API and converts it
        to the appropriate LLM exception type with relevant context.

        Args:
            error: The original OpenAI exception

        Returns:
            LLMError: The mapped exception with detailed context
        """
        import openai

        error_str = str(error).lower()

        # Authentication errors
        if isinstance(error, openai.AuthenticationError):
            return LLMAuthenticationError(
                message="OpenAI authentication failed: Invalid API key",
                provider="openai",
                details={"error": str(error)},
            )

        # Rate limiting errors
        if isinstance(error, openai.RateLimitError):
            # Try to extract retry-after header if available
            retry_after = None
            if hasattr(error, "headers") and error.headers:
                retry_after = error.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    retry_after = int(retry_after)

            return LLMRateLimitError(
                message="OpenAI rate limit exceeded",
                provider="openai",
                retry_after=retry_after,
                details={"error": str(error)},
            )

        # Validation errors
        if isinstance(error, openai.BadRequestError):
            if (
                "context_length_exceeded" in error_str
                or "maximum context length" in error_str
            ):
                return LLMProviderError(
                    message="OpenAI model context length exceeded",
                    provider="openai",
                    details={
                        "error": str(error),
                        "model": self.model_name,
                        "context_window": self.get_context_window(),
                    },
                )
            elif (
                "content_policy_violation" in error_str or "content policy" in error_str
            ):
                return LLMContentError(
                    message="OpenAI content policy violation",
                    provider="openai",
                    details={"error": str(error)},
                )
            elif "invalid_api_param" in error_str:
                return LLMConfigurationError(
                    message="Invalid parameter for OpenAI API",
                    provider="openai",
                    details={"error": str(error)},
                )

            # Default validation error
            return LLMConfigurationError(
                message="OpenAI API validation error",
                provider="openai",
                details={"error": str(error)},
            )

        # Server errors (5xx)
        if isinstance(error, openai.APIError):
            if (
                "server_error" in error_str
                or hasattr(error, "status_code")
                and str(getattr(error, "status_code", "")).startswith("5")
            ):
                return LLMServiceUnavailableError(
                    message="OpenAI API server error",
                    provider="openai",
                    details={"error": str(error)},
                )

            # General API error
            return LLMProviderError(
                message="OpenAI API error",
                provider="openai",
                details={"error": str(error)},
            )

        # Connection errors
        if isinstance(error, openai.APIConnectionError):
            return LLMProviderError(
                message="OpenAI API connection error",
                provider="openai",
                details={"error": str(error)},
            )

        # Timeout errors
        if isinstance(error, openai.APITimeoutError):
            return LLMProviderError(
                message="OpenAI API timeout",
                provider="openai",
                details={"error": str(error)},
            )

        # Tool/function related errors
        if "function" in error_str or "tool" in error_str or "parameter" in error_str:
            return LLMToolError(
                message="OpenAI tool/function error",
                provider="openai",
                details={"error": str(error)},
            )

        # Default fallback
        return LLMProviderError(
            message=f"OpenAI API error: {error!s}",
            provider="openai",
            details={"error": str(error)},
        )

    def supports_embeddings(self) -> bool:
        """
        Check if this provider supports embeddings.

        Returns:
            bool: True, as OpenAI has embedding models
        """
        return True

    def _call_with_retry(
        self,
        func: Callable,
        *args,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        retryable_errors: Optional[List[Type[Exception]]] = None,
        **kwargs,
    ) -> Any:
        """
        Call an OpenAI API function with exponential backoff retry logic.

        This method implements a retry mechanism for handling transient errors
        such as rate limits and server errors. It uses exponential backoff
        to avoid overwhelming the API during retries.

        Args:
            func: The API function to call
            *args: Positional arguments to pass to the function
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Multiplier for backoff on each retry
            retryable_errors: List of exception types that should trigger a retry
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The response from the API function

        Raises:
            LLuMinaryError: Mapped exception based on the error type
        """
        import openai

        # Default retryable errors if not specified
        if retryable_errors is None:
            retryable_errors = [
                openai.RateLimitError,
                openai.APIError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ]

        retry_count = 0
        backoff_time = initial_backoff
        last_exception = None

        while retry_count <= max_retries:  # <= because first attempt is not a retry
            try:
                return func(*args, **kwargs)

            except Exception as e:
                # Check if this is a retryable error type
                if not any(isinstance(e, err_type) for err_type in retryable_errors):
                    # Non-retryable error, map to appropriate exception type and raise
                    raise self._map_openai_error(e)

                # It's a retryable error type
                retry_count += 1
                last_exception = e

                # Check if we've exhausted retries
                if retry_count > max_retries:
                    # We've exhausted retries, map and raise the error
                    raise self._map_openai_error(e)

                # We have retries left, continue with backoff logic

                # Get retry-after time if available (for rate limits)
                retry_after = None
                if (
                    isinstance(e, openai.RateLimitError)
                    and hasattr(e, "headers")
                    and e.headers
                ):
                    retry_after_header = e.headers.get("retry-after")
                    if retry_after_header and retry_after_header.isdigit():
                        retry_after = float(retry_after_header)

                # Use retry-after if available, otherwise use exponential backoff
                wait_time = retry_after if retry_after else backoff_time

                # Sleep before retry
                time.sleep(wait_time)

                # Increase backoff for next retry
                backoff_time *= backoff_factor
                last_exception = e

            except Exception as e:
                # Non-retryable error, map to appropriate exception type and raise
                raise self._map_openai_error(e)

        # Should never get here, but just in case
        if last_exception:
            raise self._map_openai_error(last_exception)

        # Generic fallback error
        raise LLMProviderError(
            message=f"Failed after {max_retries} retries", provider="openai"
        )

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
        Stream a response from the OpenAI API, yielding chunks as they become available.

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
            Exception: If streaming fails
        """
        # Import OpenAI here to avoid import errors if not installed
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with 'pip install openai'"
            )

        # Initialize client if needed
        if not hasattr(self, "client") or self.client is None:
            self.auth()

        # Convert messages to OpenAI format
        formatted_messages = self._format_messages_for_model(messages)

        # Add system message
        if system_prompt:
            formatted_messages.insert(0, {"role": "system", "content": system_prompt})

        # Count images for cost calculation
        image_count = 0
        for message in messages:
            if message.get("message_type") == "human":
                image_count += len(message.get("image_paths", []))
                image_count += len(message.get("image_urls", []))

        # Prepare tools parameter if functions are provided
        tools = None
        if functions:
            tools = [self._convert_function_to_tool(func) for func in functions]

        try:
            # Create a streaming request using properly typed structures
            from typing import cast

            from openai.types.chat import ChatCompletionMessageParam
            from openai.types.chat.chat_completion_tool_param import (
                ChatCompletionToolParam,
            )

            # Format messages for API
            # Our messages are already properly typed with OpenAIMessage,
            # but OpenAI's API expects ChatCompletionMessageParam
            api_messages = cast(List[ChatCompletionMessageParam], formatted_messages)

            # Prepare tools if provided
            formatted_tools: Optional[List[Dict[str, Any]]] = None
            if functions:
                # Convert Python functions to tool format
                function_tools = [
                    self._convert_function_to_tool(func) for func in functions
                ]
                formatted_tools = self._format_tools_for_model(function_tools)

            # Set up the API call based on whether tools are provided
            if formatted_tools:
                # Cast tools to OpenAI's expected type
                api_tools = cast(List[ChatCompletionToolParam], formatted_tools)

                # Call with tools parameter
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    tools=api_tools,
                    stream=True,
                )
            else:
                # Call without tools parameter
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    stream=True,
                )

            # Initialize variables to accumulate data
            accumulated_text = ""
            accumulated_tokens = 0
            tool_call_data: Dict[str, Dict[str, str]] = {}

            # Track tokens
            input_tokens = self._count_tokens_from_messages(formatted_messages)

            # Process each chunk as it arrives
            for chunk in response:
                # Handle typing issues with the chunk
                from openai.types.chat import ChatCompletionChunk

                # Skip if not a proper ChatCompletionChunk with choices
                if (
                    not isinstance(chunk, ChatCompletionChunk)
                    or not hasattr(chunk, "choices")
                    or not chunk.choices
                ):
                    continue

                # Now we're sure chunk is a ChatCompletionChunk with choices attribute
                delta = chunk.choices[0].delta

                # Extract content if available
                if hasattr(delta, "content") and delta.content is not None:
                    # Safely handle content that could be None but we've already checked above
                    content_str = str(delta.content)
                    accumulated_text += content_str
                    accumulated_tokens += 1  # Approximate token count

                    # Create usage data for this chunk
                    partial_usage = {
                        "event_id": event_id,
                        "model": self.model_name,
                        "read_tokens": input_tokens,
                        "write_tokens": accumulated_tokens,
                        "images": image_count,
                        "total_tokens": input_tokens + accumulated_tokens,
                        "is_complete": False,
                        "tool_use": tool_call_data,
                    }

                    # Call the callback if provided
                    if callback:
                        callback(content_str, partial_usage)

                    # Yield the content chunk and partial usage data
                    yield content_str, partial_usage

                # Extract tool calls if available
                if hasattr(delta, "tool_calls") and delta.tool_calls is not None:
                    # Make sure tool_calls is not None before iterating
                    tool_calls = delta.tool_calls or []

                    for tool_call in tool_calls:
                        # Skip if tool_call doesn't have id attribute or id is None
                        if not hasattr(tool_call, "id") or tool_call.id is None:
                            continue

                        # Get the tool ID, which we know is not None due to check above
                        tool_id = str(tool_call.id)

                        # Initialize tool call data if not seen before
                        if tool_id not in tool_call_data:
                            # Get function name with proper type safety
                            function_name = ""
                            if (
                                hasattr(tool_call, "function")
                                and tool_call.function is not None
                                and hasattr(tool_call.function, "name")
                                and tool_call.function.name is not None
                            ):
                                function_name = str(tool_call.function.name)

                            # Create a typed tool call entry
                            tool_call_data[tool_id] = {
                                "id": tool_id,
                                "name": function_name,
                                "arguments": "",
                                "type": "function",
                            }

                        # Extract arguments with proper null checks
                        function_args = ""
                        if (
                            hasattr(tool_call, "function")
                            and tool_call.function is not None
                            and hasattr(tool_call.function, "arguments")
                            and tool_call.function.arguments is not None
                        ):
                            function_args = str(tool_call.function.arguments)

                        # Append arguments if we have a valid ID
                        if tool_id in tool_call_data:
                            # Use safe concatenation with an explicit type check
                            current_args = tool_call_data[tool_id].get("arguments", "")
                            tool_call_data[tool_id]["arguments"] = (
                                current_args + function_args
                            )

            # Calculate costs
            model_costs = self.get_model_costs()

            # Handle possible None values with explicit defaults
            read_token_cost = float(model_costs.get("read_token", 0.0) or 0.0)
            write_token_cost = float(model_costs.get("write_token", 0.0) or 0.0)
            image_token_cost = float(model_costs.get("image_token", 0.0) or 0.0)

            # Calculate costs with proper type handling
            read_cost = float(input_tokens) * read_token_cost
            write_cost = float(accumulated_tokens) * write_token_cost
            image_cost = 0.0

            if image_count > 0 and "image_token" in model_costs:
                image_cost = float(image_count) * image_token_cost

            total_cost = read_cost + write_cost + image_cost

            # Create final usage data
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
            }

            # Call the callback with an empty string to signal completion
            if callback:
                callback("", final_usage)

            # Yield an empty string with the final usage data to signal completion
            yield "", final_usage

        except Exception as e:
            raise Exception(f"Error streaming from OpenAI: {e!s}")

    def _count_tokens_from_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        Estimate the number of tokens in the messages.
        This is a simplified approximation and may not be exact.

        Args:
            messages: List of messages in OpenAI format

        Returns:
            int: Estimated token count
        """
        # Simple estimation based on characters
        text = ""
        for message in messages:
            # Extract content safely with type checking
            content = message.get("content", "")

            if isinstance(content, str):
                # For string content, add it directly
                text += content
            elif isinstance(content, list):
                # For content lists (multipart messages), extract text from each part
                for item in content:
                    if isinstance(item, dict):
                        # Handle text parts
                        if item.get("type") == "text" and "text" in item:
                            item_text = item.get("text")
                            if item_text is not None:
                                text += str(item_text)

                        # We don't count image tokens here as they are handled separately

        # Rough approximation: 4 characters per token
        return len(text) // 4

    # Add the rerank method after the supports_embeddings method
    def supports_reranking(self) -> bool:
        """
        Check if this provider and model supports document reranking.

        Returns:
            bool: True if reranking is supported, False otherwise
        """
        return (
            len(self.RERANKING_MODELS) > 0 and self.model_name in self.RERANKING_MODELS
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_scores: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Rerank documents using OpenAI's reranking capabilities.

        Args:
            query (str): The search query to rank documents against
            documents (List[str]): List of document texts to rerank
            top_n (int, optional): Number of top documents to return
            return_scores (bool): Whether to include relevance scores
            **kwargs: Additional provider-specific parameters
                - model (str, optional): Specific reranking model to use

        Returns:
            Dict[str, Any]: Dictionary with ranked documents, indices, scores, and usage info
        """
        if not self.supports_reranking():
            raise NotImplementedError(
                f"Model {self.model_name} does not support document reranking. "
                f"Available reranking models: {self.RERANKING_MODELS}"
            )

        if not documents:
            return {
                "ranked_documents": [],
                "indices": [],
                "scores": [] if return_scores else None,
                "usage": {"total_tokens": 0, "total_cost": 0.0},
            }

        # Import OpenAI here to avoid import errors if not installed
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with 'pip install openai'"
            )

        # Use specified model or default
        rerank_model = kwargs.get("model", self.DEFAULT_RERANKING_MODEL)

        # Validate that the model is supported
        if rerank_model not in self.RERANKING_MODELS:
            raise ValueError(
                f"Reranking model {rerank_model} not supported. Supported models: {self.RERANKING_MODELS}"
            )

        # Initialize client if needed
        if not hasattr(self, "client"):
            self.client = OpenAI(api_key=self.config["api_key"], base_url=self.api_base)

        try:
            # Call OpenAI's reranking endpoint
            response = self.client.embeddings.create(
                model=rerank_model,
                input=documents,
                encoding_format="float",
                dimensions=1536,
                # Standard dimension for text-embedding-3 models
            )

            # Get document embeddings
            document_embeddings = [item.embedding for item in response.data]

            # Get query embedding
            query_response = self.client.embeddings.create(
                model=rerank_model,
                input=[query],
                encoding_format="float",
                dimensions=1536,
            )
            query_embedding = query_response.data[0].embedding

            # Calculate cosine similarity scores
            def cosine_similarity(vec1, vec2):
                dot_product = sum(a * b for a, b in zip(vec1, vec2))
                magnitude1 = sum(a * a for a in vec1) ** 0.5
                magnitude2 = sum(b * b for b in vec2) ** 0.5
                if magnitude1 * magnitude2 == 0:
                    return 0
                return dot_product / (magnitude1 * magnitude2)

            # Calculate similarity for each document
            similarities = [
                cosine_similarity(query_embedding, doc_embedding)
                for doc_embedding in document_embeddings
            ]

            # Create (index, similarity) pairs and sort by similarity (highest first)
            ranked_pairs = sorted(
                enumerate(similarities), key=lambda x: x[1], reverse=True
            )

            # Get indices of ranked documents
            ranked_indices = [idx for idx, _ in ranked_pairs]

            # Limit to top_n if specified
            if top_n is not None:
                ranked_indices = ranked_indices[:top_n]
                ranked_pairs = ranked_pairs[:top_n]

            # Get ranked documents
            ranked_documents = [documents[idx] for idx in ranked_indices]

            # Get scores if requested
            scores = [score for _, score in ranked_pairs] if return_scores else None

            # Calculate token usage and cost
            total_tokens = (
                response.usage.total_tokens + query_response.usage.total_tokens
            )
            cost_per_token = self.reranking_costs.get(
                rerank_model, 0.00002
            )  # Default if unknown
            total_cost = total_tokens * cost_per_token

            return {
                "ranked_documents": ranked_documents,
                "indices": ranked_indices,
                "scores": scores,
                "usage": {
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                    "model": rerank_model,
                },
            }

        except Exception as e:
            raise ValueError(f"Error reranking documents with OpenAI: {e!s}")

    def generate_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        n: int = 1,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        response_format: str = "url",
        return_usage: bool = False,
        **kwargs,
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], Dict[str, Any]]]:
        """
        Generate images using OpenAI's DALL-E models.

        Args:
            prompt (str): Text description of the desired image
            model (str): The DALL-E model to use (dall-e-3 or dall-e-2)
            n (int): Number of images to generate
            size (str): Image size (1024x1024, 1024x1792, 1792x1024, 512x512, 256x256)
            quality (str): Image quality (standard or hd for dall-e-3)
            style (str): Image style (vivid or natural for dall-e-3)
            response_format (str): Format of response (url or b64_json)
            return_usage (bool): Whether to return usage statistics
            **kwargs: Additional parameters to pass to the API

        Returns:
            List[Dict[str, str]]: List of generated images with URLs or base64 data
            Dict[str, Any]: Usage statistics (if return_usage=True)
        """
        # Import OpenAI here to avoid import errors if not installed
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with 'pip install openai'"
            )

        # Initialize client if needed
        if not hasattr(self, "client"):
            self.client = OpenAI(api_key=self.config["api_key"], base_url=self.api_base)

        try:
            # Import the required Literal type and cast parameters
            from typing import Literal, cast

            # Cast string parameters to their expected Literal types
            size_param = cast(
                Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"],
                size,
            )
            quality_param = cast(Literal["standard", "hd"], quality)
            style_param = cast(Literal["vivid", "natural"], style)
            response_format_param = cast(Literal["url", "b64_json"], response_format)

            # Call OpenAI's image generation API
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                n=n,
                size=size_param,
                quality=quality_param,
                style=style_param,
                response_format=response_format_param,
                **kwargs,
            )

            # Process the response
            images = []
            for data in response.data:
                if hasattr(data, "url") and data.url:
                    images.append({"url": data.url})
                elif hasattr(data, "b64_json") and data.b64_json:
                    images.append({"data": data.b64_json})

            # Calculate usage statistics if requested
            if return_usage:
                # Get cost per image based on model and size
                cost_per_image = self.IMAGE_GENERATION_COSTS.get(model, {}).get(
                    size, 0.04
                )
                total_cost = cost_per_image * n

                usage_stats = {
                    "model": model,
                    "count": n,
                    "size": size,
                    "cost": total_cost,
                }
                return images, usage_stats

            return images

        except Exception as e:
            raise Exception(f"Error generating image with OpenAI: {e!s}")

    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """
        Validate OpenAI-specific configuration.

        Args:
            config: Provider configuration dictionary

        Raises:
            LLMValidationError: If configuration is invalid
        """
        # Check for required API key
        if "api_key" not in config and "use_aws_secrets" not in config:
            raise LLMValidationError(
                "Either 'api_key' or 'use_aws_secrets' must be provided for OpenAI provider",
                details={"missing_fields": ["api_key or use_aws_secrets"]},
            )

        # Validate model
        if self.model_name not in self.SUPPORTED_MODELS:
            raise LLMValidationError(
                f"Unsupported model: {self.model_name}",
                details={
                    "model": self.model_name,
                    "supported_models": self.SUPPORTED_MODELS,
                },
            )

        # If timeout is specified, ensure it's a number
        if "timeout" in config and not isinstance(config["timeout"], (int, float)):
            raise LLMValidationError(
                f"Invalid timeout: {config['timeout']}",
                details={"timeout": config["timeout"]},
            )
