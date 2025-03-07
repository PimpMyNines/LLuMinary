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
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

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
)
from ...utils.aws import get_secret
from ..base import LLM


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
        super().__init__(model_name, **kwargs)
        self.api_base = kwargs.get("api_base", None)
        self.timeout = kwargs.get("timeout", 60)

        # Store embedding costs for different models
        self.embedding_costs = {
            "text-embedding-ada-002": 0.0001,  # $0.10 per million tokens
            "text-embedding-3-small": 0.00002,  # $0.02 per million tokens
            "text-embedding-3-large": 0.00013,  # $0.13 per million tokens
        }

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
            List[Dict[str, Any]]: Messages formatted for OpenAI API

        Raises:
            Exception: If message formatting fails, particularly during image processing
        """
        formatted_messages = []
        for message in messages:
            # To handle the dual type nature, we'll maintain a flag for type
            is_content_string = False
            content_parts: List[Dict[str, Any]] = []
            content_text = ""
            tool_calls = []
            content: Union[str, List[Dict[str, Any]]]

            # Process local image files
            for image_path in message.get("image_paths", []):
                try:
                    # Convert image to base64 and format as image_url
                    image_base64 = self._encode_image(image_path)
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        }
                    )
                except Exception as e:
                    raise Exception(f"Failed to process image file {image_path}: {e!s}")

            # Process image URLs
            for image_url in message.get("image_urls", []):
                try:
                    # Download, convert to base64, and format as image_url
                    image_base64 = self._encode_image_url(image_url)
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        }
                    )
                except Exception as e:
                    raise Exception(f"Failed to process image URL {image_url}: {e!s}")

            # Add text content, handling different content formats
            if len(content_parts) > 0:
                # If we already have image content parts, add text as another part
                if message.get("message"):
                    content_parts.append({"type": "text", "text": message["message"]})
                # Use the content parts format
                content_to_use = content_parts
            else:
                # Simple case: just text, no need for content parts array
                if message.get("message"):
                    content_text = message["message"]
                    is_content_string = True
                else:
                    # Empty message, default to empty string
                    content_text = ""
                    is_content_string = True
                # Don't assign to content_to_use since we're using the if/else below

            # Final content to use in message
            # Cast depending on what type of content we have
            if is_content_string:
                content = content_text  # Use string type
            else:
                content = content_parts  # Use content parts list

            # Map message types to OpenAI roles
            # Four primary roles: user, assistant, system, tool
            role = "user"  # Default role
            if message["message_type"] == "ai":
                role = "assistant"
                # Process tool use for assistant messages
                if message.get("tool_use"):
                    tool_calls.append(
                        {
                            "type": "function",
                            "id": message["tool_use"]["id"],
                            "function": {
                                "name": message["tool_use"]["name"],
                                "arguments": json.dumps(message["tool_use"]["input"]),
                            },
                        }
                    )
            elif message["message_type"] == "system":
                role = "system"
            elif message["message_type"] == "developer":
                role = "system"  # Map developer to system for backward compatibility
            elif message["message_type"] == "tool_result":
                # Handle tool response messages differently
                role = "tool"
                # Format result message based on success/failure
                result = ""
                if message["tool_result"].get("success"):
                    result = "Tool Call Successful: " + message["tool_result"]["result"]
                else:
                    result = "Tool Call Failed: " + message["tool_result"]["error"]
                formatted_message = {
                    "role": role,
                    "tool_call_id": message["tool_result"]["tool_id"],
                    "content": result,
                }

            # Build the formatted message based on role
            if role != "tool":
                # Standard message format for user, assistant, system
                if content:
                    # Handle different content types properly
                    if (
                        isinstance(content, list)
                        and len(content) == 1
                        and isinstance(content[0], dict)
                        and "text" in content[0]
                    ):
                        # For a single content item, extract text value
                        message_content = content[0]["text"]
                    else:
                        # Use as is for other cases
                        message_content = content

                    formatted_message = {
                        "role": role,
                        "content": message_content,
                    }
                else:
                    formatted_message = {"role": role, "content": None}

                # Add tool_calls for assistant messages if needed
                if tool_calls:
                    formatted_message["tool_calls"] = tool_calls

            formatted_messages.append(formatted_message)

        return formatted_messages

    def _format_tools_for_model(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert standard tool format into OpenAI's format.

        OpenAI requires tools to be formatted in their function calling format,
        with a specific structure including function name, description, and parameters.

        Args:
            tools: List of tools in standard format or Python callables

        Returns:
            List[Dict[str, Any]]: Tools formatted for OpenAI API

        Raises:
            Exception: If tool processing fails
        """
        formatted_tools = []

        for tool in tools:
            # Handle Python function objects
            if callable(tool):
                # Convert callable to tool format using base method
                base_tool = self._convert_function_to_tool(tool)

                # Convert to OpenAI's function calling format
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": base_tool["name"],
                        "description": base_tool.get("description", ""),
                        "parameters": base_tool.get("input_schema", {}),
                        "strict": True,
                    },
                }
            else:
                # Handle dictionary specification
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                        "strict": True,
                    },
                }

            # Disable additional properties (enforce schema)
            # First make sure we have a dict with parameters
            function_dict = formatted_tool.get("function", {})
            if isinstance(function_dict, dict):
                parameters_dict = function_dict.get("parameters", {})
                if isinstance(parameters_dict, dict):
                    # Create a new dictionary with the additional property
                    updated_params = dict(parameters_dict)
                    updated_params["additionalProperties"] = False

                    # Create a new tool dictionary from scratch
                    mutable_formatted_tool = {}
                    for k, v in formatted_tool.items():
                        mutable_formatted_tool[k] = v

                    # Check if function already exists
                    if "function" in mutable_formatted_tool:
                        # Create a new function dictionary
                        function_part = {}
                        # Convert to dict if not already
                        function_dict = mutable_formatted_tool["function"]
                        if isinstance(function_dict, dict):
                            for k, v in function_dict.items():
                                function_part[k] = v
                        else:
                            # If it's not a dict, just create a minimal dict with parameters
                            function_part = {"parameters": updated_params}

                        # Replace parameters with our updated ones
                        function_part["parameters"] = updated_params
                        mutable_formatted_tool["function"] = function_part
                        formatted_tool = mutable_formatted_tool
            formatted_tools.append(formatted_tool)

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

                # Make the API call with retry logic
                from typing import cast

                from openai.types.chat import ChatCompletionMessageParam
                from openai.types.chat.chat_completion_tool_param import (
                    ChatCompletionToolParam,
                )

                # Cast the message type to satisfy OpenAI's API type requirements
                if "messages" in create_params:
                    create_params["messages"] = cast(
                        List[ChatCompletionMessageParam], create_params["messages"]
                    )

                # Cast the tools type if present and it's a list
                if create_params.get("tools") and isinstance(
                    create_params["tools"], list
                ):
                    create_params["tools"] = cast(
                        List[ChatCompletionToolParam], create_params["tools"]
                    )

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
            # Create a streaming request
            from typing import Iterable, cast

            from openai.types.chat import ChatCompletionMessageParam
            from openai.types.chat.chat_completion_tool_param import (
                ChatCompletionToolParam,
            )

            # Cast types to satisfy OpenAI's API typing requirements
            api_messages = cast(list[ChatCompletionMessageParam], formatted_messages)

            # Create API parameters dict to handle tools properly
            api_params = {
                "model": self.model_name,
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": temp,
                "stream": True,  # Enable streaming
            }

            # Only add tools if they exist
            if tools:
                # Ensure tools is iterable before casting
                if isinstance(tools, list):
                    api_params["tools"] = cast(Iterable[ChatCompletionToolParam], tools)

            # Call the API directly with individual parameters to satisfy typing
            # Create a properly cast tools parameter if tools exist
            if tools and isinstance(tools, list):
                typed_tools = cast(List[ChatCompletionToolParam], tools)
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=api_messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    tools=typed_tools,
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
            tool_call_data: Dict[str, Any] = {}

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
                    # Explicitly cast content to string
                    content_str = str(delta.content)
                    accumulated_text += content_str
                    accumulated_tokens += 1  # Approximate token count

                    # Call the callback if provided
                    if callback:
                        # Create partial usage data
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
                        callback(content_str, partial_usage)

                    # Yield the content chunk and partial usage data
                    yield content_str, {
                        "event_id": event_id,
                        "model": self.model_name,
                        "read_tokens": input_tokens,
                        "write_tokens": accumulated_tokens,
                        "images": image_count,
                        "total_tokens": input_tokens + accumulated_tokens,
                        "is_complete": False,
                        "tool_use": tool_call_data,
                    }

                # Extract tool calls if available
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        # Skip if tool_call doesn't have id attribute or id is None
                        if not hasattr(tool_call, "id") or tool_call.id is None:
                            continue

                        tool_id = tool_call.id

                        # Initialize tool call data if not seen before
                        if tool_id and tool_id not in tool_call_data:
                            function_name = ""
                            if (
                                hasattr(tool_call, "function")
                                and tool_call.function is not None
                                and hasattr(tool_call.function, "name")
                            ):
                                # Use explicit string cast to handle potential None
                                name_attr = tool_call.function.name
                                function_name = (
                                    str(name_attr) if name_attr is not None else ""
                                )

                            tool_call_data[tool_id] = {
                                "id": tool_id,
                                "name": function_name,
                                "arguments": "",
                                "type": "function",
                            }

                        # Safely get arguments with defensive checks
                        function_args = ""
                        if (
                            hasattr(tool_call, "function")
                            and tool_call.function is not None
                            and hasattr(tool_call.function, "arguments")
                        ):
                            # Handle potential None arguments
                            if tool_call.function.arguments is not None:
                                function_args = str(tool_call.function.arguments)

                        # Append arguments if we have a valid ID in our data structure
                        if tool_id in tool_call_data:
                            tool_call_data[tool_id]["arguments"] += function_args

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
            content = message.get("content", "")
            if isinstance(content, str):
                text += content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        item_text = item.get("text")
                        if item_text is not None:
                            text += str(item_text)

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
