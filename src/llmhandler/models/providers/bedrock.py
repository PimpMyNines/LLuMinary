"""
AWS Bedrock LLM provider implementation.

This module implements support for AWS Bedrock's models within the LLM handler framework.
It handles authentication, message formatting, image processing, and cost tracking for
AWS Bedrock models, primarily focusing on Claude models hosted on Bedrock.

## AWS Bedrock Messaging System Overview

AWS Bedrock uses a structured content format with specific requirements:

1. Messages are organized as an array with roles (user, assistant)
2. Each message contains "content" which is an array of content parts that can include:
   - Text (represented as simple text objects)
   - Images (represented as image objects with PNG format)
   - Tool use (represented as toolUse objects)
   - Tool results (represented as toolResult objects)
   - Reasoning content (represented as reasoningContent objects)

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

Into AWS Bedrock's format:
```
{
    "role": "user"|"assistant",
    "content": [
        {
            "text": "text content"
        },
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": raw_binary_data
                }
            }
        },
        {
            "toolUse": {
                "toolUseId": "id",
                "name": "func_name",
                "input": {...}
            }
        },
        {
            "toolResult": {
                "toolUseId": "id",
                "content": [{"text": "result"}],
                "status": "error"  # Optional, for errors only
            }
        },
        {
            "reasoningContent": {
                "reasoningText": {
                    "text": "thought process",
                    "signature": "signature"
                }
            }
        }
    ]
}
```

### Key Features
- Full support for text, images, and function calling
- Special support for Claude 3.7 thinking capabilities as reasoningContent
- PNG format with preserved transparency for images
- Smart retry mechanism for AWS API throttling
- Direct integration with AWS SDK for authentication and API calls
"""
import base64
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from ..base import LLM


class BedrockLLM(LLM):
    """
    Implementation of AWS Bedrock LLM models.

    This class provides methods for authentication, message formatting, image processing,
    and interaction with AWS Bedrock's API. It converts between the standardized
    LLM handler message format and Bedrock's specific structure.

    Attributes:
        THINKING_MODELS: List of models supporting the "thinking/reasoning" capability
        SUPPORTED_MODELS: List of Bedrock models supported by this implementation
        CONTEXT_WINDOW: Maximum token limits for each model
        COST_PER_MODEL: Cost information per token for each model
    """

    THINKING_MODELS = ["us.anthropic.claude-3-7-sonnet-20250219-v1:0"]

    SUPPORTED_MODELS = [
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    ]

    CONTEXT_WINDOW = {
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    }

    COST_PER_MODEL = {
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": {
            "read_token": 0.000001,
            "write_token": 0.000005,
            "image_cost": 0.024,  # All Bedrock Claude models support images
        },
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "read_token": 0.000003,
            "write_token": 0.000015,
            "image_cost": 0.024,
        },
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
            "read_token": 0.000003,
            "write_token": 0.000015,
            "image_cost": 0.024,
        },
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0": {
            "read_token": 0.000003,
            "write_token": 0.000015,
            "image_cost": 0.024,
        },
    }

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize a Bedrock LLM instance.

        Args:
            model_name: Name of the AWS Bedrock model to use
            **kwargs: Additional arguments passed to the base LLM class
        """
        super().__init__(model_name, **kwargs)
        # Initialize AWS Bedrock client here

    def auth(self) -> None:
        """
        Verify AWS credentials can access Bedrock.

        This method uses the AWS SDK to create a Bedrock runtime client,
        which automatically uses AWS credentials from the environment,
        AWS configuration files, or IAM roles.

        Raises:
            Exception: If authentication fails or Bedrock access is denied
        """
        try:
            import boto3
            from botocore.exceptions import ClientError

            # Create a Bedrock client to test access
            session = boto3.session.Session()
            client = session.client(service_name="bedrock-runtime")

            # Store the client in config for later use
            self.config["client"] = client

        except Exception as e:
            raise Exception(f"AWS Bedrock authentication failed: {str(e)}")

    def _get_image_bytes(self, image_path: str) -> bytes:
        """
        Read image file into bytes and convert to PNG format.

        Unlike other providers that use base64 encoded images or JPEG format,
        AWS Bedrock expects raw image bytes in PNG format with preserved
        transparency when available.

        Args:
            image_path: Path to the image file

        Returns:
            bytes: Raw image bytes in PNG format

        Raises:
            Exception: If reading or processing fails
        """
        try:
            # Open and convert image to PNG format
            with open(image_path, "rb") as f:
                img = Image.open(f)
                if img.mode in ("RGBA", "LA"):
                    # Keep alpha channel for PNG
                    pass
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Save as PNG in memory
                output = BytesIO()
                img.save(output, format="PNG")
                return output.getvalue()

        except FileNotFoundError as e:
            raise Exception(f"Failed to read image {image_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process image {image_path}: {str(e)}")

    def _download_image_from_url(self, image_url: str) -> bytes:
        """
        Download an image from a URL and convert to PNG format.

        This method:
        1. Fetches the image from the given URL
        2. Preserves transparency if present (unlike other providers)
        3. Saves as PNG format (not JPEG like other providers)
        4. Returns raw bytes (not base64 encoded)

        Args:
            image_url: URL of the image to download

        Returns:
            bytes: Raw image bytes in PNG format

        Raises:
            Exception: If download or processing fails
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            # Convert to PNG format
            img = Image.open(BytesIO(response.content))
            if img.mode in ("RGBA", "LA"):
                # Keep alpha channel for PNG
                pass
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Save as PNG in memory
            output = BytesIO()
            img.save(output, format="PNG")
            return output.getvalue()

        except requests.RequestException as e:
            raise Exception(f"Failed to download image from {image_url}: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process image from {image_url}: {str(e)}")

    def _format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert standard message format into Bedrock's format.

        This method transforms the unified message format used across providers into
        Bedrock's specific structure with roles (user/assistant) and content parts.
        Each message is converted to an object with:
        - role: either "user" or "assistant"
        - content: array of content objects (text, image, toolUse, toolResult, reasoningContent)

        Args:
            messages: List of messages in standard format

        Returns:
            List[Dict[str, Any]]: Messages formatted for Bedrock API

        Raises:
            Exception: If message processing fails, particularly during image processing
        """
        formatted_messages = []

        for msg in messages:
            # Map message type to Bedrock role
            # Only two roles are supported: "assistant" for AI, "user" for everything else
            role = "assistant" if msg["message_type"] == "ai" else "user"
            content = []

            # Add thinking/reasoning content if present
            if msg.get("thinking"):
                content.append(
                    {
                        "reasoningContent": {
                            "reasoningText": {
                                "text": msg["thinking"]["thinking"],
                                "signature": msg["thinking"]["thinking_signature"],
                            }
                        }
                    }
                )

            # Add text content first
            if msg.get("message"):
                content.append({"text": msg["message"]})

            # Process image files from local paths
            if msg.get("image_paths"):
                for image_path in msg["image_paths"]:
                    # Get image bytes will raise an exception if the file is invalid
                    image_bytes = self._get_image_bytes(image_path)
                    content.append(
                        {"image": {"format": "png", "source": {"bytes": image_bytes}}}
                    )

            # Process images from URLs
            if msg.get("image_urls"):
                for image_url in msg["image_urls"]:
                    # Download image will raise an exception if the URL is invalid
                    image_bytes = self._download_image_from_url(image_url)
                    content.append(
                        {"image": {"format": "png", "source": {"bytes": image_bytes}}}
                    )

            # Add tool use information (function calls)
            if msg.get("tool_use"):
                content.append(
                    {
                        "toolUse": {
                            "toolUseId": msg["tool_use"]["id"],
                            "name": msg["tool_use"]["name"],
                            "input": msg["tool_use"]["input"],
                        }
                    }
                )

            # Add tool result information (function responses)
            if msg.get("tool_result"):
                result = ""
                if msg["tool_result"].get("success"):
                    # Format successful tool result
                    result = msg["tool_result"]["result"]
                    content.append(
                        {
                            "toolResult": {
                                "toolUseId": msg["tool_result"]["tool_id"],
                                "content": [{"text": result}],
                            }
                        }
                    )
                else:
                    # Format failed tool result with error status
                    result = msg["tool_result"]["error"]
                    content.append(
                        {
                            "toolResult": {
                                "toolUseId": msg["tool_result"]["tool_id"],
                                "content": [{"text": result}],
                                "status": "error",
                            }
                        }
                    )

            # If no content was added, add a fallback message
            # This ensures we don't send an empty content array which would be invalid
            if not content:
                content.append({"text": "No content available"})

            # Add the completed message with role and content parts
            formatted_messages.append({"role": role, "content": content})

        return formatted_messages

    def _format_tools_for_model(
        self, tools: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert standard tool format into Bedrock's format.

        AWS Bedrock uses a different structure for tools than other providers.
        Each tool has a "toolSpec" format with name, description, and inputSchema.
        The tools are wrapped in a "tools" object.

        Args:
            tools: List of tools in standard format

        Returns:
            Dict[str, List[Dict[str, Any]]]: Tools formatted for Bedrock API, wrapped in a "tools" object

        Raises:
            Exception: If tool processing fails
        """
        formatted_tools = []

        for tool in tools:
            formatted_tools.append(
                {
                    "toolSpec": {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "inputSchema": {"json": tool.get("input_schema", {})},
                    }
                }
            )

        return {"tools": formatted_tools}

    def _raw_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        top_k: int = 200,
        tools: List[Dict[str, Any]] = None,
        thinking_budget: int = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response using AWS Bedrock's API.

        This method handles:
        1. Message formatting using Bedrock's content structure
        2. Setting up inference configuration
        3. Implementation of retry logic for AWS API throttling
        4. Making the API request via the AWS SDK
        5. Extracting and processing the response
        6. Calculating token usage and costs
        7. Extracting any reasoning content or tool calls

        Args:
            event_id: Unique identifier for this generation event
            system_prompt: System-level instructions for the model
            messages: List of messages in the standard format
            max_tokens: Maximum number of tokens to generate
            temp: Temperature for generation (0.0 = deterministic)
            top_k: Top K tokens to consider (only for some models)
            tools: Optional list of function/tool definitions
            thinking_budget: Optional token budget for thinking/reasoning

        Returns:
            Tuple containing:
            - Generated text response
            - Usage statistics (tokens, costs, tool use, thinking)

        Raises:
            Exception: If the API call fails or returns an error after retries
        """
        import time

        from botocore.exceptions import ClientError

        # Define retry parameters
        max_retries = 3
        base_delay = 1  # Base delay in seconds

        for attempt in range(max_retries):
            try:
                # Format messages for Bedrock's API
                formatted_messages = self._format_messages_for_model(messages)

                # Prepare inference_config parameters
                inference_config = {"temperature": temp, "maxTokens": max_tokens}

                # Set up model-specific parameters
                additional_model_fields = {}
                if "claude" in self.model_name:
                    additional_model_fields = {"top_k": top_k}

                # Configure thinking/reasoning for supported models
                if thinking_budget:
                    # Special configuration for thinking models
                    inference_config["temperature"] = 1
                    inference_config["maxTokens"] += thinking_budget
                    additional_model_fields.pop("top_k")
                    additional_model_fields["reasoning_config"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    }

                # Make Converse API Call with tools if provided
                if tools:
                    formatted_tools = self._format_tools_for_model(tools)
                    response = self.config["client"].converse(
                        modelId=self.model_name,
                        messages=formatted_messages,
                        system=[{"text": system_prompt}],
                        inferenceConfig=inference_config,
                        additionalModelRequestFields=additional_model_fields,
                        toolConfig=formatted_tools,
                    )
                else:
                    # Make standard call without tools
                    response = self.config["client"].converse(
                        modelId=self.model_name,
                        messages=formatted_messages,
                        system=[{"text": system_prompt}],
                        inferenceConfig=inference_config,
                        additionalModelRequestFields=additional_model_fields,
                    )

                # Initialize response text and usage statistics
                response_text = ""

                # Extract usage information and calculate costs
                usage_stats = {
                    "read_tokens": response.get("usage", {}).get("inputTokens", 0),
                    "write_tokens": response.get("usage", {}).get("outputTokens", 0),
                    "total_tokens": response.get("usage", {}).get("totalTokens", 0),
                    "images": len(
                        [
                            msg
                            for msg in messages
                            if msg.get("image_paths") or msg.get("image_urls")
                        ]
                    ),
                }

                # Calculate costs based on token usage and model rates
                usage_stats["read_cost"] = (
                    usage_stats["read_tokens"]
                    * self.COST_PER_MODEL[self.model_name]["read_token"]
                )
                usage_stats["write_cost"] = (
                    usage_stats["write_tokens"]
                    * self.COST_PER_MODEL[self.model_name]["write_token"]
                )
                usage_stats["image_cost"] = (
                    usage_stats["images"]
                    * self.COST_PER_MODEL[self.model_name]["image_cost"]
                    if usage_stats["images"] > 0
                    else 0
                )
                usage_stats["total_cost"] = (
                    usage_stats["read_cost"]
                    + usage_stats["write_cost"]
                    + usage_stats["image_cost"]
                )

                # Extract message from response
                output = response.get("output", {})
                message = output.get("message", {})

                # Process content blocks from the response
                if isinstance(message.get("content"), list):
                    for content_block in message.get("content", []):
                        # Extract text content
                        message = content_block.get("text", None)
                        # Extract reasoning content
                        reasoningContent = content_block.get("reasoningContent", None)
                        # Extract tool use information
                        tool_use = content_block.get("toolUse", None)

                        if message:
                            # Add text to response
                            response_text += message
                        elif reasoningContent:
                            # Extract reasoning/thinking information
                            reasoning_text = reasoningContent.get("reasoningText", {})
                            usage_stats.update(
                                {
                                    "thinking": reasoning_text.get("text", ""),
                                    "thinking_signature": reasoning_text.get(
                                        "signature", ""
                                    ),
                                }
                            )
                        elif tool_use:
                            # Extract tool use information
                            usage_stats["tool_use"] = {
                                "id": tool_use.get("toolUseId", ""),
                                "name": tool_use.get("name", ""),
                                "input": tool_use.get("input", {}),
                            }

                return response_text, usage_stats

            except ClientError as e:
                # Handle AWS-specific errors with retry logic
                error_code = e.response["Error"]["Code"]
                # Retry for throttling and quota errors
                if error_code in [
                    "ThrottlingException",
                    "TooManyRequestsException",
                    "ServiceQuotaExceededException",
                ]:
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        delay = (2**attempt) * base_delay  # Exponential backoff
                        time.sleep(delay)
                        continue
                raise Exception(
                    f"Bedrock API error after {max_retries} retries: {str(e)}"
                )
            except Exception as e:
                raise Exception(f"Unexpected error in Bedrock API call: {str(e)}")

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

    def supports_image_input(self) -> bool:
        """
        Check if the current model supports image input.

        All Bedrock Claude models support images with PNG format.

        Returns:
            bool: True for all Bedrock Claude models
        """
        return True
