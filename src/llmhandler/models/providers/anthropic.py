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
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from PIL import Image

from ..base import LLM
from ...utils import get_secret


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

    THINKING_MODELS = ["claude-3-7-sonnet-20250219"]

    SUPPORTED_MODELS = [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
    ]

    CONTEXT_WINDOW = {
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-7-sonnet-20250219": 200000,
    }

    COST_PER_MODEL = {
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
        "claude-3-7-sonnet-20250219": {
            "read_token": 0.000003,
            "write_token": 0.000015,
            "image_cost": 0.024,
        },
    }

    def __init__(self, model_name: str, **kwargs):
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

    def auth(self) -> None:
        """
        Authenticate with Anthropic using API key from AWS Secrets Manager or direct parameter.

        This method:
        1. Uses API key from config if already provided in constructor
        2. Tries to get the API key from AWS Secrets Manager if not provided

        Raises:
            Exception: If authentication fails and no API key is available
        """
        # If API key is already set in config, we're already authenticated
        if "api_key" in self.config and self.config["api_key"]:
            return

        try:
            from ..utils import get_secret

            # Get the API key from Secrets Manager
            secret = get_secret("anthropic_api_key", required_keys=["api_key"])
            self.config["api_key"] = secret["api_key"]

        except Exception as e:
            raise Exception(f"Anthropic authentication failed: {str(e)}")

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
            Exception: If image processing fails
        """
        try:
            with open(image_path, "rb") as image_file:
                # Convert to JPEG format
                img = Image.open(image_file)

                # Handle transparency (RGBA or LA modes) by adding white background
                if img.mode in ("RGBA", "LA"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Save as JPEG in memory with 95% quality (Anthropic's recommendation)
                output = BytesIO()
                img.save(output, format="JPEG", quality=95)
                return base64.b64encode(output.getvalue()).decode("utf-8")
        except Exception as e:
            raise Exception(f"Failed to encode image {image_path}: {str(e)}")

    def _download_image_from_url(self, image_url: str) -> str:
        """
        Download an image from a URL and convert it to base64.

        This method:
        1. Downloads the image from the URL
        2. Processes it the same way as local images (RGB conversion, JPEG format)
        3. Encodes as base64

        Args:
            image_url: URL of the image to download

        Returns:
            str: Base64 encoded image string

        Raises:
            Exception: If download or encoding fails
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            # Convert to JPEG format
            img = Image.open(BytesIO(response.content))

            # Handle transparency (RGBA or LA modes) by adding white background
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Save as JPEG in memory with 95% quality
            output = BytesIO()
            img.save(output, format="JPEG", quality=95)
            return base64.b64encode(output.getvalue()).decode("utf-8")

        except requests.RequestException as e:
            raise Exception(f"Failed to download image from {image_url}: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to process image from {image_url}: {str(e)}")

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

        Raises:
            Exception: If message formatting fails, particularly during image processing
        """
        formatted_messages = []

        for msg in messages:
            # Map message type to Anthropic role:
            # - "ai" -> "assistant" (AI responses)
            # - any other value -> "user" (user inputs, tools)
            role = "assistant" if msg["message_type"] == "ai" else "user"
            content_parts = []

            # Process image files from local paths
            if msg.get("image_paths"):
                for image_path in msg["image_paths"]:
                    try:
                        # Encode image and add as an image content part
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
                    except Exception as e:
                        # Raise exception to halt processing if image fails
                        raise Exception(
                            f"Failed to process image file {image_path}: {str(e)}"
                        )

            # Process images from URLs
            if msg.get("image_urls"):
                for image_url in msg["image_urls"]:
                    try:
                        # Download, encode image and add as an image content part
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
                    except Exception as e:
                        # Raise exception to halt processing if image fails
                        raise Exception(
                            f"Failed to process image URL {image_url}: {str(e)}"
                        )

            # Add thinking information (Claude 3.7 specific)
            if msg.get("thinking"):
                content_parts.append(
                    {
                        "type": "thinking",
                        "thinking": msg["thinking"]["thinking"],
                        "signature": msg["thinking"]["thinking_signature"],
                    }
                )

            # Add text content if present
            if "message" in msg and msg["message"]:
                content_parts.append({"type": "text", "text": msg["message"]})

            # Add tool use information (function calls)
            if msg.get("tool_use"):
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
                # Format the result string based on success or failure
                result = ""
                if msg["tool_result"].get("success"):
                    result = "Tool Call Successful: " + msg["tool_result"]["result"]
                else:
                    result = "Tool Call Failed: " + msg["tool_result"]["error"]

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
                content_parts.append({"type": "text", "text": "No content available"})

            # Add the completed message with role and content parts
            formatted_messages.append({"role": role, "content": content_parts})

        return formatted_messages

    def _raw_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        tools: List[Dict[str, Any]] = None,
        thinking_budget: int = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response using Anthropic's API.

        This method handles:
        1. Message formatting using Anthropic's content structure
        2. Configuration of generation parameters
        3. Making the API request with proper headers
        4. Extracting and processing the response
        5. Calculating token usage and costs
        6. Extracting any thinking content or tool calls

        Args:
            event_id: Unique identifier for this generation event
            system_prompt: System-level instructions for the model
            messages: List of messages in the standard format
            max_tokens: Maximum number of tokens to generate
            temp: Temperature for generation (0.0 = deterministic)
            tools: Optional list of function/tool definitions
            thinking_budget: Optional token budget for thinking (Claude 3.7 only)

        Returns:
            Tuple containing:
            - Generated text response
            - Usage statistics (tokens, costs, tool use, thinking)

        Raises:
            Exception: If the API call fails
        """
        try:
            # Format messages for Anthropic's API
            formatted_messages = self._format_messages_for_model(messages)

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
                request_body["tools"] = tools

            # Configure thinking if budget is provided and model supports it
            if thinking_budget:
                # Enable thinking with specified token budget
                request_body["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                # Adjust max tokens to account for thinking budget
                request_body["max_tokens"] += thinking_budget
                # Set temperature to 1 for thinking models (recommended)
                request_body["temperature"] = 1

            # Make API call to Anthropic
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.config["api_key"],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=request_body,
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Parse response JSON
            response_data = response.json()

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

            # Extract text response and any special content
            response_text = ""
            for content in response_data["content"]:
                # Combine all text content parts
                if content["type"] == "text":
                    response_text += content.get("text", "")

                # Extract thinking content if present
                if content["type"] == "thinking":
                    usage_stats["thinking"] = content["thinking"]
                    usage_stats["thinking_signature"] = content["signature"]

                # Extract tool use information if present
                if content["type"] == "tool_use":
                    usage_stats["tool_use"] = content

            return response_text, usage_stats

        except Exception as e:
            # Propagate all exceptions
            raise Exception(str(e))

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

    def _convert_function_to_tool(self, function: Callable) -> Dict[str, Any]:
        """Convert a function to Anthropic tool format

        Args:
            function: Callable function to convert

        Returns:
            Tool definition in Anthropic format
        """
        # Get function details using the base implementation
        base_tool = super()._convert_function_to_tool(function)

        # Convert to Anthropic-specific format
        anthropic_tool = {
            "type": "function",
            "function": {
                "name": base_tool["name"],
                "description": base_tool.get("description", ""),
                "parameters": {"type": "object", "properties": {}},
            },
        }

        # Convert input schema to parameters
        if "input_schema" in base_tool and "properties" in base_tool["input_schema"]:
            anthropic_tool["function"]["parameters"]["properties"] = base_tool[
                "input_schema"
            ]["properties"]

            # Add required parameters if present
            if "required" in base_tool["input_schema"]:
                anthropic_tool["function"]["parameters"]["required"] = base_tool[
                    "input_schema"
                ]["required"]

        return anthropic_tool

    def _convert_functions_to_tools(
        self, functions: List[Callable]
    ) -> List[Dict[str, Any]]:
        """Convert functions to Anthropic tool format

        Args:
            functions: List of callable functions to convert

        Returns:
            List of tool definitions in Anthropic format
        """
        tools = []
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
        functions: List[Callable] = None,
        callback: Callable[[str, Dict[str, Any]], None] = None,
    ):
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
            Exception: If streaming fails
        """
        # Import Anthropic here to avoid import errors if not installed
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with 'pip install anthropic'"
            )

        # Initialize client if needed
        if not hasattr(self, "client"):
            self.auth()

        # Convert messages to Anthropic format
        formatted_messages = self._format_messages_for_model(messages)

        # Prepare tools if functions are provided
        tools = None
        if functions:
            tools = []
            for function in functions:
                # Parse function signature and create tool schema
                name = getattr(function, "__name__", str(function))
                docstring = getattr(function, "__doc__", "")
                schema = {
                    "name": name,
                    "description": docstring or f"Call {name} function",
                    "input_schema": {},  # Simplified schema
                }
                tools.append(schema)

        # Count images for cost calculation
        image_count = 0
        for message in messages:
            if message.get("message_type") == "human":
                image_count += len(message.get("image_paths", []))
                image_count += len(message.get("image_urls", []))

        try:
            # Create a streaming request
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temp,
                tools=tools,
                stream=True,  # Enable streaming
            )

            # Initialize variables to accumulate data
            accumulated_text = ""
            accumulated_tokens = 0
            tool_call_data = {}

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
                            callback(content, partial_usage)

                        # Yield the content chunk and partial usage data
                        yield content, {
                            "event_id": event_id,
                            "model": self.model_name,
                            "read_tokens": input_tokens,
                            "write_tokens": accumulated_tokens,
                            "images": image_count,
                            "total_tokens": input_tokens + accumulated_tokens,
                            "is_complete": False,
                            "tool_use": tool_call_data,
                        }

                # Check for tool calls in the delta
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "tool_use"):
                    tool_use = chunk.delta.tool_use
                    if tool_use:
                        tool_id = tool_use.id

                        # Initialize or update tool call data
                        if tool_id not in tool_call_data:
                            tool_call_data[tool_id] = {
                                "id": tool_id,
                                "name": tool_use.name
                                if hasattr(tool_use, "name")
                                else "",
                                "arguments": tool_use.input
                                if hasattr(tool_use, "input")
                                else "",
                                "type": "function",
                            }
                        elif hasattr(tool_use, "input") and tool_use.input:
                            # Append to existing arguments if this is a partial update
                            tool_call_data[tool_id]["arguments"] += tool_use.input

            # Calculate costs
            model_costs = self.get_model_costs()
            read_cost = input_tokens * model_costs["read_token"]
            write_cost = accumulated_tokens * model_costs["write_token"]

            # Calculate image cost if applicable
            image_cost = 0
            if image_count > 0 and "image_token" in model_costs:
                image_cost = image_count * model_costs["image_token"]

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
            raise Exception(f"Error streaming from Anthropic: {str(e)}")
