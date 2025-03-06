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
import base64
import logging
import pathlib
import warnings
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from google import genai
from google.genai import types
from PIL import Image

from ..base import LLM
from ...utils import get_secret

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
        SUPPORTED_MODELS: List of Gemini models supported by this implementation
        CONTEXT_WINDOW: Maximum token limits for each model
        SUPPORTS_IMAGES: Whether this provider supports image inputs
        COST_PER_MODEL: Cost information per token for each model
    """

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

    def auth(self) -> None:
        """
        Authenticate with Google using API key from AWS Secrets Manager.

        This method retrieves the Google API key from the environment or AWS Secrets Manager,
        then initializes the Google Gemini client with the appropriate API version.

        Raises:
            Exception: If authentication fails
        """
        try:
            # Get the API key from Secrets Manager
            secret = get_secret("google_api_key", required_keys=["api_key"])
            api_key = secret["api_key"]

            # Initialize the Google client with appropriate API version
            # Note: Experimental models may require different API versions
            if self.model_name == "gemini-2.0-flash-thinking-exp-01-21":
                self.client = genai.Client(
                    api_key=api_key, http_options={"api_version": "v1alpha"}
                )
            else:
                self.client = genai.Client(api_key=api_key)

        except Exception as e:
            raise Exception(f"Google authentication failed: {str(e)}")

    def _process_image(self, image_source: str, is_url: bool = False) -> Any:
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
            Exception: If image processing fails
        """
        try:
            if is_url:
                # For URLs, fetch the image content first
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                # Convert response content to PIL Image
                image_data = BytesIO(response.content)
                return Image.open(image_data)
            else:
                # For local paths, load directly using PIL
                return Image.open(image_source)
        except Exception as e:
            raise Exception(
                f"Failed to process image {'URL' if is_url else 'file'} {image_source}: {str(e)}"
            )

    def _format_messages_for_model(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """
        Format messages for Google's API.

        Converts the standardized message format used across all providers
        to Google's specific content/part structure. Each message is converted
        to a Content object with appropriate role and parts (text, images, function calls).

        Args:
            messages: List of messages in the standard format

        Returns:
            List[Content]: Messages formatted for Google's API

        Raises:
            Exception: If formatting fails, particularly during image processing
        """
        formatted_contents = []

        for message in messages:
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
            for image_path in message.get("image_paths", []):
                try:
                    image_part = self._process_image(image_path)
                    parts.append(image_part)
                except Exception as e:
                    raise Exception(
                        f"Failed to process image file {image_path}: {str(e)}"
                    )

            # Process image URLs
            for image_url in message.get("image_urls", []):
                try:
                    image_part = self._process_image(image_url, is_url=True)
                    parts.append(image_part)
                except Exception as e:
                    raise Exception(
                        f"Failed to process image URL {image_url}: {str(e)}"
                    )

            # Add function call information (outgoing tool use)
            if message.get("tool_use"):
                # Convert tool_use to Google's function_call format
                parts.append(
                    types.Part.from_function_call(
                        name=message["tool_use"]["name"],
                        args=message["tool_use"]["input"],
                    )
                )

            # Add function response information (incoming tool results)
            if message.get("tool_result"):
                name = message["tool_result"]["tool_id"]
                # Format response based on success or failure
                if message["tool_result"].get("success"):
                    function_response = {"result": message["tool_result"]["result"]}
                else:
                    function_response = {"error": message["tool_result"]["error"]}

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

    def _raw_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        tools: List[Dict[str, Any]] = None,
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
            tools: Optional list of function/tool definitions

        Returns:
            Tuple containing:
            - Generated text response
            - Usage statistics (tokens, costs, tool use)

        Raises:
            Exception: If the API call fails
        """
        try:
            # Ensure we're authenticated
            if self.client is None:
                self.auth()

            # Format messages for Google's API
            formatted_contents = self._format_messages_for_model(messages)

            # Create generation config with appropriate parameters
            if tools:
                # If tools are provided, configure them in the request
                # Setting automatic_function_calling to disable prevents the model
                # from automatically choosing a function without explicit instructions
                config = types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temp,
                    tools=tools,
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

            # Make the API request to generate content
            response = self.client.models.generate_content(
                model=self.model_name, contents=formatted_contents, config=config
            )

            # Extract usage information from response metadata
            # Note: For some API versions, these fields may not be available
            usage_metadata = getattr(response, "usage_metadata", None) or {}
            input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage_metadata, "total_token_count", None) or (
                input_tokens + output_tokens
            )

            # Extract the text response, handling potential errors gracefully
            try:
                response_text = getattr(response, "text", None) or ""
            except:
                response_text = ""

            # Count how many images were included in the request
            num_images = sum(
                len(msg.get("image_paths", [])) + len(msg.get("image_urls", []))
                for msg in messages
            )

            # Calculate costs based on token usage and our cost model
            costs = self.get_model_costs()
            read_cost = float(input_tokens) * float(costs["read_token"])
            write_cost = float(output_tokens) * float(costs["write_token"])
            image_cost = float(num_images) * float(costs["image_cost"])

            # Prepare complete usage statistics
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

            # Extract any function/tool calls from the response
            try:
                tool_use = getattr(response, "function_calls", None) or []
                if tool_use:
                    # Format tool use information in a standardized way
                    usage_stats["tool_use"] = {
                        "id": tool_use[0].id,
                        "name": tool_use[0].name,
                        "input": tool_use[0].args,
                    }
            except:
                # If extraction fails, continue without tool use info
                pass

            return response_text, usage_stats

        except Exception as e:
            raise Exception(f"Google API generation failed: {str(e)}")

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
            Exception: If streaming fails
        """
        # Import Google Generative AI here to avoid import errors if not installed
        try:
            import google.generativeai as genai
            from google.generativeai.types import (content_types,
                                                   generation_types)
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. Install with 'pip install google-generativeai'"
            )

        # Initialize client if needed
        if not hasattr(self, "genai"):
            self.auth()

        # Convert messages to Google format
        try:
            formatted_messages = self._format_messages_for_model(messages)
        except Exception as e:
            raise Exception(f"Error formatting messages for Google: {str(e)}")

        # Prepare function declarations if functions are provided
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

        # Count images for cost calculation
        image_count = 0
        for message in messages:
            if message.get("message_type") == "human":
                image_count += len(message.get("image_paths", []))
                image_count += len(message.get("image_urls", []))

        # Prepare system instruction if provided
        if system_prompt:
            # For Google, we add the system prompt as a user message at the beginning
            system_content = content_types.Content(
                role="user", parts=[content_types.Part.from_text(system_prompt)]
            )
            formatted_messages.insert(0, system_content)

        try:
            # Create the model
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temp,
                },
                tools=function_declarations,
                safety_settings=self.safety_settings
                if hasattr(self, "safety_settings")
                else None,
            )

            # Create a streaming request
            response = model.generate_content(
                formatted_messages, stream=True  # Enable streaming
            )

            # Initialize variables to accumulate data
            accumulated_text = ""
            accumulated_tokens = 0
            tool_call_data = {}

            # Rough estimation of input tokens
            input_tokens = 0
            for message in messages:
                # Estimate based on message content
                if "message" in message:
                    # Rough estimation: count words and multiply by 1.3
                    text = message.get("message", "")
                    input_tokens += int(len(text.split()) * 1.3)

                # Add token estimates for images
                input_tokens += (
                    len(message.get("image_paths", [])) * 1000
                )  # Rough estimate
                input_tokens += (
                    len(message.get("image_urls", [])) * 1000
                )  # Rough estimate

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

                # Check for function calls
                # Google API returns function calls at the end, not incrementally
                if hasattr(chunk, "candidates") and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, "content") and hasattr(
                        candidate.content, "parts"
                    ):
                        for part in candidate.content.parts:
                            if hasattr(part, "function_call"):
                                function_call = part.function_call
                                tool_id = (
                                    f"func_{len(tool_call_data)}"  # Generate an ID
                                )

                                tool_call_data[tool_id] = {
                                    "id": tool_id,
                                    "name": function_call.name,
                                    "arguments": str(
                                        function_call.args
                                    ),  # Convert args to string
                                    "type": "function",
                                }

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
            raise Exception(f"Error streaming from Google: {str(e)}")
