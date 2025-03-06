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

import logging
import pathlib
import warnings
from io import BytesIO
from typing import Any, Callable, Dict, List, Tuple

import requests
from PIL import Image
from google import genai
from google.genai import types

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
            AuthenticationError: If authentication fails
        """
        from ...exceptions import AuthenticationError

        try:
            # Try to get API key from environment variables first
            import os

            api_key = os.environ.get("GOOGLE_API_KEY")

            # If not in environment, get from Secrets Manager
            if not api_key:
                try:
                    secret = get_secret("google_api_key",
                                        required_keys=["api_key"])
                    api_key = secret["api_key"]
                except Exception as secret_error:
                    raise LLMAuthenticationError(
                        message=f"Failed to retrieve Google API key: {secret_error!s}",
                        provider="GoogleLLM",
                        details={
                            "original_error": str(secret_error),
                            "message": "API key not found in environment or secrets manager",
                        },
                    )

            if not api_key:
                raise LLMAuthenticationError(
                    message="Google API key not found in environment or secrets manager",
                    provider="GoogleLLM",
                    details={
                        "source_checked": ["GOOGLE_API_KEY",
                                           "AWS Secrets Manager"]
                    },
                )

            # Initialize the Google client with appropriate API version
            # Note: Experimental models may require different API versions
            try:
                if self.model_name == "gemini-2.0-flash-thinking-exp-01-21":
                    self.client = genai.Client(
                        api_key=api_key, http_options={"api_version": "v1alpha"}
                    )
                else:
                    self.client = genai.Client(api_key=api_key)
            except Exception as client_error:
                raise LLMAuthenticationError(
                    message=f"Failed to initialize Google client: {client_error!s}",
                    provider="GoogleLLM",
                    details={
                        "original_error": str(client_error),
                        "model": self.model_name,
                    },
                )

        except AuthenticationError:
            # Re-raise authentication errors directly
            raise
        except Exception as e:
            # Map generic errors to authentication errors
            raise LLMAuthenticationError(
                message=f"Google authentication failed: {e!s}",
                provider="GoogleLLM",
                details={"original_error": str(e)},
            )

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
            ContentError: If the image content is invalid or unsupported
            LLMMistake: If image processing fails due to other reasons
        """
        from ...exceptions import ContentError, LLMMistake

        try:
            if is_url:
                # For URLs, fetch the image content first
                try:
                    response = requests.get(image_source, timeout=10)
                    response.raise_for_status()
                except requests.RequestException as req_error:
                    # Detailed handling of different HTTP errors
                    status_code = getattr(req_error.response, "status_code",
                                          None)
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

        except (LLMMistake, ContentError):
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

    def _format_messages_for_model(self, messages: List[Dict[str, Any]]) -> \
    List[Any]:
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
                except LLMMistake:
                    # Re-raise LLMMistake exceptions
                    raise
                except Exception as e:
                    # Convert to LLMMistake if not already
                    from ...exceptions import LLMMistake

                    raise LLMMistake(
                        message=f"Failed to process image file {image_path}: {e!s}",
                        error_type="image_processing_error",
                        provider="GoogleLLM",
                        details={"path": image_path, "original_error": str(e)},
                    )

            # Process image URLs
            for image_url in message.get("image_urls", []):
                try:
                    image_part = self._process_image(image_url, is_url=True)
                    parts.append(image_part)
                except LLMMistake:
                    # Re-raise LLMMistake exceptions
                    raise
                except Exception as e:
                    # Convert to LLMMistake if not already
                    from ...exceptions import LLMMistake

                    raise LLMMistake(
                        message=f"Failed to process image URL {image_url}: {e!s}",
                        error_type="image_url_error",
                        provider="GoogleLLM",
                        details={"url": image_url, "original_error": str(e)},
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
                    function_response = {
                        "result": message["tool_result"]["result"]}
                else:
                    function_response = {
                        "error": message["tool_result"]["error"]}

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
        Map a Google API exception to an appropriate LLuMinary exception.

        This method examines the error message and type to determine the most
        appropriate custom exception to raise, providing standardized error
        handling across providers.

        Args:
            error: Original Google API exception

        Returns:
            Exception: Appropriate LLuMinary exception
        """
        from ...exceptions import LLMMistake

        error_message = str(error).lower()
        error_type = type(error).__name__

        # Authentication and permission errors
        if (
                "api key" in error_message
                or "authentication" in error_message
                or "credential" in error_message
                or "permission" in error_message
                or "authorization" in error_message
                or "unauthenticated" in error_message
                or "access denied" in error_message
        ):
            return LLMAuthenticationError(
                message=f"Google authentication failed: {error!s}",
                provider="GoogleLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        # Rate limit errors
        if (
                "rate limit" in error_message
                or "quota" in error_message
                or "too many requests" in error_message
                or "request rate" in error_message
                or "resource exhausted" in error_message
                or "limit exceeded" in error_message
        ):
            # Extract retry-after value if available, or use default
            retry_after = 60  # Default retry delay in seconds
            if hasattr(error, "retry_after"):
                retry_after = error.retry_after
            elif "retry after" in error_message:
                # Try to extract retry time from message
                try:
                    # Simple extraction; might need refinement for different formats
                    import re

                    match = re.search(r"retry after (\d+)", error_message)
                    if match:
                        retry_after = int(match.group(1))
                except (ValueError, IndexError):
                    pass

            return LLMRateLimitError(
                message=f"Google API rate limit exceeded: {error!s}",
                provider="GoogleLLM",
                retry_after=retry_after,
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        # Service availability errors
        if (
                "unavailable" in error_message
                or "server error" in error_message
                or "service error" in error_message
                or "internal server error" in error_message
                or "503" in error_message
                or "502" in error_message
                or "500" in error_message
                or "backend error" in error_message
                or "deadline exceeded" in error_message
                or "timeout" in error_message
        ):
            return LLMServiceUnavailableError(
                message=f"Google API service unavailable: {error!s}",
                provider="GoogleLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        # Configuration errors
        if (
                "invalid model" in error_message
                or "model not found" in error_message
                or "configuration" in error_message
                or "invalid parameter" in error_message
                or "invalid value" in error_message
                or "incorrect request" in error_message
                or "unsupported" in error_message
        ):
            return LLMConfigurationError(
                message=f"Google API configuration error: {error!s}",
                provider="GoogleLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        # Content moderation/policy violations
        if (
                "content" in error_message
                and "policy" in error_message
                or "safety" in error_message
                or "harmful" in error_message
                or "inappropriate" in error_message
                or "blocked" in error_message
                or "violation" in error_message
                or "moderation" in error_message
        ):
            return LLMContentError(
                message=f"Google API content policy violation: {error!s}",
                provider="GoogleLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        # Format errors
        if (
                "format" in error_message
                or "invalid json" in error_message
                or "parse error" in error_message
                or "serialization" in error_message
                or "deserialization" in error_message
        ):
            return LLMFormatError(
                message=f"Google API format error: {error!s}",
                provider="GoogleLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        # Tool/function errors
        if "function" in error_message or "tool" in error_message:
            return LLMToolError(
                message=f"Google API tool error: {error!s}",
                provider="GoogleLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        # Default case: general LLMMistake
        return LLMMistake(
            message=f"Google API error: {error!s}",
            error_type="api_error",
            provider="GoogleLLM",
            details={"original_error": str(error), "error_type": error_type},
        )

    def _call_with_retry(self, func, *args, max_retries=3, retry_delay=1,
                         **kwargs):
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

        from ...exceptions import RateLimitError, ServiceUnavailableError

        attempts = 0
        last_error = None

        while attempts <= max_retries:
            try:
                return func(*args, **kwargs)
            except (RateLimitError, ServiceUnavailableError) as e:
                last_error = e
                attempts += 1

                if attempts > max_retries:
                    # Re-raise the exception after max retries
                    raise

                # Determine retry delay with exponential backoff and jitter
                # If the error provides a retry_after value, use that
                if isinstance(e, RateLimitError) and e.retry_after:
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
            LLMMistake: If the API call fails with a recoverable error
            AuthenticationError: If authentication fails
            RateLimitError: If rate limits are exceeded
            ServiceUnavailableError: If the service is unavailable
        """
        from ...exceptions import (
            AuthenticationError,
            ConfigurationError,
            LLMMistake,
            RateLimitError,
            ServiceUnavailableError,
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
                    model=self.model_name, contents=formatted_contents,
                    config=config
                )

            try:
                response = self._call_with_retry(api_call, max_retries=3,
                                                 retry_delay=1)
            except Exception as e:
                # Map API call errors through our error mapping system
                if isinstance(
                        e,
                        (
                                AuthenticationError,
                                RateLimitError,
                                ServiceUnavailableError,
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
                input_tokens = getattr(usage_metadata, "prompt_token_count",
                                       0) or 0
                output_tokens = (
                        getattr(usage_metadata, "candidates_token_count",
                                0) or 0
                )
                total_tokens = getattr(usage_metadata, "total_token_count",
                                       None) or (
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
            except Exception:
                # No need to raise an exception for tool extraction failures
                # Log the error and continue
                pass

            return response_text, usage_stats

        except (
                AuthenticationError,
                ConfigurationError,
                RateLimitError,
                ServiceUnavailableError,
                LLMMistake,
        ):
            # Re-raise already mapped exceptions
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
            LLMMistake: If the API call fails with a recoverable error
            AuthenticationError: If authentication fails
            RateLimitError: If rate limits are exceeded
            ServiceUnavailableError: If the service is unavailable
            ImportError: If the Google Generative AI package is not installed
        """
        # Import required exceptions
        from ...exceptions import (
            AuthenticationError,
            ConfigurationError,
            LLMMistake,
            RateLimitError,
            ServiceUnavailableError,
        )

        # Import Google Generative AI here to avoid import errors if not installed
        try:
            import google.generativeai as genai
            from google.generativeai.types import content_types, \
                generation_types
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
            formatted_messages = self._format_messages_for_model(messages)
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
                provider="GoogleLLM",
                details={"original_error": str(e)},
            )

        # Count images for cost calculation
        image_count = 0
        for message in messages:
            if message.get("message_type") == "human":
                image_count += len(message.get("image_paths", []))
                image_count += len(message.get("image_urls", []))

        # Prepare system instruction if provided
        try:
            if system_prompt:
                # For Google, we add the system prompt as a user message at the beginning
                system_content = content_types.Content(
                    role="user",
                    parts=[content_types.Part.from_text(system_prompt)]
                )
                formatted_messages.insert(0, system_content)
        except Exception as e:
            raise LLMConfigurationError(
                message=f"Failed to configure system prompt for Google streaming: {e!s}",
                provider="GoogleLLM",
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
                    self.safety_settings if hasattr(self,
                                                    "safety_settings") else None
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
                            content = content[len(system_prompt):].lstrip()

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
                                try:
                                    function_call = part.function_call
                                    tool_id = (
                                        f"func_{len(tool_call_data)}"
                                    # Generate an ID
                                    )

                                    tool_call_data[tool_id] = {
                                        "id": tool_id,
                                        "name": function_call.name,
                                        "arguments": str(
                                            function_call.args
                                        ),  # Convert args to string
                                        "type": "function",
                                    }
                                except Exception:
                                    # Log but don't fail the streaming process
                                    pass

            # Calculate costs
            model_costs = self.get_model_costs()
            read_cost = input_tokens * model_costs["read_token"]
            write_cost = accumulated_tokens * model_costs["write_token"]

            # Calculate image cost if applicable
            image_cost = 0
            if image_count > 0 and "image_cost" in model_costs:
                image_cost = image_count * model_costs["image_cost"]

            total_cost = read_cost + write_cost + image_cost

            # Create final usage data
            final_usage = {
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
                AuthenticationError,
                ConfigurationError,
                RateLimitError,
                ServiceUnavailableError,
                LLMMistake,
        ):
            # Re-raise already mapped exceptions
            raise
        except Exception as e:
            # Map to appropriate exception type via the error mapper
            mapped_error = self._map_google_error(e)
            raise mapped_error
