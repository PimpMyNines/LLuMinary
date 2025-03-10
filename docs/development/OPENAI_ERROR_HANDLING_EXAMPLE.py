"""Example implementation of standardized error handling for the OpenAI provider.

This file demonstrates the recommended approach to implementing error handling in
the OpenAI provider for the LLuMinary library. It shows comprehensive error handling
patterns including authentication, API calls with retry logic, content validation,
and image processing.

Overview:
---------
This example provides a reference implementation of standardized error handling
in the OpenAI provider, including:
- Authentication with fallbacks (environment variables, AWS Secrets)
- API calls with exponential backoff retry for rate limits
- Provider-specific error mapping
- Response validation and content extraction
- Image processing error handling

Usage:
------
Use this file as a reference when implementing error handling in
provider modules. The patterns demonstrated here can be adapted for
all LLuMinary providers.

Related Files:
-------------
- exceptions.py: Contains the exception hierarchy
- openai.py: Actual OpenAI provider implementation
- ERROR_HANDLING_IMPLEMENTATION_PLAN.md: Overall error handling plan
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from lluminary.exceptions import LLMHandlerError, LLMMistake, ProviderError


class OpenAILLM:
    """Example implementation with standardized error handling."""

    def auth(self) -> None:
        """
        Authenticate with OpenAI API with proper error handling.

        Raises:
            ProviderError: If authentication fails with detailed context
        """
        try:
            # First try environment variable
            api_key = os.environ.get("OPENAI_API_KEY")

            # Then try AWS Secrets Manager if configured
            if not api_key and self.config.get("use_aws_secrets"):
                try:
                    from lluminary.utils.aws import get_secret

                    api_key = get_secret("OPENAI_API_KEY")
                except Exception as aws_error:
                    raise ProviderError(
                        message="Failed to retrieve OpenAI API key from AWS Secrets",
                        provider="openai",
                        details={"error": str(aws_error)},
                    ) from aws_error

            # Validate API key
            if not api_key:
                raise ProviderError(
                    message="OpenAI API key not found in environment or AWS Secrets",
                    provider="openai",
                    details={
                        "config": {
                            k: "..." if k == "api_key" else v
                            for k, v in self.config.items()
                        }
                    },
                )

            # Initialize client
            import openai

            self.client = openai.OpenAI(api_key=api_key)

            # Verify authentication by making a minimal API call
            self.client.models.list(limit=1)

        except openai.AuthenticationError as e:
            raise ProviderError(
                message="OpenAI authentication failed: Invalid API key",
                provider="openai",
                details={"error": str(e)},
            ) from e
        except openai.APIConnectionError as e:
            raise ProviderError(
                message="OpenAI API connection error: Failed to connect to API",
                provider="openai",
                details={"error": str(e)},
            ) from e
        except Exception as e:
            raise ProviderError(
                message=f"OpenAI authentication failed: {e!s}",
                provider="openai",
                details={"error": str(e)},
            ) from e

    def _call_api_with_retry(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
    ) -> object:
        """
        Call OpenAI API with exponential backoff retry for rate limits.

        Args:
            messages: Formatted messages for the API
            max_tokens: Maximum tokens to generate
            temperature: Temperature setting
            tools: Optional tools/functions
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Multiplier for backoff on each retry

        Returns:
            API response object

        Raises:
            ProviderError: For non-recoverable errors with detailed context
        """
        import openai

        attempt = 0
        last_exception = None
        backoff_time = initial_backoff

        while attempt <= max_retries:
            try:
                # Prepare API call params
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }

                # Add tools if provided
                if tools:
                    params["tools"] = tools

                # Make API call
                response = self.client.chat.completions.create(**params)
                return response

            except openai.RateLimitError as e:
                attempt += 1

                # Extract retry information from headers if available
                retry_after = None
                if hasattr(e, "headers") and e.headers:
                    retry_after = e.headers.get("retry-after")

                # Use retry-after from headers or exponential backoff
                if retry_after and retry_after.isdigit():
                    backoff_time = float(retry_after)
                else:
                    backoff_time = initial_backoff * (backoff_factor**attempt)

                # If we've exceeded retries, raise with details
                if attempt > max_retries:
                    raise ProviderError(
                        message=f"OpenAI limit exceeded after {max_retries} retries",
                        provider="openai",
                        details={
                            "error": str(e),
                            "retry_attempts": attempt,
                            "retry_after": retry_after,
                        },
                    ) from e

                # Otherwise wait and retry
                time.sleep(backoff_time)
                last_exception = e

            except openai.APIError as e:
                # Map to appropriate error type and raise immediately
                raise self._map_openai_error(e) from e

            except Exception as e:
                # Unexpected error, don't retry
                raise ProviderError(
                    message=f"Unexpected error calling OpenAI API: {e!s}",
                    provider="openai",
                    details={"error": str(e)},
                ) from e

        # Should never get here, but just in case
        raise ProviderError(
            message=f"OpenAI API call failed after {max_retries} retries",
            provider="openai",
            details={"last_error": str(last_exception)},
        )

    def _map_openai_error(self, error: Exception) -> ProviderError:
        """
        Map OpenAI-specific errors to LLMHandler error types.

        Args:
            error: The original OpenAI error

        Returns:
            Mapped ProviderError with appropriate context
        """
        import openai

        error_str = str(error)

        # Authentication errors
        if isinstance(error, openai.AuthenticationError):
            return ProviderError(
                message="OpenAI authentication failed: Invalid API key",
                provider="openai",
                details={"error": error_str},
            )

        # Rate limiting errors (should be handled by retry logic)
        if isinstance(error, openai.RateLimitError):
            retry_after = None
            if hasattr(error, "headers") and error.headers:
                retry_after = error.headers.get("retry-after")

            return ProviderError(
                message="OpenAI rate limit exceeded",
                provider="openai",
                details={"error": error_str, "retry_after": retry_after},
            )

        # Validation errors
        if isinstance(error, openai.BadRequestError):
            if "context_length_exceeded" in error_str:
                return ProviderError(
                    message="OpenAI model context length exceeded",
                    provider="openai",
                    details={
                        "error": error_str,
                        "model": self.model_name,
                        "context_window": self.get_context_window(),
                    },
                )

            return ProviderError(
                message="OpenAI API validation error",
                provider="openai",
                details={"error": error_str},
            )

        # Server errors
        if isinstance(error, openai.APIError):
            return ProviderError(
                message="OpenAI API error",
                provider="openai",
                details={"error": error_str},
            )

        # Connection errors
        if isinstance(error, openai.APIConnectionError):
            return ProviderError(
                message="OpenAI API connection error",
                provider="openai",
                details={"error": error_str},
            )

        # Timeout errors
        if isinstance(error, openai.APITimeoutError):
            return ProviderError(
                message="OpenAI API timeout",
                provider="openai",
                details={"error": error_str},
            )

        # Default fallback
        return ProviderError(
            message=f"OpenAI API error: {error_str}",
            provider="openai",
            details={"error": error_str},
        )

    def _raw_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking_budget: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate a response with standardized error handling.

        Args:
            event_id: Unique identifier for this generation event
            system_prompt: System-level instructions for the model
            messages: List of messages in the standard format
            max_tokens: Maximum number of tokens to generate
            temp: Temperature for generation
            tools: List of tools to use
            thinking_budget: Number of tokens allowed for thinking

        Returns:
            Tuple containing:
                - Generated response text
                - Usage statistics
                - Updated messages list

        Raises:
            ProviderError: For provider-specific errors
            LLMMistake: For issues with the response format or content
        """
        try:
            # Format messages for the model
            formatted_messages = []

            # Add system message if provided
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})

            # Process remaining messages
            try:
                formatted_messages.extend(self._format_messages_for_model(messages))
            except Exception as e:
                raise ProviderError(
                    message=f"Failed to format messages for OpenAI: {e!s}",
                    provider="openai",
                    details={"error": str(e)},
                ) from e

            # Make API call with retry logic
            try:
                response = self._call_api_with_retry(
                    messages=formatted_messages,
                    max_tokens=max_tokens,
                    temperature=temp,
                    tools=tools,
                )
            except ProviderError:
                # Let ProviderError propagate
                raise
            except Exception as e:
                # Wrap unexpected errors
                raise ProviderError(
                    message=f"Unexpected error in OpenAI API call: {e!s}",
                    provider="openai",
                    details={"error": str(e)},
                ) from e

            # Extract content from response
            try:
                if not response.choices:
                    raise LLMMistake(
                        message="OpenAI returned empty choices array",
                        error_type="content",
                        provider="openai",
                    )

                content = response.choices[0].message.content

                # Check for empty content (when using tool calls)
                if content is None:
                    # Extract tool calls if available
                    tool_calls = response.choices[0].message.tool_calls
                    content = self._format_tool_calls(tool_calls) if tool_calls else ""

                # Validate response
                if not content.strip():
                    raise LLMMistake(
                        message="OpenAI returned empty response",
                        error_type="content",
                        provider="openai",
                    )
            except LLMMistake:
                # Let LLMMistake propagate
                raise
            except Exception as e:
                # Wrap other content extraction errors
                raise LLMMistake(
                    message=f"Failed to extract content from OpenAI response: {e!s}",
                    error_type="format",
                    provider="openai",
                    details={"error": str(e)},
                ) from e

            # Calculate usage statistics
            try:
                usage_stats = self._calculate_usage_from_response(response, messages)
            except Exception as e:
                # Non-fatal - use default stats but log error
                print(f"Warning: Failed to calculate usage statistics: {e!s}")
                usage_stats = {
                    "read_tokens": len(str(formatted_messages)) // 4,
                    "write_tokens": len(content) // 4,
                    "total_tokens": (len(str(formatted_messages)) + len(content)) // 4,
                    "read_cost": 0,
                    "write_cost": 0,
                    "total_cost": 0,
                }

            return content, usage_stats, messages

        except LLMHandlerError:
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise ProviderError(
                message=f"Unexpected error in OpenAI generation: {e!s}",
                provider="openai",
                details={"error": str(e)},
            ) from e

    def _encode_image(self, image_path: str) -> Dict[str, Any]:
        """
        Encode an image file with error handling.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with encoded image data

        Raises:
            ProviderError: If image processing fails
        """
        try:
            import base64

            if not os.path.exists(image_path):
                raise ProviderError(
                    message=f"Image file not found: {image_path}",
                    provider="openai",
                    details={"image_path": image_path},
                )

            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

            # Determine MIME type
            from PIL import Image

            image = Image.open(image_path)
            mime_type = f"image/{image.format.lower()}"

            return {
                "image": f"data:{mime_type};base64,{encoded_string}",
                "width": image.width,
                "height": image.height,
            }

        except ProviderError:
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            raise ProviderError(
                message=f"Failed to encode image {image_path}: {e!s}",
                provider="openai",
                details={"error": str(e), "image_path": image_path},
            ) from e

    def _encode_image_url(self, image_url: str) -> Dict[str, Any]:
        """
        Download and encode an image from URL with error handling.

        Args:
            image_url: URL of the image to download and encode

        Returns:
            Dictionary with encoded image data

        Raises:
            ProviderError: If image download or processing fails
        """
        try:
            import base64
            from io import BytesIO

            import requests
            from PIL import Image

            # Get image from URL
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            except requests.RequestException as e:
                raise ProviderError(
                    message=f"Failed to download image from URL {image_url}: {e!s}",
                    provider="openai",
                    details={"error": str(e), "url": image_url},
                ) from e

            # Process image
            try:
                image = Image.open(BytesIO(response.content))
                mime_type = f"image/{image.format.lower()}"
                encoded_string = base64.b64encode(response.content).decode("utf-8")

                return {
                    "image": f"data:{mime_type};base64,{encoded_string}",
                    "width": image.width,
                    "height": image.height,
                }
            except Exception as e:
                raise ProviderError(
                    message=f"Failed to process image from URL {image_url}: {e!s}",
                    provider="openai",
                    details={"error": str(e), "url": image_url},
                ) from e

        except ProviderError:
            # Let our custom exceptions propagate
            raise
        except Exception as e:
            raise ProviderError(
                message=f"Unexpected error processing image URL {image_url}: {e!s}",
                provider="openai",
                details={"error": str(e), "url": image_url},
            ) from e
