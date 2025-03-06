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

from io import BytesIO
from typing import Any, Dict, List, Tuple

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

        Authentication methods are tried in this order:
        1. Explicitly provided AWS credentials in config
        2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
        3. AWS configuration files (~/.aws/credentials)
        4. IAM role attached to the instance (EC2, Lambda, etc.)

        Raises:
            AuthenticationError: If authentication fails or Bedrock access is denied
        """
        from ...exceptions import AuthenticationError, ConfigurationError

        try:
            import boto3
            from botocore.exceptions import (
                ClientError,
                NoCredentialsError,
                ProfileNotFound,
            )

            # Initialize parameters for session creation
            session_kwargs = {}
            client_kwargs = {"service_name": "bedrock-runtime"}

            # Check for explicit credentials in config
            if self.config.get("aws_access_key_id") and self.config.get(
                    "aws_secret_access_key"
            ):
                session_kwargs["aws_access_key_id"] = self.config[
                    "aws_access_key_id"]
                session_kwargs["aws_secret_access_key"] = self.config[
                    "aws_secret_access_key"
                ]

                # Include session token if provided (for temporary credentials)
                if self.config.get("aws_session_token"):
                    session_kwargs["aws_session_token"] = self.config[
                        "aws_session_token"
                    ]

            # Check for explicit region in config
            if self.config.get("region_name"):
                session_kwargs["region_name"] = self.config["region_name"]
                client_kwargs["region_name"] = self.config["region_name"]

            # Check for explicit profile in config
            if self.config.get("profile_name"):
                session_kwargs["profile_name"] = self.config["profile_name"]

            # Create the AWS session with appropriate parameters
            try:
                session = boto3.session.Session(**session_kwargs)
            except (NoCredentialsError, ProfileNotFound) as e:
                # Handle specific session creation errors
                raise LLMAuthenticationError(
                    message=f"AWS Bedrock authentication failed: {e!s}",
                    provider="BedrockLLM",
                    details={
                        "original_error": str(e),
                        "error_type": type(e).__name__,
                        "credential_source": "Failed to create session with provided credentials",
                    },
                )

            # Check for explicit endpoint URL in config
            if self.config.get("endpoint_url"):
                client_kwargs["endpoint_url"] = self.config["endpoint_url"]

            # Create the Bedrock client with appropriate parameters
            try:
                client = session.client(**client_kwargs)

                # Validate access by making a simple API call
                try:
                    # List models to verify access (common validation method)
                    client.list_foundation_models(maxResults=1)
                except ClientError as e:
                    # Handle specific Bedrock access errors
                    error_code = e.response["Error"]["Code"]
                    error_message = e.response["Error"]["Message"]

                    # Map to appropriate error type
                    if error_code in [
                        "AccessDeniedException",
                        "UnauthorizedException",
                        "InvalidSignatureException",
                    ]:
                        raise LLMAuthenticationError(
                            message=f"AWS Bedrock access denied: {error_message}",
                            provider="BedrockLLM",
                            details={
                                "error_code": error_code,
                                "original_error": error_message,
                                "action": "Ensure IAM permissions include bedrock:ListFoundationModels",
                            },
                        )
                    elif error_code in [
                        "ValidationException",
                        "InvalidRequestException",
                    ]:
                        # Configuration error (likely incorrect endpoint or region)
                        raise LLMConfigurationError(
                            message=f"AWS Bedrock configuration error: {error_message}",
                            provider="BedrockLLM",
                            details={
                                "error_code": error_code,
                                "original_error": error_message,
                                "region": client.meta.region_name,
                                "endpoint": client.meta.endpoint_url,
                            },
                        )
                    else:
                        # Re-raise mapped through our error mapper
                        raise self._map_aws_error(e)

                # Store the client in config for later use
                self.config["client"] = client

            except (NoCredentialsError, ProfileNotFound) as e:
                # Handle specific client creation errors
                raise LLMAuthenticationError(
                    message=f"AWS Bedrock client creation failed: {e!s}",
                    provider="BedrockLLM",
                    details={
                        "original_error": str(e),
                        "error_type": type(e).__name__,
                        "credential_source": "No credentials found in environment, config, or instance role",
                    },
                )

        except (AuthenticationError, ConfigurationError):
            # Re-raise already typed errors
            raise
        except Exception as e:
            # Map any other errors through our error mapping system
            mapped_error = self._map_aws_error(e)
            raise mapped_error

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
            ContentError: If the image content is invalid or unsupported
            LLMMistake: If reading or processing fails for other reasons
        """
        import os

        from ...exceptions import ContentError, LLMMistake

        try:
            # First validate the path exists
            if not os.path.exists(image_path):
                raise LLMMistake(
                    message=f"Image file not found: {image_path}",
                    error_type="image_processing_error",
                    provider="BedrockLLM",
                    details={"path": image_path},
                )

            # Check file access permissions
            if not os.access(image_path, os.R_OK):
                raise LLMMistake(
                    message=f"Permission denied reading image file: {image_path}",
                    error_type="image_processing_error",
                    provider="BedrockLLM",
                    details={"path": image_path, "error": "Permission denied"},
                )

            # Check if it's actually a file (not a directory)
            if not os.path.isfile(image_path):
                raise LLMMistake(
                    message=f"Not a valid image file: {image_path}",
                    error_type="image_processing_error",
                    provider="BedrockLLM",
                    details={"path": image_path, "error": "Not a file"},
                )

            # Check file size (AWS Bedrock has a 5MB limit)
            file_size = os.path.getsize(image_path)
            if file_size > 5 * 1024 * 1024:  # 5MB in bytes
                raise LLMMistake(
                    message=f"Image file too large: {image_path} ({file_size} bytes). Maximum size is 5MB.",
                    error_type="image_processing_error",
                    provider="BedrockLLM",
                    details={
                        "path": image_path,
                        "size": file_size,
                        "max_size": 5 * 1024 * 1024,
                    },
                )

            # Open and convert image to PNG format
            try:
                with open(image_path, "rb") as f:
                    img = Image.open(f)

                    # Verify image format is valid
                    try:
                        img.verify()  # Verify that it's a valid image
                        # Reopen the image since verify() closes it
                        with open(image_path, "rb") as f:
                            img = Image.open(f)
                    except Exception as verify_error:
                        raise LLMContentError(
                            message=f"Invalid or corrupted image file: {image_path}",
                            provider="BedrockLLM",
                            details={
                                "path": image_path,
                                "original_error": str(verify_error),
                            },
                        )

                    # Handle different image modes
                    if img.mode in ("RGBA", "LA"):
                        # Keep alpha channel for PNG
                        pass
                    elif img.mode != "RGB":
                        img = img.convert("RGB")

                    # Check dimensions (AWS Bedrock has limits)
                    width, height = img.size
                    if width > 4096 or height > 4096:
                        # Resize large images (AWS Bedrock max is typically 4096x4096)
                        aspect_ratio = width / height
                        if width > height:
                            new_width = min(width, 4096)
                            new_height = int(new_width / aspect_ratio)
                        else:
                            new_height = min(height, 4096)
                            new_width = int(new_height * aspect_ratio)

                        img = img.resize((new_width, new_height), Image.LANCZOS)

                    # Save as PNG in memory
                    output = BytesIO()
                    img.save(output, format="PNG")
                    return output.getvalue()
            except ContentError:
                # Re-raise ContentError directly
                raise
            except OSError as io_error:
                # Handle file I/O errors
                raise LLMMistake(
                    message=f"Error reading image file {image_path}: {io_error!s}",
                    error_type="image_processing_error",
                    provider="BedrockLLM",
                    details={"path": image_path,
                             "original_error": str(io_error)},
                )
            except Exception as processing_error:
                # Handle image processing errors
                raise LLMContentError(
                    message=f"Failed to process image {image_path}: {processing_error!s}",
                    provider="BedrockLLM",
                    details={
                        "path": image_path,
                        "original_error": str(processing_error),
                    },
                )

        except (ContentError, LLMMistake):
            # Re-raise specialized errors
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise LLMMistake(
                message=f"Unexpected error processing image {image_path}: {e!s}",
                error_type="image_processing_error",
                provider="BedrockLLM",
                details={"path": image_path, "original_error": str(e)},
            )

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
            ContentError: If the image content is invalid or unsupported
            LLMMistake: If download or processing fails for other reasons
        """
        from ...exceptions import ContentError, LLMMistake

        try:
            # Fetch the image from URL with proper error handling
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
            except requests.HTTPError as http_error:
                # Handle different HTTP error codes
                status_code = (
                    http_error.response.status_code
                    if hasattr(http_error, "response")
                    else None
                )

                if status_code == 404:
                    raise LLMMistake(
                        message=f"Image URL not found (404): {image_url}",
                        error_type="image_url_error",
                        provider="BedrockLLM",
                        details={
                            "url": image_url,
                            "status_code": 404,
                            "original_error": str(http_error),
                        },
                    )
                elif status_code in (401, 403):
                    raise LLMMistake(
                        message=f"Access denied to image URL: {image_url}",
                        error_type="image_url_error",
                        provider="BedrockLLM",
                        details={
                            "url": image_url,
                            "status_code": status_code,
                            "original_error": str(http_error),
                        },
                    )
                elif status_code and status_code >= 500:
                    raise LLMMistake(
                        message=f"Server error when fetching image URL: {image_url}",
                        error_type="image_url_error",
                        provider="BedrockLLM",
                        details={
                            "url": image_url,
                            "status_code": status_code,
                            "original_error": str(http_error),
                        },
                    )
                else:
                    raise LLMMistake(
                        message=f"HTTP error when fetching image URL {image_url}: {http_error!s}",
                        error_type="image_url_error",
                        provider="BedrockLLM",
                        details={
                            "url": image_url,
                            "status_code": status_code,
                            "original_error": str(http_error),
                        },
                    )
            except requests.RequestException as request_error:
                raise LLMMistake(
                    message=f"Failed to fetch image URL {image_url}: {request_error!s}",
                    error_type="image_url_error",
                    provider="BedrockLLM",
                    details={"url": image_url,
                             "original_error": str(request_error)},
                )

            # Check content size (AWS Bedrock has a 5MB limit)
            content_size = len(response.content)
            if content_size > 5 * 1024 * 1024:  # 5MB in bytes
                raise LLMMistake(
                    message=f"Image from URL too large: {image_url} ({content_size} bytes). Maximum size is 5MB.",
                    error_type="image_url_error",
                    provider="BedrockLLM",
                    details={
                        "url": image_url,
                        "size": content_size,
                        "max_size": 5 * 1024 * 1024,
                    },
                )

            # Process the image data
            try:
                img = Image.open(BytesIO(response.content))

                # Verify image format is valid
                try:
                    img.verify()
                    # Reopen since verify() closes the image
                    img = Image.open(BytesIO(response.content))
                except Exception as verify_error:
                    raise LLMContentError(
                        message=f"Invalid or unsupported image format from URL: {image_url}",
                        provider="BedrockLLM",
                        details={"url": image_url,
                                 "original_error": str(verify_error)},
                    )

                # Handle different image modes
                if img.mode in ("RGBA", "LA"):
                    # Keep alpha channel for PNG
                    pass
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Check dimensions (AWS Bedrock has limits)
                width, height = img.size
                if width > 4096 or height > 4096:
                    # Resize large images (AWS Bedrock max is typically 4096x4096)
                    aspect_ratio = width / height
                    if width > height:
                        new_width = min(width, 4096)
                        new_height = int(new_width / aspect_ratio)
                    else:
                        new_height = min(height, 4096)
                        new_width = int(new_height * aspect_ratio)

                    img = img.resize((new_width, new_height), Image.LANCZOS)

                # Save as PNG in memory
                output = BytesIO()
                img.save(output, format="PNG")
                return output.getvalue()

            except ContentError:
                # Re-raise ContentError directly
                raise
            except Exception as processing_error:
                # Handle image processing errors
                raise LLMContentError(
                    message=f"Failed to process image from URL {image_url}: {processing_error!s}",
                    provider="BedrockLLM",
                    details={"url": image_url,
                             "original_error": str(processing_error)},
                )

        except (ContentError, LLMMistake):
            # Re-raise specialized errors
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            raise LLMMistake(
                message=f"Unexpected error processing image URL {image_url}: {e!s}",
                error_type="image_url_error",
                provider="BedrockLLM",
                details={"url": image_url, "original_error": str(e)},
            )

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
                                "signature": msg["thinking"][
                                    "thinking_signature"],
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
                        {"image": {"format": "png",
                                   "source": {"bytes": image_bytes}}}
                    )

            # Process images from URLs
            if msg.get("image_urls"):
                for image_url in msg["image_urls"]:
                    # Download image will raise an exception if the URL is invalid
                    image_bytes = self._download_image_from_url(image_url)
                    content.append(
                        {"image": {"format": "png",
                                   "source": {"bytes": image_bytes}}}
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

    def _map_aws_error(self, error: Exception) -> Exception:
        """
        Map AWS specific errors to LLuMinary custom exceptions.

        This method examines AWS ClientError and other exceptions to determine
        the most appropriate custom exception to raise, providing standardized
        error handling across all providers.

        Args:
            error: Original AWS exception (usually ClientError)

        Returns:
            Exception: Appropriate LLuMinary exception
        """
        from botocore.exceptions import ClientError

        from ...exceptions import LLMMistake

        # Handle AWS ClientError with structured error information
        if isinstance(error, ClientError):
            # Extract error code and message from ClientError
            error_code = error.response["Error"]["Code"]
            error_message = str(error)

            # Authentication and authorization errors
            if error_code in [
                "AccessDeniedException",
                "UnauthorizedException",
                "InvalidSignatureException",
                "ExpiredTokenException",
                "TokenRefreshRequired",
                "InvalidAccessKeyId",
                "SignatureDoesNotMatch",
                "MissingAuthenticationToken",
            ]:
                return LLMAuthenticationError(
                    message=f"Bedrock API authentication failed: {error_message}",
                    provider="BedrockLLM",
                    details={"error_code": error_code,
                             "original_error": error_message},
                )

            # Rate limit and throttling errors
            elif error_code in [
                "ThrottlingException",
                "TooManyRequestsException",
                "RequestLimitExceeded",
                "ServiceQuotaExceededException",
                "LimitExceededException",
                "RateLimitExceededException",
                "ProvisionedThroughputExceededException",
            ]:
                # Extract retry-after if available
                retry_after = 30  # Default retry delay for AWS
                headers = error.response.get("ResponseMetadata", {}).get(
                    "HTTPHeaders", {}
                )
                if headers and "retry-after" in headers:
                    try:
                        retry_after = int(headers["retry-after"])
                    except (ValueError, TypeError):
                        pass

                return LLMRateLimitError(
                    message=f"Bedrock API rate limit exceeded: {error_message}",
                    provider="BedrockLLM",
                    retry_after=retry_after,
                    details={"error_code": error_code,
                             "original_error": error_message},
                )

            # Service availability errors
            elif error_code in [
                "ServiceUnavailableException",
                "InternalServerException",
                "ServiceFailure",
                "InternalFailure",
                "InternalServiceError",
                "ServiceException",
                "ServerException",
                "503",
            ]:
                return LLMServiceUnavailableError(
                    message=f"Bedrock API service unavailable: {error_message}",
                    provider="BedrockLLM",
                    details={"error_code": error_code,
                             "original_error": error_message},
                )

            # Configuration errors
            elif error_code in [
                "ValidationException",
                "InvalidParameterException",
                "InvalidRequestException",
                "MalformedPolicyDocumentException",
                "InvalidInputException",
                "ModelNotReadyException",
                "BadRequestException",
                "400",
            ]:
                return LLMConfigurationError(
                    message=f"Bedrock API configuration error: {error_message}",
                    provider="BedrockLLM",
                    details={"error_code": error_code,
                             "original_error": error_message},
                )

            # Content moderation and policy issues
            elif error_code in [
                "ContentException",
                "ContentModerationException",
                "AbuseDetected",
                "InvalidContentTypeException",
                "ContentLengthExceededException",
            ]:
                return LLMContentError(
                    message=f"Bedrock API content policy violation: {error_message}",
                    provider="BedrockLLM",
                    details={"error_code": error_code,
                             "original_error": error_message},
                )

            # Format errors
            elif error_code in [
                "SerializationException",
                "MalformedQueryString",
                "ParserError",
                "MalformedInputException",
            ]:
                return LLMFormatError(
                    message=f"Bedrock API format error: {error_message}",
                    provider="BedrockLLM",
                    details={"error_code": error_code,
                             "original_error": error_message},
                )

            # Tool-related errors
            elif error_code in [
                "ToolException",
                "InvalidToolException",
                "ToolExecutionFailure",
            ]:
                return LLMToolError(
                    message=f"Bedrock API tool error: {error_message}",
                    provider="BedrockLLM",
                    details={"error_code": error_code,
                             "original_error": error_message},
                )

            # Generic fallback for unknown AWS errors
            else:
                return LLMMistake(
                    message=f"Bedrock API error: {error_message}",
                    error_type="api_error",
                    provider="BedrockLLM",
                    details={"error_code": error_code,
                             "original_error": error_message},
                )

        # Handle non-ClientError exceptions
        error_message = str(error).lower()
        error_type = type(error).__name__

        # Check for patterns in other exceptions
        if any(
                term in error_message
                for term in
                ["credential", "access", "permission", "auth", "token", "key"]
        ):
            return LLMAuthenticationError(
                message=f"Bedrock API authentication failed: {error!s}",
                provider="BedrockLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        elif any(
                term in error_message
                for term in ["throttl", "rate", "limit", "quota", "exceeded"]
        ):
            retry_after = 60  # Default retry delay
            return LLMRateLimitError(
                message=f"Bedrock API rate limit exceeded: {error!s}",
                provider="BedrockLLM",
                retry_after=retry_after,
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        elif any(
                term in error_message
                for term in ["unavailable", "down", "timeout", "outage", "5xx"]
        ):
            return LLMServiceUnavailableError(
                message=f"Bedrock API service unavailable: {error!s}",
                provider="BedrockLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        elif any(
                term in error_message for term in
                ["config", "param", "invalid", "model"]
        ):
            return LLMConfigurationError(
                message=f"Bedrock API configuration error: {error!s}",
                provider="BedrockLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        elif any(
                term in error_message
                for term in ["content", "moderation", "policy", "abuse"]
        ):
            return LLMContentError(
                message=f"Bedrock API content policy violation: {error!s}",
                provider="BedrockLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        elif any(term in error_message for term in
                 ["format", "json", "xml", "parse"]):
            return LLMFormatError(
                message=f"Bedrock API format error: {error!s}",
                provider="BedrockLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        elif any(term in error_message for term in ["tool", "function"]):
            return LLMToolError(
                message=f"Bedrock API tool error: {error!s}",
                provider="BedrockLLM",
                details={"original_error": str(error),
                         "error_type": error_type},
            )

        # Default fallback for other errors
        return LLMMistake(
            message=f"Bedrock API error: {error!s}",
            error_type="api_error",
            provider="BedrockLLM",
            details={"original_error": str(error), "error_type": error_type},
        )

    def _call_with_retry(self, func, *args, max_retries=3, retry_delay=1,
                         **kwargs):
        """
        Execute an AWS API call with automatic retry for transient errors.

        This method implements an exponential backoff retry mechanism for handling
        transient errors from AWS, such as throttling or temporary service
        unavailability.

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

        from botocore.exceptions import ClientError

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
                    # Otherwise use exponential backoff with jitter
                    delay = min(
                        retry_delay * (2 ** (attempts - 1)), 60
                    )  # Cap at 60 seconds
                    # Add jitter to avoid thundering herd
                    delay = delay * (0.5 + random.random())

                # Wait before retrying
                time.sleep(delay)
            except ClientError as e:
                # For ClientError, check if it's a retryable error
                error_code = e.response["Error"]["Code"]

                if error_code in [
                    "ThrottlingException",
                    "TooManyRequestsException",
                    "ServiceQuotaExceededException",
                    "LimitExceededException",
                    "ProvisionedThroughputExceededException",
                    "ServiceUnavailableException",
                    "InternalServerException",
                    "ServiceFailure",
                ]:
                    # Map to our custom exception
                    mapped_error = self._map_aws_error(e)
                    last_error = mapped_error
                    attempts += 1

                    if attempts > max_retries:
                        # Re-raise the mapped exception after max retries
                        raise mapped_error

                    # Get retry-after if available from headers
                    headers = e.response.get("ResponseMetadata", {}).get(
                        "HTTPHeaders", {}
                    )
                    if headers and "retry-after" in headers:
                        try:
                            delay = int(headers["retry-after"])
                        except (ValueError, TypeError):
                            # Fall back to exponential backoff with jitter
                            delay = min(retry_delay * (2 ** (attempts - 1)), 60)
                            delay = delay * (0.5 + random.random())
                    else:
                        # Use exponential backoff with jitter
                        delay = min(retry_delay * (2 ** (attempts - 1)), 60)
                        delay = delay * (0.5 + random.random())

                    # Wait before retrying
                    time.sleep(delay)
                else:
                    # For non-retryable ClientError, map and raise immediately
                    mapped_error = self._map_aws_error(e)
                    raise mapped_error
            except Exception as e:
                # For all other exceptions, map and raise immediately
                mapped_error = self._map_aws_error(e)
                raise mapped_error

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
            AuthenticationError: If authentication fails
            ConfigurationError: For model or parameter errors
            RateLimitError: If rate limits are exceeded
            ServiceUnavailableError: If AWS service is unavailable
            ContentError: For content policy violations
            LLMMistake: For other API errors
        """
        from ...exceptions import (
            AuthenticationError,
            ConfigurationError,
            ContentError,
            FormatError,
            LLMMistake,
            RateLimitError,
            ServiceUnavailableError,
            ToolError,
        )

        try:
            # Ensure we're authenticated
            if "client" not in self.config:
                try:
                    self.auth()
                except Exception as e:
                    # Map any authentication errors through our error mapping system
                    raise self._map_aws_error(e)

            # Format messages for AWS Bedrock
            try:
                formatted_messages = self._format_messages_for_model(messages)
            except LLMMistake:
                # Re-raise LLMMistake exceptions
                raise
            except Exception as e:
                # Map message formatting errors
                raise self._map_aws_error(e)

            # Prepare API request configuration
            try:
                # Prepare inference_config parameters
                inference_config = {"temperature": temp,
                                    "maxTokens": max_tokens}

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
            except Exception as e:
                # Handle configuration errors
                raise LLMConfigurationError(
                    message=f"Failed to configure Bedrock API request: {e!s}",
                    provider="BedrockLLM",
                    details={"original_error": str(e)},
                )

            # Define API call functions for retry mechanism
            def make_api_call_with_tools():
                formatted_tools = self._format_tools_for_model(tools)
                return self.config["client"].converse(
                    modelId=self.model_name,
                    messages=formatted_messages,
                    system=[{"text": system_prompt}],
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields,
                    toolConfig=formatted_tools,
                )

            def make_api_call_without_tools():
                return self.config["client"].converse(
                    modelId=self.model_name,
                    messages=formatted_messages,
                    system=[{"text": system_prompt}],
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields,
                )

            # Make the API call with retry logic
            try:
                if tools:
                    response = self._call_with_retry(
                        make_api_call_with_tools, max_retries=3, retry_delay=1
                    )
                else:
                    response = self._call_with_retry(
                        make_api_call_without_tools, max_retries=3,
                        retry_delay=1
                    )
            except Exception as e:
                # If the exception is already one of our custom types, re-raise it
                if isinstance(
                        e,
                        (
                                AuthenticationError,
                                ConfigurationError,
                                RateLimitError,
                                ServiceUnavailableError,
                                ContentError,
                                FormatError,
                                ToolError,
                                LLMMistake,
                        ),
                ):
                    raise
                # Otherwise, map it using our error mapper
                raise self._map_aws_error(e)

            # Process the response and extract usage information
            try:
                # Initialize response text and usage statistics
                response_text = ""

                # Extract usage information, with fallbacks for missing data
                usage_stats = {
                    "read_tokens": response.get("usage", {}).get("inputTokens",
                                                                 0) or 0,
                    "write_tokens": response.get("usage", {}).get(
                        "outputTokens", 0)
                                    or 0,
                    "total_tokens": response.get("usage", {}).get("totalTokens",
                                                                  0)
                                    or 0,
                    "images": len(
                        [
                            msg
                            for msg in messages
                            if msg.get("image_paths") or msg.get("image_urls")
                        ]
                    ),
                }

                # Calculate costs based on token usage and model rates
                costs = self.get_model_costs()
                usage_stats["read_cost"] = float(
                    usage_stats["read_tokens"]) * float(
                    costs["read_token"]
                )
                usage_stats["write_cost"] = float(
                    usage_stats["write_tokens"]) * float(
                    costs["write_token"]
                )
                usage_stats["image_cost"] = (
                    float(usage_stats["images"]) * float(costs["image_cost"])
                    if usage_stats["images"] > 0
                    else 0
                )
                usage_stats["total_cost"] = (
                        usage_stats["read_cost"]
                        + usage_stats["write_cost"]
                        + usage_stats["image_cost"]
                )

                # Round costs to avoid floating-point precision issues
                for cost_key in ["read_cost", "write_cost", "image_cost",
                                 "total_cost"]:
                    if cost_key in usage_stats:
                        usage_stats[cost_key] = round(usage_stats[cost_key], 6)
            except Exception as e:
                # Handle errors in usage data extraction with reasonable defaults
                usage_stats = {
                    "read_tokens": 0,
                    "write_tokens": 0,
                    "total_tokens": 0,
                    "images": 0,
                    "read_cost": 0,
                    "write_cost": 0,
                    "image_cost": 0,
                    "total_cost": 0,
                }
                # Log this error but continue - usage stats are secondary to response
                print(f"Warning: Error extracting usage stats: {e!s}")

            # Extract message content from response
            try:
                # Extract message from response
                output = response.get("output", {})
                message = output.get("message", {})

                # Process content blocks from the response
                if isinstance(message.get("content"), list):
                    for content_block in message.get("content", []):
                        # Extract text content
                        text_content = content_block.get("text", None)
                        # Extract reasoning content
                        reasoningContent = content_block.get("reasoningContent",
                                                             None)
                        # Extract tool use information
                        tool_use = content_block.get("toolUse", None)

                        if text_content is not None:
                            # Add text to response
                            response_text += text_content
                        elif reasoningContent:
                            # Extract reasoning/thinking information
                            reasoning_text = reasoningContent.get(
                                "reasoningText", {})
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

                # If no text response was found, this is an error
                if (
                        not response_text
                        and not usage_stats.get("thinking")
                        and not usage_stats.get("tool_use")
                ):
                    raise LLMContentError(
                        message="Bedrock API returned empty response with no text, thinking, or tool use",
                        provider="BedrockLLM",
                        details={"response_structure": str(response)},
                    )
            except ContentError:
                # Re-raise content errors
                raise
            except Exception as e:
                # Map errors in response processing
                raise LLMContentError(
                    message=f"Failed to extract content from Bedrock API response: {e!s}",
                    provider="BedrockLLM",
                    details={
                        "original_error": str(e),
                        "response_structure": str(response),
                    },
                )

            return response_text, usage_stats

        except (
                AuthenticationError,
                ConfigurationError,
                RateLimitError,
                ServiceUnavailableError,
                ContentError,
                FormatError,
                ToolError,
                LLMMistake,
        ):
            # Re-raise already mapped exceptions
            raise
        except Exception as e:
            # Last resort: catch any remaining unmapped exceptions
            mapped_error = self._map_aws_error(e)
            raise mapped_error

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
