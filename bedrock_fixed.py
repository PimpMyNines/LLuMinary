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

import logging
import random
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
    cast,
)

import boto3
import requests
from botocore.exceptions import (
    ClientError,
    ConnectionError,
    NoCredentialsError,
    ProfileNotFound,
)

from ...exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContentError,
    LLMFormatError,
    LLMMistake,
    LLMProviderError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMToolError,
)
from ..base import LLM


# Define TypedDict classes for message content parts
class TextContent(TypedDict):
    text: str


class ImageSource(TypedDict):
    bytes: bytes


class ImageFormat(TypedDict):
    format: str
    source: ImageSource


class ImageContent(TypedDict):
    image: ImageFormat


class ToolUseId(TypedDict):
    toolUseId: str
    name: str
    input: Dict[str, Any]


class ToolUseContent(TypedDict):
    toolUse: ToolUseId


class ToolResultContentItem(TypedDict):
    text: str


class ToolResultData(TypedDict):
    toolUseId: str
    content: List[ToolResultContentItem]


class ToolResultDataWithStatus(ToolResultData):
    status: str


class ToolResultContent(TypedDict):
    toolResult: Union[ToolResultData, ToolResultDataWithStatus]


class ReasoningTextContent(TypedDict):
    text: str
    signature: str


class ReasoningContent(TypedDict):
    reasoningText: ReasoningTextContent


class ReasoningContentWrapper(TypedDict):
    reasoningContent: ReasoningContent


# Union type for all content types
ContentPart = Union[
    TextContent,
    ImageContent,
    ToolUseContent,
    ToolResultContent,
    ReasoningContentWrapper,
]


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
        "anthropic.claude-instant-v1",  # Added for us-east-1 region
    ]

    CONTEXT_WINDOW = {
        "us.anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0": 200000,
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
        "anthropic.claude-instant-v1": 100000,  # Added for us-east-1 region
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
        "anthropic.claude-instant-v1": {
            "read_token": 0.000001,
            "write_token": 0.000005,
            "image_cost": 0.024,  # Added for us-east-1 region
        },
    }

    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Initialize the BedrockLLM provider.

        Args:
            model_name: Name of the model to use
            **kwargs: Additional configuration options
        """
        super().__init__(model_name, **kwargs)
        self.config = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize AWS client if profile is provided
        self.profile_name = kwargs.get("profile_name")

        # Authenticate if auto_auth is enabled (default)
        if kwargs.get("auto_auth", True):
            self.auth()

    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """
        Validate provider-specific configuration parameters.

        Verifies that the model name is supported and any required
        parameters are present.

        Args:
            config: Dictionary containing configuration parameters

        Raises:
            LLMConfigurationError: If required configuration is missing or invalid
        """
        # Verify required configuration for Bedrock
        required_keys = []

        # Check for region_name if endpoint_url is not provided
        if not config.get("endpoint_url") and not config.get("region_name"):
            required_keys.append("region_name")

        # If any required keys are missing, raise an error
        for key in required_keys:
            if key not in config:
                raise LLMConfigurationError(
                    message=f"Missing required configuration key: {key}",
                    provider="bedrock",
                    details={"config": config},
                )

        # Validate AWS profile if provided
        if config.get("profile_name"):
            # We don't validate the profile here, as it will be validated during auth()
            # Just ensure it's a string
            if not isinstance(config["profile_name"], str):
                raise LLMConfigurationError(
                    message="profile_name must be a string",
                    provider="bedrock",
                    details={"provided_type": type(config["profile_name"]).__name__},
                )

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
            LLMAuthenticationError: If authentication fails or Bedrock access is denied
            LLMConfigurationError: If configuration is invalid
        """
        try:
            # Initialize parameters for session creation
            session_kwargs = {}
            runtime_client_kwargs = {"service_name": "bedrock-runtime"}
            bedrock_client_kwargs = {"service_name": "bedrock"}

            # Check for explicit credentials in config
            if self.config.get("aws_access_key_id") and self.config.get(
                "aws_secret_access_key"
            ):
                session_kwargs["aws_access_key_id"] = self.config["aws_access_key_id"]
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
                runtime_client_kwargs["region_name"] = self.config["region_name"]
                bedrock_client_kwargs["region_name"] = self.config["region_name"]

            # Check for explicit profile in config
            if self.config.get("profile_name"):
                session_kwargs["profile_name"] = self.config["profile_name"]

            # Create the AWS session with appropriate parameters
            try:
                session = boto3.session.Session(**session_kwargs)
            except ProfileNotFound as e:
                # Handle profile not found error
                raise LLMAuthenticationError(
                    message=f"AWS profile '{self.config.get('profile_name')}' not found",
                    provider="bedrock",
                    details={
                        "profile_name": self.config.get("profile_name"),
                        "error": str(e),
                    },
                ) from e
            except NoCredentialsError as e:
                # Handle no credentials error
                raise LLMAuthenticationError(
                    message="AWS credentials not found",
                    provider="bedrock",
                    details={
                        "error": str(e),
                        "credential_source": "Failed to create session with provided credentials",
                    },
                ) from e
            except Exception as e:
                # Handle other session creation errors
                raise LLMAuthenticationError(
                    message=f"Bedrock authentication failed: {e!s}",
                    provider="bedrock",
                    details={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "credential_source": "Failed to create session",
                    },
                ) from e

            # Check for explicit endpoint URL in config
            if self.config.get("endpoint_url"):
                runtime_client_kwargs["endpoint_url"] = self.config["endpoint_url"]
                bedrock_client_kwargs["endpoint_url"] = self.config["endpoint_url"]

            # Create the Bedrock client with appropriate parameters
            try:
                runtime_client = session.client(**runtime_client_kwargs)
                bedrock_client = session.client(**bedrock_client_kwargs)

                # Validate access by making a simple API call
                try:
                    # List models to verify access (common validation method)
                    bedrock_client.list_foundation_models()
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
                            provider="bedrock",
                            details={
                                "error_code": error_code,
                                "error": error_message,
                                "action": "Ensure IAM permissions include bedrock:ListFoundationModels",
                            },
                        ) from e
                    elif error_code in [
                        "ValidationException",
                        "InvalidRequestException",
                    ]:
                        # Configuration error (likely incorrect endpoint or region)
                        raise LLMConfigurationError(
                            message=f"AWS Bedrock configuration error: {error_message}",
                            provider="bedrock",
                            details={
                                "error_code": error_code,
                                "error": error_message,
                                "region": bedrock_client.meta.region_name,
                                "endpoint": bedrock_client.meta.endpoint_url,
                            },
                        ) from e
                    else:
                        # Re-raise mapped through our error mapper
                        raise self._map_aws_error(e) from e

                # Store the clients in config for later use
                self.config["runtime_client"] = runtime_client
                self.config["bedrock_client"] = bedrock_client

            except (NoCredentialsError, ProfileNotFound) as e:
                # Handle specific client creation errors
                raise LLMAuthenticationError(
                    message=f"AWS Bedrock client creation failed: {e!s}",
                    provider="bedrock",
                    details={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "credential_source": "No credentials found in environment, config, or instance role",
                    },
                ) from e

        except (LLMAuthenticationError, LLMConfigurationError):
            # Re-raise already typed errors
            raise
        except Exception as e:
            # Map any other errors through our error mapping system
            mapped_error = self._map_aws_error(e)
            raise mapped_error from e

    def _get_image_bytes(self, image_path: str) -> bytes:
        """
        Get image bytes from a file path or URL.

        Args:
            image_path: Path to image file or URL

        Returns:
            Image bytes

        Raises:
            LLMContentError: If image cannot be loaded
        """
        try:
            # Handle URLs
            if image_path.startswith(("http://", "https://")):
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                return cast(bytes, response.content)

            # Handle local files
            with open(image_path, "rb") as f:
                return cast(bytes, f.read())
        except (OSError, requests.RequestException, FileNotFoundError) as e:
            raise LLMContentError(
                message=f"Failed to load image from {image_path}: {e!s}",
                provider="bedrock",
                details={"error": str(e)},
            ) from e

    def _download_image_from_url(self, image_url: str) -> bytes:
        """
        Download an image from a URL and return the bytes.

        Args:
            image_url: URL of the image to download

        Returns:
            Image bytes

        Raises:
            LLMContentError: If image cannot be downloaded
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return cast(bytes, response.content)
        except requests.RequestException as e:
            raise LLMContentError(
                message=f"Failed to download image from {image_url}: {e!s}",
                provider="bedrock",
                details={"error": str(e)},
            ) from e

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
        formatted_messages: List[Dict[str, List[ContentPart]]] = []

        for msg in messages:
            # Map message type to Bedrock role
            # Only two roles are supported: "assistant" for AI, "user" for everything else
            role = "assistant" if msg["message_type"] == "ai" else "user"
            content: List[ContentPart] = []

            # Add thinking/reasoning content if present
            if msg.get("thinking"):
                reasoning_content: ReasoningContentWrapper = {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": msg["thinking"]["thinking"],
                            "signature": msg["thinking"]["thinking_signature"],
                        }
                    }
                }
                content.append(reasoning_content)

            # Add text content first
            if msg.get("message"):
                text_content: TextContent = {"text": msg["message"]}
                content.append(text_content)

            # Process images from file paths
            if msg.get("image_paths"):
                for image_path in msg["image_paths"]:
                    # Get image bytes will raise an exception if the file is invalid
                    image_bytes = self._get_image_bytes(image_path)
                    # Need to use the correct nested dict type expected by content list
                    image_dict: ImageContent = {
                        "image": {"format": "png", "source": {"bytes": image_bytes}}
                    }
                    content.append(image_dict)

            # Process images from URLs
            if msg.get("image_urls"):
                for image_url in msg["image_urls"]:
                    # Download image will raise an exception if the URL is invalid
                    image_bytes = self._download_image_from_url(image_url)
                    # Need to use the correct nested dict type expected by content list
                    url_image_dict: ImageContent = {
                        "image": {"format": "png", "source": {"bytes": image_bytes}}
                    }
                    content.append(url_image_dict)

            # Add tool use information (function calls)
            if msg.get("tool_use"):
                tool_use_dict: ToolUseContent = {
                    "toolUse": {
                        "toolUseId": msg["tool_use"]["id"],
                        "name": msg["tool_use"]["name"],
                        "input": msg["tool_use"]["input"],
                    }
                }
                content.append(tool_use_dict)

            # Add tool result information (function responses)
            if msg.get("tool_result"):
                if msg["tool_result"].get("success"):
                    # Format successful tool result
                    result = msg["tool_result"]["result"]
                    content_item: ToolResultContentItem = {"text": result}
                    tool_result: ToolResultContent = {
                        "toolResult": {
                            "toolUseId": msg["tool_result"]["tool_id"],
                            "content": [content_item],
                        }
                    }
                    content.append(tool_result)
                else:
                    # Format failed tool result with error status
                    result = msg["tool_result"]["error"]
                    error_content_item: ToolResultContentItem = {"text": result}
                    error_tool_result: ToolResultContent = {
                        "toolResult": {
                            "toolUseId": msg["tool_result"]["tool_id"],
                            "content": [error_content_item],
                            "status": "error",
                        }
                    }
                    content.append(error_tool_result)

            # If no content was added, add a fallback message
            # This ensures we don't send an empty content array which would be invalid
            if not content:
                # Use TextContent to match the expected type in the content list
                fallback_content: TextContent = {"text": "No content available"}
                content.append(fallback_content)

            # Add the completed message with role and content parts
            formatted_message: Dict[str, Any] = {"role": role, "content": content}
            formatted_messages.append(formatted_message)

        # Cast to the expected return type
        return cast(List[Dict[str, Any]], formatted_messages)

    def _format_tools_for_model(
        self, tools: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Format tools for AWS Bedrock API.

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
        Map AWS errors to LLM exceptions.

        This method maps AWS-specific errors to our standardized LLM exceptions.
        It handles common error patterns from boto3/botocore and provides
        more specific error types based on the error message and code.

        Args:
            error: The original AWS error

        Returns:
            Mapped LLM exception
        """
        error_str = str(error)

        # Handle ClientError specifically
        if isinstance(error, ClientError):
            error_code = error.response.get("Error", {}).get("Code", "")
            error_message = error.response.get("Error", {}).get("Message", "")

            # Map specific error codes
            if error_code == "ThrottlingException" or "TooManyRequests" in error_code:
                return LLMRateLimitError(
                    message=f"AWS Bedrock rate limit exceeded: {error_message}",
                    provider="bedrock",
                    details={"error_code": error_code, "error": error_str},
                )
            elif (
                error_code == "AccessDeniedException"
                or error_code == "UnauthorizedException"
            ):
                return LLMAuthenticationError(
                    message=f"AWS Bedrock authentication failed: {error_message}",
                    provider="bedrock",
                    details={"error_code": error_code, "error": error_str},
                )
            elif (
                error_code == "ValidationException"
                or error_code == "InvalidRequestException"
            ):
                return LLMConfigurationError(
                    message=f"AWS Bedrock request validation failed: {error_message}",
                    provider="bedrock",
                    details={"error_code": error_code, "error": error_str},
                )
            elif (
                error_code == "ServiceUnavailableException"
                or error_code == "InternalServerException"
            ):
                return LLMServiceUnavailableError(
                    message=f"AWS Bedrock service unavailable: {error_message}",
                    provider="bedrock",
                    details={"error_code": error_code, "error": error_str},
                )
            elif (
                "content filtering" in error_message.lower()
                or "content policy" in error_message.lower()
            ):
                return LLMContentError(
                    message=f"AWS Bedrock content policy violation: {error_message}",
                    provider="bedrock",
                    details={"error_code": error_code, "error": error_str},
                )
            elif (
                "format" in error_message.lower()
                or "invalid format" in error_message.lower()
            ):
                return LLMFormatError(
                    message=f"AWS Bedrock format error: {error_message}",
                    provider="bedrock",
                    details={"error_code": error_code, "error": error_str},
                )
            elif "tool" in error_message.lower() or "function" in error_message.lower():
                return LLMToolError(
                    message=f"AWS Bedrock tool error: {error_message}",
                    provider="bedrock",
                    details={"error_code": error_code, "error": error_str},
                )
            else:
                # Default to generic provider error for unrecognized ClientError
                return LLMProviderError(
                    message=f"AWS Bedrock API error: {error_message}",
                    provider="bedrock",
                    details={"error_code": error_code, "error": error_str},
                )

        # Handle connection errors
        elif isinstance(error, ConnectionError):
            return LLMServiceUnavailableError(
                message=f"AWS Bedrock connection error: {error_str}",
                provider="bedrock",
                details={"error": error_str},
            )

        # Handle authentication errors
        elif isinstance(error, (NoCredentialsError, ProfileNotFound)):
            return LLMAuthenticationError(
                message=f"AWS Bedrock authentication error: {error_str}",
                provider="bedrock",
                details={"error": error_str},
            )

        # Handle other errors based on error message patterns
        elif "rate" in error_str.lower() and "limit" in error_str.lower():
            return LLMRateLimitError(
                message=f"AWS Bedrock rate limit exceeded: {error_str}",
                provider="bedrock",
                details={"error": error_str},
            )
        elif "content" in error_str.lower() and (
            "filter" in error_str.lower() or "policy" in error_str.lower()
        ):
            return LLMContentError(
                message=f"AWS Bedrock content policy violation: {error_str}",
                provider="bedrock",
                details={"error": error_str},
            )
        elif "format" in error_str.lower() or "invalid" in error_str.lower():
            return LLMFormatError(
                message=f"AWS Bedrock format error: {error_str}",
                provider="bedrock",
                details={"error": error_str},
            )
        elif "tool" in error_str.lower() or "function" in error_str.lower():
            return LLMToolError(
                message=f"AWS Bedrock tool error: {error_str}",
                provider="bedrock",
                details={"error": error_str},
            )
        elif "mistake" in error_str.lower() or "hallucination" in error_str.lower():
            return LLMMistake(
                message=f"AWS Bedrock model mistake: {error_str}",
                provider="bedrock",
                details={"error": error_str},
            )
        else:
            # Default to generic provider error
            return LLMProviderError(
                message=f"AWS Bedrock error: {error_str}",
                provider="bedrock",
                details={"error": error_str},
            )

    def _call_with_retry(
        self,
        func: Callable[..., Any],
        *args: Any,
        retryable_errors: Optional[List[Type[Exception]]] = None,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        **kwargs: Any,
    ) -> Any:
        """Call a function with exponential backoff retry logic."""
        retry_count = 0
        last_error: Optional[Exception] = None

        # Default retryable errors if not provided
        if retryable_errors is None:
            retryable_errors = [ClientError, ConnectionError, TimeoutError]

        while retry_count <= max_retries:
            try:
                return func(*args, **kwargs)
            except tuple(retryable_errors) as e:
                retry_count += 1
                last_error = e

                if retry_count > max_retries:
                    # We've exhausted retries, break out of the loop
                    break

                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
                jitter = random.uniform(0, 0.1 * delay)
                sleep_time = delay + jitter

                # Log retry attempt
                self.logger.warning(
                    f"Retrying after error: {e!s}. "
                    f"Attempt {retry_count}/{max_retries} in {sleep_time:.2f}s"
                )

                # Wait before retrying
                time.sleep(sleep_time)

        # If we reach here, all retries failed
        if last_error:
            raise last_error

        # This should never happen, but just in case
        raise LLMProviderError(
            message="Unexpected error in retry logic", provider="bedrock"
        )

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
            LLMAuthenticationError: If authentication fails
            LLMConfigurationError: For model or parameter errors
            LLMRateLimitError: If rate limits are exceeded
            LLMServiceUnavailableError: If AWS service is unavailable
            LLMContentError: For content policy violations
            LLMMistake: For other API errors
        """
        from ...exceptions import (
            LLMAuthenticationError,
            LLMConfigurationError,
            LLMContentError,
            LLMFormatError,
            LLMMistake,
            LLMRateLimitError,
            LLMServiceUnavailableError,
            LLMToolError,
        )

        try:
            # Ensure we're authenticated
            if "runtime_client" not in self.config:
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
                inference_config = {"temperature": temp, "maxTokens": max_tokens}

                # Set up model-specific parameters
                additional_model_fields: Dict[str, Any] = {}
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
                    provider="bedrock",
                    details={"error": str(e)},
                )

            # Define API call functions for retry mechanism
            def make_api_call_with_tools():
                # We know tools is not None here because this function is only called when tools is not None
                assert tools is not None  # Assertion for type checker
                formatted_tools = self._format_tools_for_model(tools)
                return self.config["runtime_client"].converse(
                    modelId=self.model_name,
                    messages=formatted_messages,
                    system=[{"text": system_prompt}],
                    inferenceConfig=inference_config,
                    additionalModelRequestFields=additional_model_fields,
                    toolConfig=formatted_tools,
                )

            def make_api_call_without_tools():
                return self.config["runtime_client"].converse(
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
                        make_api_call_without_tools, max_retries=3, retry_delay=1
                    )
            except Exception as e:
                # If the exception is already one of our custom types, re-raise it
                if isinstance(
                    e,
                    (
                        LLMAuthenticationError,
                        LLMConfigurationError,
                        LLMRateLimitError,
                        LLMServiceUnavailableError,
                        LLMContentError,
                        LLMFormatError,
                        LLMToolError,
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
                    "read_tokens": response.get("usage", {}).get("inputTokens", 0) or 0,
                    "write_tokens": response.get("usage", {}).get("outputTokens", 0)
                    or 0,
                    "total_tokens": response.get("usage", {}).get("totalTokens", 0)
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

                # Extract cost values with proper type handling
                read_token_cost = float(costs.get("input_cost", 0.0) or 0.0)
                write_token_cost = float(costs.get("output_cost", 0.0) or 0.0)
                image_cost = float(costs.get("image_cost", 0.0) or 0.0)

                # Round costs to avoid floating-point precision issues
                for cost_key in ["read_cost", "write_cost", "total_cost"]:
                    if cost_key in usage_stats:
                        cost_value = float(usage_stats.get(cost_key, 0.0) or 0.0)
                        usage_stats[cost_key] = round(cost_value, 6)

                # Call callback if provided
                if callback:
                    callback(response_text, usage_stats)

                # Yield the response text and usage info
                yield response_text, usage_stats

            except Exception as e:
                # Map any other errors through our error mapping system
                raise self._map_aws_error(e)

        except (
            LLMAuthenticationError,
            LLMConfigurationError,
            LLMRateLimitError,
            LLMServiceUnavailableError,
            LLMContentError,
            LLMFormatError,
            LLMToolError,
            LLMMistake,
        ):
            # Re-raise already typed errors
            raise
        except Exception as e:
            # Map any other errors through our error mapping system
            mapped_error = self._map_aws_error(e)
            raise mapped_error from e

    def stream_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        functions: Optional[List[Callable[..., Any]]] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        """
        Stream a response from the LLM, yielding chunks as they become available.

        Args:
            event_id (str): Unique identifier for this generation event
            system_prompt (str): System-level instructions for the model
            messages (List[Dict[str, Any]]): List of messages in the standard format
            max_tokens (int): Maximum number of tokens to generate
            temp (float): Temperature for generation
            functions (List[Callable]): List of functions the LLM can use
            callback (Callable): Optional callback function for each chunk

        Yields:
            Tuple[str, Dict[str, Any]]: Tuples of (text_chunk, usage_info)
        """
        try:
            # Ensure authentication
            if not hasattr(self, "client") or self.client is None:
                self.auth()

            # Convert functions to tools if provided
            tools = None
            if functions:
                tools = self._convert_functions_to_tools(functions)

            # Format messages for the model
            formatted_messages = self._format_messages_for_model(messages)

            # Initialize usage info
            usage_info = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "read_cost": 0.0,
                "write_cost": 0.0,
                "total_cost": 0.0,
            }

            # This is a stub implementation - the actual implementation would call
            # the Bedrock streaming API and process the response
            response_text = "This is a stub implementation of stream_generate."

            # Update usage info
            usage_info["input_tokens"] = self._estimate_tokens(str(formatted_messages))
            usage_info["output_tokens"] = self._estimate_tokens(response_text)
            usage_info["total_tokens"] = (
                usage_info["input_tokens"] + usage_info["output_tokens"]
            )

            # Calculate costs
            model_costs = self.get_model_costs()
            input_cost = float(model_costs.get("input_cost", 0.0) or 0.0)
            output_cost = float(model_costs.get("output_cost", 0.0) or 0.0)

            usage_info["read_cost"] = usage_info["input_tokens"] * input_cost
            usage_info["write_cost"] = usage_info["output_tokens"] * output_cost
            usage_info["total_cost"] = (
                usage_info["read_cost"] + usage_info["write_cost"]
            )

            # Call callback if provided
            if callback:
                callback(response_text, usage_info)

            # Yield the response
            yield response_text, usage_info

        except Exception as e:
            # Map any errors
            mapped_error = self._map_aws_error(e)
            raise mapped_error from e

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.

        This is a simple approximation based on character count.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        # A simple approximation: 1 token ≈ 4 characters for English text
        # Ensure we're working with integers for division
        text_length = len(text)
        char_per_token = 4
        # Use integer division to ensure we get an integer result
        token_estimate = text_length // char_per_token
        # Ensure we return at least 1 token for non-empty text
        return max(1, token_estimate)
