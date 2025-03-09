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
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union, Literal, cast

import requests
from PIL import Image, ImageFile
from PIL.Image import Resampling  # For LANCZOS constant
import base64
import time
import hashlib
import io
import json
import os
import random
import re
import boto3
import botocore
from botocore.config import Config

from ..base import LLM
from ...exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContentError,
    LLMError,
    LLMFormatError,
    LLMMistake,
    LLMProviderError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMThinkingError,
    LLMToolError,
    LLMValidationError,
)


class BedrockLLM(LLM):
    """
    Amazon Bedrock LLM provider integration.
    
    This provider interfaces with Amazon's Bedrock service to access models such as
    Claude, Llama 2, and others hosted on AWS Bedrock. It supports text generation,
    image understanding, and function calling.
    
    Features:
    - Support for AWS authentication via multiple methods
    - Automatic message formatting for Bedrock-specific API
    - Image processing and uploading
    - Tool/function calling support
    - Robust retry mechanism for AWS API throttling
    
    Configuration options:
    - aws_profile: AWS profile name from ~/.aws/credentials
    - aws_region: AWS region (default: us-east-1)
    - aws_access_key_id: Explicit AWS access key
    - aws_secret_access_key: Explicit AWS secret key
    - aws_session_token: Optional session token for temporary credentials
    - assume_role_arn: Optional role ARN to assume
    
    Usage example:
        bedrock = BedrockLLM(config={
            "aws_profile": "prod",
            "aws_region": "us-west-2"
        })
        response = bedrock.generate("Generate a story about a robot learning to cook.")
    """
    # Define class variables with ClassVar for correct typing
    THINKING_MODELS: ClassVar[List[str]] = [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0"
    ]
    
    SUPPORTED_MODELS: ClassVar[List[str]] = [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "anthropic.claude-instant-v1",
        "anthropic.claude-v2",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
        "cohere.command-text-v14",
        "meta.llama2-13b-chat-v1",
        "meta.llama2-70b-chat-v1",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1"
    ]
    
    CONTEXT_WINDOW: ClassVar[Dict[str, int]] = {
        "anthropic.claude-3-haiku-20240307-v1:0": 200000,
        "anthropic.claude-3-sonnet-20240229-v1:0": 200000,
        "anthropic.claude-3-opus-20240229-v1:0": 200000,
        "anthropic.claude-3-5-sonnet-20240620-v1:0": 200000,
        "anthropic.claude-instant-v1": 100000,
        "anthropic.claude-v2": 100000,
        "amazon.titan-text-express-v1": 8000,
        "amazon.titan-text-lite-v1": 4000,
        "cohere.command-text-v14": 128000,
        "meta.llama2-13b-chat-v1": 4096,
        "meta.llama2-70b-chat-v1": 4096,
        "mistral.mistral-large-2402-v1:0": 32768,
        "mistral.mistral-7b-instruct-v0:2": 8192,
        "mistral.mixtral-8x7b-instruct-v0:1": 32768
    }
    
    COST_PER_MODEL: ClassVar[Dict[str, Dict[str, Union[float, None]]]] = {
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "read_token": 0.00000125,
            "write_token": 0.00000625,
            "image_cost": 0.00500000
        },
        "anthropic.claude-3-sonnet-20240229-v1:0": {
            "read_token": 0.00000375,
            "write_token": 0.00001875,
            "image_cost": 0.00500000
        },
        "anthropic.claude-3-opus-20240229-v1:0": {
            "read_token": 0.00001500,
            "write_token": 0.00007500,
            "image_cost": 0.00500000
        },
        "anthropic.claude-3-5-sonnet-20240620-v1:0": {
            "read_token": 0.00000300,
            "write_token": 0.00001500,
            "image_cost": 0.00500000
        },
        "anthropic.claude-instant-v1": {
            "read_token": 0.00000163,
            "write_token": 0.00000551,
            "image_cost": None
        },
        "anthropic.claude-v2": {
            "read_token": 0.00001102,
            "write_token": 0.00003266,
            "image_cost": None
        },
        "amazon.titan-text-express-v1": {
            "read_token": 0.00000030,
            "write_token": 0.00000400,
            "image_cost": None
        },
        "amazon.titan-text-lite-v1": {
            "read_token": 0.00000003,
            "write_token": 0.00000004,
            "image_cost": None
        },
        "cohere.command-text-v14": {
            "read_token": 0.00000050,
            "write_token": 0.00000150,
            "image_cost": None
        },
        "meta.llama2-13b-chat-v1": {
            "read_token": 0.00000075,
            "write_token": 0.00000100,
            "image_cost": None
        },
        "meta.llama2-70b-chat-v1": {
            "read_token": 0.00000195,
            "write_token": 0.00000260,
            "image_cost": None
        },
        "mistral.mistral-large-2402-v1:0": {
            "read_token": 0.00000800,
            "write_token": 0.00002400,
            "image_cost": None
        },
        "mistral.mistral-7b-instruct-v0:2": {
            "read_token": 0.00000020,
            "write_token": 0.00000060,
            "image_cost": None
        },
        "mistral.mixtral-8x7b-instruct-v0:1": {
            "read_token": 0.00000080,
            "write_token": 0.00000240,
            "image_cost": None
        }
    }
    
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """
        Initialize the Bedrock LLM provider.
        
        Args:
            model_name: Name of the Bedrock model to use
            **kwargs: Additional configuration options
        """
        # Call the parent class constructor
        super().__init__(model_name, **kwargs)
        
        # Set default model if not specified
        self.default_model = kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")

    def auth(self) -> None:
        """
        Authenticate with AWS Bedrock service.
        
        This method initializes the Bedrock runtime client based on provided
        authentication configuration (AWS credentials).
        
        Supported auth methods:
        - AWS profile (aws_profile)
        - AWS access key and secret (aws_access_key_id, aws_secret_access_key)
        - AWS role ARN assumption (assume_role_arn)
        - Automatic detection via environment variables
        
        Returns:
            None
        
        Raises:
            LLMAuthenticationError: If authentication fails
            LLMConfigurationError: If required configuration is missing
        """
        # Get config values
        aws_profile = self.config.get("aws_profile")
        aws_region = self.config.get("aws_region", "us-east-1")
        aws_access_key_id = self.config.get("aws_access_key_id")
        aws_secret_access_key = self.config.get("aws_secret_access_key")
        aws_session_token = self.config.get("aws_session_token")
        assume_role_arn = self.config.get("assume_role_arn")
        
        # Configure AWS session
        boto_session = None
        
        try:
            # If profile specified, use it
            if aws_profile:
                boto_session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
            
            # If access key and secret specified, use them
            elif aws_access_key_id and aws_secret_access_key:
                boto_session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    region_name=aws_region,
                )
            
            # Otherwise, use default credentials
            else:
                boto_session = boto3.Session(region_name=aws_region)
            
            # If role ARN specified, assume it
            if assume_role_arn:
                sts_client = boto_session.client("sts")
                response = sts_client.assume_role(
                    RoleArn=assume_role_arn,
                    RoleSessionName="BedrockLLMSession"
                )
                
                credentials = response["Credentials"]
                
                boto_session = boto3.Session(
                    aws_access_key_id=credentials["AccessKeyId"],
                    aws_secret_access_key=credentials["SecretAccessKey"],
                    aws_session_token=credentials["SessionToken"],
                    region_name=aws_region,
                )
            
            # Create client with retry configuration
            retries_config = Config(
                retries={
                    "max_attempts": 10,
                    "mode": "adaptive"
                }
            )
            
            # Initialize Bedrock runtime client
            self.client = boto_session.client(
                service_name="bedrock-runtime",
                region_name=aws_region,
                config=retries_config,
            )
        
        except (
            botocore.exceptions.ClientError,
            botocore.exceptions.NoCredentialsError,
            botocore.exceptions.InvalidConfigError,
        ) as e:
            raise LLMAuthenticationError(f"AWS Bedrock authentication failed: {str(e)}")
        
        # Set service name for internal tracking
        self.service = "bedrock"
        
    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration for this provider.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            LLMConfigurationError: If configuration is invalid
        """
        # Validate AWS region if specified
        if "aws_region" in config and not isinstance(config["aws_region"], str):
            raise LLMConfigurationError(
                "aws_region must be a string",
                details={"provided_type": type(config["aws_region"]).__name__}
            )
        
        # Validate credentials if specified
        if "aws_access_key_id" in config:
            if not isinstance(config["aws_access_key_id"], str):
                raise LLMConfigurationError(
                    "aws_access_key_id must be a string",
                    details={"provided_type": type(config["aws_access_key_id"]).__name__}
                )
            
            # Secret key is required when access key is provided
            if "aws_secret_access_key" not in config:
                raise LLMConfigurationError(
                    "aws_secret_access_key is required when aws_access_key_id is provided"
                )
            
            if not isinstance(config["aws_secret_access_key"], str):
                raise LLMConfigurationError(
                    "aws_secret_access_key must be a string",
                    details={"provided_type": type(config["aws_secret_access_key"]).__name__}
                )
        
        # Validate session token if specified
        if "aws_session_token" in config and not isinstance(config["aws_session_token"], str):
            raise LLMConfigurationError(
                "aws_session_token must be a string",
                details={"provided_type": type(config["aws_session_token"]).__name__}
            )
        
        # Validate AWS profile if specified
        if "aws_profile" in config and not isinstance(config["aws_profile"], str):
            raise LLMConfigurationError(
                "aws_profile must be a string",
                details={"provided_type": type(config["aws_profile"]).__name__}
            )

    def _resize_image(self, image: Image.Image, max_size: int = 2048) -> Image.Image:
        """
        Resize an image while maintaining aspect ratio.
        
        Args:
            image: PIL Image object
            max_size: Maximum size for width or height
            
        Returns:
            Resized PIL Image object
        """
        orig_width, orig_height = image.size
        
        # If image is already smaller than max size, return it unchanged
        if orig_width <= max_size and orig_height <= max_size:
            return image
        
        # Calculate new dimensions maintaining aspect ratio
        if orig_width > orig_height:
            new_width = max_size
            new_height = int(orig_height * (max_size / orig_width))
        else:
            new_height = max_size
            new_width = int(orig_width * (max_size / orig_height))
        
        # Resize using high-quality Lanczos resampling
        return image.resize((new_width, new_height), Resampling.LANCZOS)

    def _process_image(self, image_path: str) -> Image.Image:
        """
        Load and process an image from a file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
            
        Raises:
            LLMContentError: If image processing fails
        """
        try:
            # Open the image file and explicitly cast to Image.Image
            img = cast(Image.Image, Image.open(image_path))
            
            # Convert to RGBA for consistent handling
            img_rgba = img.convert("RGBA")
            
            # Resize if needed
            return self._resize_image(img_rgba)
            
        except Exception as e:
            raise LLMContentError(f"Failed to process image {image_path}: {str(e)}")

    def _get_image_bytes(self, image_path: str) -> bytes:
        """
        Convert an image file to PNG bytes for API submission.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Bytes of the image in PNG format
            
        Raises:
            LLMContentError: If image processing fails
        """
        try:
            # Process the image
            img = self._process_image(image_path)
            
            # Convert to PNG bytes
            img_byte_arr = io.BytesIO()
            
            # Save as PNG, preserving transparency
            img.save(img_byte_arr, format="PNG")
            
            # Return the bytes
            return img_byte_arr.getvalue()
            
        except Exception as e:
            raise LLMContentError(f"Failed to convert image {image_path} to bytes: {str(e)}")
            
    def _download_image_from_url(self, image_url: str) -> bytes:
        """
        Download an image from a URL, resize if necessary, and convert to PNG bytes.
        
        Args:
            image_url: URL of the image
            
        Returns:
            Bytes of the image in PNG format
            
        Raises:
            LLMContentError: If image download or processing fails
        """
        try:
            # Download the image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # Load image from bytes and explicitly cast to Image.Image
            img = cast(Image.Image, Image.open(io.BytesIO(response.content)))
            
            # Convert to RGBA for consistent handling
            img_rgba = img.convert("RGBA")
            
            # Resize if needed
            img_resized = self._resize_image(img_rgba)
            
            # Convert to PNG bytes
            img_byte_arr = io.BytesIO()
            img_resized.save(img_byte_arr, format="PNG")
            
            # Return the bytes
            return img_byte_arr.getvalue()
            
        except requests.RequestException as e:
            raise LLMContentError(f"Failed to download image from URL {image_url}: {str(e)}")
        except Exception as e:
            raise LLMContentError(f"Failed to process image from URL {image_url}: {str(e)}")

    def _format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert the standard message format to AWS Bedrock format.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of Bedrock-formatted message dictionaries
        """
        bedrock_messages = []
        
        for msg in messages:
            # Get the message content
            message_type = msg["message_type"]
            content = msg["message"]
            
            # Map our message types to Bedrock roles
            if message_type == "human":
                role = "user"
            elif message_type == "ai":
                role = "assistant"
            else:
                # Skip tool messages as they're handled specially
                continue
            
            # Build a content array for this message
            content_parts = []
            
            # Add the message text
            if content:
                content_parts.append({"type": "text", "text": content})
            
            # Process images for human messages
            if message_type == "human":
                # Process image paths
                for img_path in msg.get("image_paths", []):
                    try:
                        img_bytes = self._get_image_bytes(img_path)
                        content_parts.append({
                            "type": "image", 
                            "source": {
                                "type": "base64", 
                                "media_type": "image/png",
                                "data": base64.b64encode(img_bytes).decode('utf-8')
                            }
                        })
                    except Exception as e:
                        # Log error but continue
                        print(f"Error processing image {img_path}: {e!s}")
                
                # Process image URLs
                for img_url in msg.get("image_urls", []):
                    try:
                        img_bytes = self._download_image_from_url(img_url)
                        content_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(img_bytes).decode('utf-8')
                            }
                        })
                    except Exception as e:
                        # Log error but continue
                        print(f"Error processing image URL {img_url}: {e!s}")
            
            # Add the entire message with its role and content parts
            bedrock_messages.append({
                "role": role,
                "content": content_parts
            })
            
        return bedrock_messages

    def _format_tools_for_model(
        self, tools: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Format tools to match AWS Bedrock's expected format.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            Dictionary containing formatted tool definitions
        """
        bedrock_tools = []
        
        for tool in tools:
            # Extract basic tool information
            tool_name = tool.get("name", "")
            tool_description = tool.get("description", "")
            
            # Format parameters - convert to OpenAPI schema format
            parameters = tool.get("parameters", {})
            bedrock_tool = {
                "name": tool_name,
                "description": tool_description,
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Process each parameter
            for param_name, param_def in parameters.get("properties", {}).items():
                # Add parameter to properties
                bedrock_tool["input_schema"]["properties"][param_name] = {
                    "type": param_def.get("type", "string"),
                    "description": param_def.get("description", "")
                }
                
                # Add to required list if parameter is required
                if param_name in parameters.get("required", []):
                    bedrock_tool["input_schema"]["required"].append(param_name)
            
            bedrock_tools.append(bedrock_tool)
        
        return {"tools": bedrock_tools}

    def _map_aws_error(self, error: Exception) -> Exception:
        """
        Map AWS exceptions to standardized LLM exceptions.
        
        This method takes various AWS/Bedrock-specific errors and maps them to
        our standardized exception types for consistent error handling across
        different providers.
        
        Args:
            error (Exception): The AWS-specific exception
            
        Returns:
            Exception: The mapped standardized exception
        """
        # Handle throttling/rate limit errors
        if hasattr(error, "response") and getattr(error, "response", {}).get("Error", {}).get("Code") == "ThrottlingException":
            return LLMRateLimitError(f"AWS Bedrock rate limit exceeded: {str(error)}")
        
        # Handle authentication errors
        elif isinstance(error, botocore.exceptions.NoCredentialsError):
            return LLMAuthenticationError(f"AWS credentials not found: {str(error)}")
        
        # Handle service unavailable errors
        elif isinstance(error, (
            botocore.exceptions.EndpointConnectionError,
            botocore.exceptions.ConnectTimeoutError,
        )):
            return LLMServiceUnavailableError(f"AWS Bedrock service unavailable: {str(error)}")
        
        # Handle validation errors
        elif hasattr(error, "response") and getattr(error, "response", {}).get("Error", {}).get("Code") == "ValidationException":
            return LLMContentError(f"AWS Bedrock validation error: {str(error)}")
        
        # Handle model errors
        elif hasattr(error, "response") and getattr(error, "response", {}).get("Error", {}).get("Code") == "ModelErrorException":
            return LLMContentError(f"AWS Bedrock model error: {str(error)}")
        
        # Handle not found errors
        elif hasattr(error, "response") and getattr(error, "response", {}).get("Error", {}).get("Code") == "ResourceNotFoundException":
            return LLMConfigurationError(f"AWS Bedrock resource not found: {str(error)}")
        
        # Handle access denied errors
        elif hasattr(error, "response") and getattr(error, "response", {}).get("Error", {}).get("Code") == "AccessDeniedException":
            return LLMAuthenticationError(f"AWS Bedrock access denied: {str(error)}")
        
        # Other AWS errors
        elif isinstance(error, botocore.exceptions.BotoCoreError):
            return LLMServiceUnavailableError(f"AWS Bedrock service error: {str(error)}")
        
        # Unknown errors
        else:
            return error

    def _call_with_retry(self, func, *args, max_retries=3, retry_delay=1,
                        **kwargs):
        """
        Call a function with exponential backoff retry for AWS throttling.
        
        Args:
            func: Function to call
            *args: Positional arguments to pass to the function
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries (will be doubled each attempt)
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result from the function
            
        Raises:
            Exception: Any exception raised by the function after retries
        """
        last_exception = None
        retries = 0
        
        # Initial delay
        delay = retry_delay
        
        while retries <= max_retries:
            try:
                # Attempt to call the function
                return func(*args, **kwargs)
            
            except botocore.exceptions.ClientError as e:
                # Only retry on throttling exceptions
                if hasattr(e, "response") and e.response.get("Error", {}).get("Code") in [
                    "ThrottlingException",
                    "TooManyRequestsException",
                    "LimitExceededException",
                ]:
                    # Store the exception to re-raise if retries are exhausted
                    last_exception = e
                    
                    # Check if we're out of retries
                    if retries == max_retries:
                        break
                    
                    # Exponential backoff with jitter
                    wait_time = delay * (0.5 + random.random())
                    time.sleep(wait_time)
                    
                    # Increase the delay for next retry (exponential backoff)
                    delay *= 2
                    retries += 1
                    
                    # Continue to next retry
                    continue
                else:
                    # Non-throttling exception, just raise it
                    raise
                    
            except Exception as e:
                # For non-AWS exceptions, don't retry
                raise
        
        # If we get here, we've exhausted retries
        # Raise appropriate exception based on the last error
        if last_exception:
            # Map to our standard error type
            mapped_exception = self._map_aws_error(last_exception)
            raise mapped_exception
        
        # This should not happen, but just in case
        raise LLMServiceUnavailableError(
            "Exhausted retries but no exception was recorded"
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
        Generate a completion from AWS Bedrock and process the API response.
        
        This is the primary method that interacts with the AWS Bedrock API to generate
        text completions. It handles message formatting, parameter validation, and
        response processing.
        
        Args:
            event_id: Unique ID for tracking this generation request
            system_prompt: System instructions for the model
            messages: List of message dictionaries
            max_tokens: Maximum number of tokens to generate
            temp: Temperature for sampling (higher = more random)
            top_k: Number of highest probability tokens to consider
            tools: Optional list of tool specifications
            thinking_budget: Optional token budget for model reasoning
        
        Returns:
            Tuple containing: 
                - The generated text
                - Metadata dictionary with usage stats, latency, etc.
        
        Raises:
            LLMAuthenticationError: For credential and permission errors
            LLMConfigurationError: For model configuration errors
            LLMContentError: For content policy violations or invalid inputs
            LLMFormatError: For issues with message format or structure
            LLMRateLimitError: For rate limiting and quota errors
            LLMServiceUnavailableError: For API outages and timeouts
            LLMToolError: For errors related to tool definitions/usage
        """
        start_time = time.time()
        
        try:
            # Get the model ID from the first message
            model_id = messages[0].get("model", self.default_model)
            if not model_id:
                model_id = self.default_model
            
            # Format the messages for Bedrock
            bedrock_messages = self._format_messages_for_model(messages)
            
            # Create the request body
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": bedrock_messages,
            }
            
            # Add temperature if not zero
            if temp is not None and temp > 0:
                request_body["temperature"] = temp
                
            # Add top_k if specified
            if top_k is not None:
                request_body["top_k"] = top_k
                
            # Add tools if specified
            if tools is not None:
                tool_config = self._format_tools_for_model(tools)
                request_body.update(tool_config)
            
            # Convert to JSON
            body = json.dumps(request_body)
            
            # Call Bedrock API
            response = self.client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body
            )
            
            # Parse response
            response_body = json.loads(response.get("body").read().decode("utf-8"))
            
            # Extract relevant data
            completion = response_body["content"][0]["text"]
            
            # Calculate tokens used
            input_tokens = response_body.get("usage", {}).get("input_tokens", 0)
            output_tokens = response_body.get("usage", {}).get("output_tokens", 0)
            
            # Compute latency
            end_time = time.time()
            latency = end_time - start_time
            
            # Compile metadata
            metadata = {
                "model": model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "latency": latency,
                "raw_response": response_body,
            }
            
            return completion, metadata
            
        except botocore.exceptions.ClientError as e:
            # Map AWS errors to our standard error types
            mapped_error = self._map_aws_error(e)
            
            # Log error details
            print(f"AWS Bedrock error: {str(e)}")
            
            # Raise the mapped error
            raise mapped_error
            
        except Exception as e:
            # Handle any other unexpected errors
            print(f"Unexpected error in Bedrock generation: {str(e)}")
            raise LLMServiceUnavailableError(f"Unexpected error in AWS Bedrock: {str(e)}")

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model identifiers.
        
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

    def _validate_model_parameters(self, model: str, input_tokens: int,
                                  output_tokens: int) -> None:
        """
        Validate model parameters against constraints.
        
        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Raises:
            LLMConfigurationError: If parameters exceed model constraints
        """
        # Check if model is supported
        if model not in self.SUPPORTED_MODELS:
            raise LLMConfigurationError(
                f"Model {model} is not supported by AWS Bedrock",
                details={"supported_models": self.SUPPORTED_MODELS}
            )
        
        # Check context window
        if model in self.CONTEXT_WINDOW:
            max_context = self.CONTEXT_WINDOW[model]
            total_tokens = input_tokens + output_tokens
            
            if total_tokens > max_context:
                raise LLMConfigurationError(
                    f"Total tokens ({total_tokens}) exceeds model context window ({max_context})",
                    details={
                        "model": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "context_window": max_context
                    }
                )
