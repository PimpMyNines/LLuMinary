"""
Validation utilities for LLuMinary provider parameters.

This module contains validation functions for common parameters across providers,
helping to standardize error messages and improve the overall user experience.
"""

from typing import Any, Dict, List, Optional, Set

from ..exceptions import LLMValidationError


def validate_model_name(model_name: str, supported_models: List[str]) -> None:
    """
    Validate that a model name is in the list of supported models.

    Args:
        model_name: The model name to validate
        supported_models: List of supported model names

    Raises:
        LLMValidationError: If the model is not supported
    """
    if not model_name:
        raise LLMValidationError(
            "Model name cannot be empty",
            details={"supported_models": supported_models},
        )

    if model_name not in supported_models:
        raise LLMValidationError(
            f"Model '{model_name}' is not supported. Supported models: {', '.join(supported_models)}",
            details={
                "supported_models": supported_models,
                "provided_model": model_name,
            },
        )


def validate_messages(messages: List[Dict[str, Any]]) -> None:
    """
    Validate that the messages list has the expected format.

    Args:
        messages: List of messages to validate

    Raises:
        LLMValidationError: If messages are invalid
    """
    if not isinstance(messages, list):
        raise LLMValidationError(
            f"Messages must be a list, got {type(messages).__name__}",
            details={"provided_type": type(messages).__name__},
        )

    if not messages:
        raise LLMValidationError(
            "Messages list cannot be empty",
            details={"reason": "At least one message is required"},
        )

    required_keys = {"message_type", "message"}

    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise LLMValidationError(
                f"Message at index {i} must be a dictionary, got {type(message).__name__}",
                details={"index": i, "provided_type": type(message).__name__},
            )

        missing_keys = required_keys - set(message.keys())
        if missing_keys:
            raise LLMValidationError(
                f"Message at index {i} is missing required keys: {', '.join(missing_keys)}",
                details={
                    "index": i,
                    "missing_keys": list(missing_keys),
                    "message": message,
                },
            )

        valid_message_types = {"human", "ai", "system"}
        if message["message_type"] not in valid_message_types:
            raise LLMValidationError(
                f"Message at index {i} has invalid message_type: '{message['message_type']}'. "
                f"Valid types are: {', '.join(valid_message_types)}",
                details={
                    "index": i,
                    "invalid_type": message["message_type"],
                    "valid_types": list(valid_message_types),
                },
            )

        if not isinstance(message["message"], str):
            raise LLMValidationError(
                f"Message content at index {i} must be a string, got {type(message['message']).__name__}",
                details={
                    "index": i,
                    "provided_type": type(message["message"]).__name__,
                },
            )


def validate_temperature(temp: float) -> None:
    """
    Validate that the temperature is within the acceptable range.

    Args:
        temp: The temperature value to validate

    Raises:
        LLMValidationError: If temperature is invalid
    """
    if not isinstance(temp, (int, float)):
        raise LLMValidationError(
            f"Temperature must be a number, got {type(temp).__name__}",
            details={"provided_type": type(temp).__name__},
        )

    if temp < 0 or temp > 2:
        raise LLMValidationError(
            f"Temperature must be between 0 and 2, got {temp}",
            details={"provided_temp": temp, "valid_range": [0, 2]},
        )


def validate_max_tokens(max_tokens: int, context_window: Optional[int] = None) -> None:
    """
    Validate that max_tokens is a positive integer within context window limits if specified.

    Args:
        max_tokens: The max tokens value to validate
        context_window: Optional context window size to check against

    Raises:
        LLMValidationError: If max_tokens is invalid
    """
    if not isinstance(max_tokens, int):
        raise LLMValidationError(
            f"max_tokens must be an integer, got {type(max_tokens).__name__}",
            details={"provided_type": type(max_tokens).__name__},
        )

    if max_tokens <= 0:
        raise LLMValidationError(
            f"max_tokens must be positive, got {max_tokens}",
            details={"provided_value": max_tokens},
        )

    if context_window and max_tokens > context_window:
        raise LLMValidationError(
            f"max_tokens ({max_tokens}) exceeds the model's context window ({context_window})",
            details={"provided_value": max_tokens, "context_window": context_window},
        )


def validate_tools(tools: List[Dict[str, Any]]) -> None:
    """
    Validate that tools have the expected format.

    Args:
        tools: List of tool definitions to validate

    Raises:
        LLMValidationError: If tools are invalid
    """
    if not isinstance(tools, list):
        raise LLMValidationError(
            f"Tools must be a list, got {type(tools).__name__}",
            details={"provided_type": type(tools).__name__},
        )

    required_keys = {"name", "description"}

    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise LLMValidationError(
                f"Tool at index {i} must be a dictionary, got {type(tool).__name__}",
                details={"index": i, "provided_type": type(tool).__name__},
            )

        missing_keys = required_keys - set(tool.keys())
        if missing_keys:
            raise LLMValidationError(
                f"Tool at index {i} is missing required keys: {', '.join(missing_keys)}",
                details={"index": i, "missing_keys": list(missing_keys), "tool": tool},
            )

        if not isinstance(tool["name"], str):
            raise LLMValidationError(
                f"Tool name at index {i} must be a string, got {type(tool['name']).__name__}",
                details={"index": i, "provided_type": type(tool["name"]).__name__},
            )

        if not isinstance(tool["description"], str):
            raise LLMValidationError(
                f"Tool description at index {i} must be a string, got {type(tool['description']).__name__}",
                details={
                    "index": i,
                    "provided_type": type(tool["description"]).__name__,
                },
            )

        if "input_schema" in tool and not isinstance(tool["input_schema"], dict):
            raise LLMValidationError(
                f"Tool input_schema at index {i} must be a dictionary, got {type(tool['input_schema']).__name__}",
                details={
                    "index": i,
                    "provided_type": type(tool["input_schema"]).__name__,
                },
            )


def validate_categories(categories: Dict[str, str]) -> None:
    """
    Validate classification categories.

    Args:
        categories: Dictionary mapping category names to descriptions

    Raises:
        LLMValidationError: If categories are invalid
    """
    if not isinstance(categories, dict):
        raise LLMValidationError(
            f"Categories must be a dictionary, got {type(categories).__name__}",
            details={"provided_type": type(categories).__name__},
        )

    if not categories:
        raise LLMValidationError(
            "Categories dictionary cannot be empty",
            details={"reason": "At least one category is required"},
        )

    for name, description in categories.items():
        if not isinstance(name, str):
            raise LLMValidationError(
                f"Category name must be a string, got {type(name).__name__}",
                details={"provided_type": type(name).__name__, "value": str(name)},
            )

        if not isinstance(description, str):
            raise LLMValidationError(
                f"Category description for '{name}' must be a string, got {type(description).__name__}",
                details={"category": name, "provided_type": type(description).__name__},
            )


def validate_provider_config(
    config: Dict[str, Any],
    required_keys: Set[str] = None,
    optional_keys: Set[str] = None,
) -> None:
    """
    Validate a provider configuration dictionary.

    Args:
        config: The configuration dictionary to validate
        required_keys: Set of keys that must be present
        optional_keys: Set of keys that may be present but are not required

    Raises:
        LLMValidationError: If the configuration is invalid
    """
    if not isinstance(config, dict):
        raise LLMValidationError(
            f"Configuration must be a dictionary, got {type(config).__name__}",
            details={"provided_type": type(config).__name__},
        )

    if required_keys:
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise LLMValidationError(
                f"Configuration is missing required keys: {', '.join(missing_keys)}",
                details={
                    "missing_keys": list(missing_keys),
                    "provided_keys": list(config.keys()),
                },
            )

    if required_keys and optional_keys:
        allowed_keys = required_keys.union(optional_keys)
        extra_keys = set(config.keys()) - allowed_keys
        if extra_keys:
            # This is a warning rather than an error
            print(
                f"Warning: Configuration contains unexpected keys: {', '.join(extra_keys)}"
            )


def validate_image_path(image_path: str) -> None:
    """
    Validate that an image path exists and is readable.

    Args:
        image_path: Path to validate

    Raises:
        LLMValidationError: If the image path is invalid
    """
    import os

    if not isinstance(image_path, str):
        raise LLMValidationError(
            f"Image path must be a string, got {type(image_path).__name__}",
            details={"provided_type": type(image_path).__name__},
        )

    if not os.path.exists(image_path):
        raise LLMValidationError(
            f"Image file does not exist: {image_path}",
            details={"path": image_path},
        )

    if not os.path.isfile(image_path):
        raise LLMValidationError(
            f"Image path is not a file: {image_path}",
            details={"path": image_path},
        )

    if not os.access(image_path, os.R_OK):
        raise LLMValidationError(
            f"Image file is not readable: {image_path}",
            details={"path": image_path},
        )


def validate_image_url(image_url: str) -> None:
    """
    Validate that an image URL has a valid format.

    Args:
        image_url: URL to validate

    Raises:
        LLMValidationError: If the image URL is invalid
    """
    import re

    if not isinstance(image_url, str):
        raise LLMValidationError(
            f"Image URL must be a string, got {type(image_url).__name__}",
            details={"provided_type": type(image_url).__name__},
        )

    # Basic URL validation
    url_pattern = re.compile(
        r"^(https?|ftp)://"  # http://, https://, ftp://
        r"([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"  # domain
        r"(:[0-9]+)?"  # optional port
        r"(/[^/\s]+)*/?$"  # path
    )

    if not url_pattern.match(image_url):
        raise LLMValidationError(
            f"Invalid image URL format: {image_url}",
            details={"url": image_url},
        )


def validate_api_key(api_key: str, provider_name: str) -> None:
    """
    Validate that an API key has a valid format.

    Args:
        api_key: The API key to validate
        provider_name: Name of the provider (for error messages)

    Raises:
        LLMValidationError: If the API key is invalid
    """
    if not isinstance(api_key, str):
        raise LLMValidationError(
            f"{provider_name} API key must be a string, got {type(api_key).__name__}",
            details={
                "provider": provider_name,
                "provided_type": type(api_key).__name__,
            },
        )

    if not api_key:
        raise LLMValidationError(
            f"{provider_name} API key cannot be empty",
            details={"provider": provider_name},
        )

    # Basic validation based on common API key formats
    provider_specific_validation = {
        "openai": lambda key: len(key) > 20 and key.startswith(("sk-", "org-")),
        "anthropic": lambda key: len(key) > 20,
        "cohere": lambda key: len(key) > 20,
        "google": lambda key: len(key) > 20,
    }

    if provider_name.lower() in provider_specific_validation:
        validator = provider_specific_validation[provider_name.lower()]
        if not validator(api_key):
            raise LLMValidationError(
                f"Invalid {provider_name} API key format",
                details={"provider": provider_name},
            )
