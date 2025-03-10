"""
Typing tests for verifying the type safety of shared type definitions.

This module provides examples of how to use the shared type definitions from types.py
and verifies that they can be properly type-checked with mypy in strict mode.

To run type checking on this file:
    python -m mypy --strict src/lluminary/models/types_test.py
"""

from typing import Any, Dict, List, cast

from .types import (
    BaseAPIRequest,
    BaseAPIResponse,
    BaseAssistantMessage,
    BaseImageContent,
    BaseImageSource,
    BaseMessage,
    BaseParameterProperty,
    BaseParameters,
    BaseSystemMessage,
    BaseTextContent,
    BaseToolCall,
    BaseToolCallData,
    BaseToolDefinition,
    BaseToolResultContent,
    BaseToolUseContent,
    BaseUsageStatistics,
    BaseUserMessage,
    ErrorDetails,
)


def test_message_types() -> None:
    """Test message type structures."""
    # Create a basic user message
    user_message: BaseUserMessage = {
        "role": "user",
        "content": "Hello, world!",
    }

    # Create a user message with structured content
    text_content: BaseTextContent = {
        "type": "text",
        "text": "This is a text message",
    }

    image_source: BaseImageSource = {
        "url": "https://example.com/image.jpg",
        "mime_type": "image/jpeg",
    }

    image_content: BaseImageContent = {
        "type": "image",
        "source": image_source,
    }

    # This would be a content list in practice
    content_list: List[Dict[str, Any]] = [
        cast(Dict[str, Any], text_content),
        cast(Dict[str, Any], image_content),
    ]

    structured_user_message: BaseUserMessage = {
        "role": "user",
        "content": content_list,
    }

    # Create a system message
    system_message: BaseSystemMessage = {
        "role": "system",
        "content": "You are a helpful assistant.",
    }

    # Create an assistant message
    assistant_message: BaseAssistantMessage = {
        "role": "assistant",
        "content": "I'm here to help!",
    }

    # Create a message list
    messages: List[BaseMessage] = [
        user_message,
        system_message,
        assistant_message,
    ]

    # Verify message structure
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], str)
    assert messages[1]["role"] == "system"
    assert isinstance(messages[1]["content"], str)
    assert messages[2]["role"] == "assistant"
    assert isinstance(messages[2]["content"], str)


def test_tool_types() -> None:
    """Test tool type structures."""
    # Create a tool parameter property
    string_param: BaseParameterProperty = {
        "type": "string",
        "description": "A string parameter",
    }

    number_param: BaseParameterProperty = {
        "type": "number",
        "description": "A number parameter",
        "minimum": 0,
        "maximum": 100,
    }

    # Create parameters
    params: BaseParameters = {
        "type": "object",
        "properties": {
            "name": string_param,
            "age": number_param,
        },
        "required": ["name"],
    }

    # Create a tool definition
    tool_def: BaseToolDefinition = {
        "name": "get_user_info",
        "description": "Get information about a user",
        "input_schema": cast(Dict[str, Any], params),
    }

    # Create a tool call
    tool_call: BaseToolCall = {
        "name": "get_user_info",
        "input": {"name": "John", "age": 30},
        "id": "call_123456",
    }

    # Create tool call data
    tool_call_data: BaseToolCallData = {
        "name": "get_user_info",
        "input": {"name": "John", "age": 30},
        "id": "call_123456",
        "tool_id": "tool_123",
        "result": {"name": "John Doe", "age": 30, "email": "john@example.com"},
    }

    # Create tool use content
    tool_use_content: BaseToolUseContent = {
        "type": "tool_use",
        "name": "get_user_info",
        "input": {"name": "John", "age": 30},
        "tool_id": "tool_123",
    }

    # Create tool result content
    tool_result_content: BaseToolResultContent = {
        "type": "tool_result",
        "tool_id": "tool_123",
        "result": {"name": "John Doe", "age": 30, "email": "john@example.com"},
    }

    # Verify tool structure
    assert tool_def["name"] == "get_user_info"
    assert "name" in params["properties"]
    assert "age" in params["properties"]
    assert params["required"] == ["name"]
    assert tool_call["name"] == "get_user_info"
    assert tool_call["input"]["name"] == "John"
    assert tool_call_data["result"] is not None
    assert tool_use_content["type"] == "tool_use"
    assert tool_result_content["type"] == "tool_result"


def test_api_structures() -> None:
    """Test API request and response structures."""
    # Create an API request
    request: BaseAPIRequest = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello, world!"},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "tools": [
            {
                "name": "get_user_info",
                "description": "Get information about a user",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "User name"},
                    },
                    "required": ["name"],
                },
            }
        ],
        "stream": False,
    }

    # Create an API response
    response: BaseAPIResponse = {
        "id": "resp_123456",
        "object": "chat.completion",
        "created": 1625097683,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 12,
            "total_tokens": 22,
        },
    }

    # Verify API structures
    assert request["model"] == "gpt-4"
    assert response["id"] == "resp_123456"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(request["max_tokens"], int)
    assert isinstance(request["temperature"], float)


def test_error_structures() -> None:
    """Test error detail structures."""
    # Create error details
    error_details: ErrorDetails = {
        "error": "Authentication failed",
        "error_type": "AuthenticationError",
        "provider": "openai",
        "model": "gpt-4",
        "status_code": 401,
        "request_id": "req_123456",
    }

    # Verify error structures
    assert error_details["error"] == "Authentication failed"
    assert error_details["provider"] == "openai"
    assert error_details["status_code"] == 401


def test_usage_statistics() -> None:
    """Test usage statistics structures."""
    # Create usage statistics
    usage: BaseUsageStatistics = {
        "read_tokens": 10,
        "write_tokens": 15,
        "total_tokens": 25,
        "read_cost": 0.0001,
        "write_cost": 0.0003,
        "total_cost": 0.0004,
        "images": 0,
        "image_cost": 0.0,
        "retry_count": 0,
        "event_id": "event_123456",
        "model": "gpt-4",
        "provider": "openai",
        "successful": True,
    }

    # Failed case with error
    failed_usage: BaseUsageStatistics = {
        "read_tokens": 5,
        "write_tokens": 0,
        "total_tokens": 5,
        "read_cost": 0.00005,
        "write_cost": 0.0,
        "total_cost": 0.00005,
        "images": 0,
        "image_cost": 0.0,
        "retry_count": 3,
        "event_id": "event_123457",
        "model": "gpt-4",
        "provider": "openai",
        "successful": False,
        "error": "Rate limit exceeded",
        "error_type": "RateLimitError",
    }

    # Verify usage statistics
    assert usage["total_tokens"] == 25
    assert usage["successful"] is True
    assert failed_usage["successful"] is False
    assert failed_usage["retry_count"] == 3
    assert failed_usage["error"] == "Rate limit exceeded"
