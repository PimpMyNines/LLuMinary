"""
Shared type definitions for LLM providers.

This module defines common TypedDict structures that can be shared across different
provider implementations. It establishes a standard structure for common objects like
message formats, content parts, tools, and API request/response structures.

Provider-specific implementations should extend these base types to add any
provider-specific fields while maintaining compatibility with the shared structure.
"""

from typing import Any, Dict, List, Literal, Optional, Protocol, TypedDict, Union


# Basic content part types
class BaseTextContent(TypedDict):
    """Base type for text content across providers."""

    type: Literal["text"]
    text: str


class BaseImageSource(TypedDict, total=False):
    """Base type for image source specification."""

    url: Optional[str]
    path: Optional[str]
    base64: Optional[str]
    mime_type: Optional[str]


class BaseImageContent(TypedDict):
    """Base type for image content across providers."""

    type: Literal["image"]
    source: Union[str, Dict[str, Any], "BaseImageSource"]


class BaseToolUseContent(TypedDict):
    """Base type for tool use content across providers."""

    type: Literal["tool_use"]
    name: str
    input: Dict[str, Any]
    tool_id: Optional[str]


class BaseToolResultContent(TypedDict):
    """Base type for tool result content across providers."""

    type: Literal["tool_result"]
    tool_id: str
    result: Dict[str, Any]


class BaseThinkingContent(TypedDict):
    """Base type for thinking/reasoning content across providers."""

    type: Literal["thinking"]
    thinking: str


# Message types
class BaseMessage(TypedDict, total=False):
    """Base type for messages across providers."""

    role: str
    content: Union[str, List[Dict[str, Any]]]


# User-specific message type
class BaseUserMessageDict(TypedDict):
    """Base type for user message role."""

    role: Literal["user"]


class BaseUserMessage(BaseMessage, BaseUserMessageDict):
    """Base type for user messages."""

    pass


# Assistant-specific message type
class BaseAssistantMessageDict(TypedDict):
    """Base type for assistant message role."""

    role: Literal["assistant"]


class BaseAssistantMessage(BaseMessage, BaseAssistantMessageDict):
    """Base type for assistant messages."""

    pass


# System-specific message type
class BaseSystemMessageDict(TypedDict):
    """Base type for system message role."""

    role: Literal["system"]


class BaseSystemMessage(BaseMessage, BaseSystemMessageDict):
    """Base type for system messages."""

    pass


# Tool-specific message type
class BaseToolMessageDict(TypedDict):
    """Base type for tool message role."""

    role: Literal["tool"]
    tool_id: str


class BaseToolMessage(BaseMessage, BaseToolMessageDict):
    """Base type for tool messages."""

    pass


# Tool and function definitions
class BaseParameterProperty(TypedDict, total=False):
    """Base type for tool parameter property definition."""

    type: str
    description: Optional[str]
    enum: Optional[List[str]]
    format: Optional[str]
    minimum: Optional[int]
    maximum: Optional[int]
    default: Optional[Any]


class BaseParameters(TypedDict, total=False):
    """Base type for tool parameters structure."""

    type: str
    properties: Dict[str, "BaseParameterProperty"]
    required: List[str]


class BaseToolDefinition(TypedDict):
    """Base type for tool definition across providers."""

    name: str
    description: str
    input_schema: Dict[str, Any]


class BaseFunctionDefinition(TypedDict):
    """Base type for function definition across providers."""

    name: str
    description: str
    parameters: "BaseParameters"


# Tool call types
class BaseToolCall(TypedDict):
    """Base type for tool call information."""

    name: str
    input: Dict[str, Any]
    id: Optional[str]


class BaseToolCallData(TypedDict, total=False):
    """Base type for accumulated tool call data."""

    name: str
    input: Dict[str, Any]
    id: str
    tool_id: str
    result: Optional[Any]


# API request and response structures
class BaseAPIRequest(TypedDict, total=False):
    """Base type for API request parameters."""

    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int
    temperature: float
    tools: List[Dict[str, Any]]
    stream: bool
    stop: Union[str, List[str]]
    top_p: float
    frequency_penalty: float
    presence_penalty: float


class BaseAPIResponse(TypedDict, total=False):
    """Base type for API response."""

    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]


# Usage statistics
class BaseUsageStatistics(TypedDict, total=False):
    """Base type for usage statistics across providers."""

    read_tokens: int
    write_tokens: int
    total_tokens: int
    read_cost: float
    write_cost: float
    total_cost: float
    images: int
    image_cost: float
    retry_count: int
    event_id: str
    model: str
    provider: str
    successful: bool
    error: Optional[str]
    error_type: Optional[str]


# Protocol classes for external libs
class SecretManagerClient(Protocol):
    """Protocol for AWS Secrets Manager client."""

    def get_secret_value(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get secret value from AWS Secrets Manager."""
        ...


# Error detail types
class ErrorDetails(TypedDict, total=False):
    """Standardized error details structure."""

    error: str
    error_type: str
    model: str
    provider: str
    status_code: int
    retry_after: Optional[int]
    request_id: Optional[str]
    original_error: Optional[str]
