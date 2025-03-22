"""
Shared type definitions for LLM providers.

This module defines common TypedDict structures that can be shared across different
provider implementations. It establishes a standard structure for common objects like
message formats, content parts, tools, and API request/response structures.

Provider-specific implementations should extend these base types to add any
provider-specific fields while maintaining compatibility with the shared structure.

The type definitions are organized into several categories:
1. Content part types - Individual components of messages (text, images, tool use)
2. Message types - Complete message structures with different roles 
3. Tool and function definitions - Schemas for function/tool calling
4. API request and response structures - Standardized API interfaces
5. Usage statistics - Unified tracking of token usage and costs
6. Error handling - Consistent error reporting structures

This creates a unified type system that works across all providers while maintaining
their individual capabilities.
"""

from enum import Enum
from typing import (
    Any, 
    Dict, 
    List, 
    Literal, 
    Optional, 
    Protocol, 
    Tuple, 
    TypedDict, 
    Union,
    cast,
    Iterator
)


# Provider enum for consistent provider identification
class Provider(str, Enum):
    """Enumeration of supported LLM providers."""
    
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    BEDROCK = "bedrock"
    COHERE = "cohere"
    MISTRAL = "mistral"  # Future implementation


# Content Types

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


# Union type for all content part types
ContentType = Union[
    BaseTextContent,
    BaseImageContent,
    BaseToolUseContent,
    BaseToolResultContent,
    BaseThinkingContent
]


# Standard role types for better type checking
RoleType = Literal["user", "assistant", "system", "tool", "function"]


# Standard image formats
ImageFormat = Literal["jpeg", "png", "webp", "gif"]


# Message types - Core message interface
class BaseMessage(TypedDict, total=False):
    """Base type for messages across providers."""

    role: RoleType
    content: Union[str, List[ContentType], List[Dict[str, Any]]]
    name: Optional[str]  # For named messages (e.g., function calls in some providers)
    

# User-specific message type
class BaseUserMessageDict(TypedDict):
    """Base type for user message role."""

    role: Literal["user"]


class BaseUserMessage(BaseMessage, BaseUserMessageDict):
    """Base type for user messages."""

    content: Union[str, List[ContentType]]


# Assistant-specific message type
class BaseAssistantMessageDict(TypedDict):
    """Base type for assistant message role."""

    role: Literal["assistant"]


class BaseAssistantMessage(BaseMessage, BaseAssistantMessageDict):
    """Base type for assistant messages."""

    content: Union[str, List[ContentType], None]
    tool_calls: Optional[List[Dict[str, Any]]]


# System-specific message type
class BaseSystemMessageDict(TypedDict):
    """Base type for system message role."""

    role: Literal["system"]


class BaseSystemMessage(BaseMessage, BaseSystemMessageDict):
    """Base type for system messages."""

    content: Union[str, List[ContentType]]


# Tool-specific message type
class BaseToolMessageDict(TypedDict):
    """Base type for tool message role."""

    role: Literal["tool"]
    tool_id: str
    tool_call_id: Optional[str]  # For providers that use tool_call_id


class BaseToolMessage(BaseMessage, BaseToolMessageDict):
    """Base type for tool messages."""

    content: Union[str, Dict[str, Any]]


# Message collections
MessageList = List[Union[BaseUserMessage, BaseAssistantMessage, BaseSystemMessage, BaseToolMessage]]


# Input/Output message formats
class StandardMessageInput(TypedDict, total=False):
    """Standardized input message format used across the library."""
    
    message_type: Literal["human", "ai", "system", "tool"]
    message: str
    image_paths: Optional[List[str]]
    image_urls: Optional[List[str]]
    tool_use: Optional[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    thinking: Optional[Dict[str, str]]
    name: Optional[str]


class StandardMessageOutput(TypedDict, total=False):
    """Standardized output message format returned by providers."""
    
    message_type: Literal["human", "ai", "system", "tool"]
    message: str
    tool_calls: Optional[List[Dict[str, Any]]]
    finished_reason: Optional[str]
    usage: Optional[Dict[str, Any]]
    model: str
    provider: Provider


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
    items: Optional[Dict[str, Any]]  # For array types
    properties: Optional[Dict[str, Any]]  # For nested object types
    required: Optional[List[str]]  # For nested object types


class BaseParameters(TypedDict, total=False):
    """Base type for tool parameters structure."""

    type: str
    properties: Dict[str, "BaseParameterProperty"]
    required: List[str]
    description: Optional[str]  # Some providers allow description at parameter level


class BaseToolDefinition(TypedDict, total=False):
    """Base type for tool definition across providers."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    required: Optional[bool]  # Whether the tool is required


class BaseFunctionDefinition(TypedDict, total=False):
    """Base type for function definition across providers."""

    name: str
    description: str
    parameters: "BaseParameters"
    required: Optional[bool]  # Whether the function is required


# Tool type definitions
ToolType = Literal["function", "tool", "retrieval", "code_interpreter"]


# Tool call types - Used during function/tool calling
class BaseToolCall(TypedDict, total=False):
    """Base type for tool call information."""

    name: str
    input: Dict[str, Any]
    id: Optional[str]
    type: Optional[ToolType]


class BaseToolCallData(TypedDict, total=False):
    """Base type for accumulated tool call data."""

    name: str
    input: Dict[str, Any]
    id: str
    tool_id: str
    type: Optional[ToolType]
    result: Optional[Any]
    error: Optional[str]
    status: Optional[Literal["success", "error"]]


# Streaming tool call types
class BaseStreamingToolCall(TypedDict, total=False):
    """Base type for streaming tool call information."""
    
    id: str
    name: str
    type: ToolType
    input: Dict[str, Any]
    is_complete: bool  # Indicates if the tool call is complete
    partial_input: Optional[str]  # For partial arguments in streaming mode


class BaseToolCallChoice(TypedDict, total=False):
    """Base type for tool call choice in API responses."""
    
    index: int
    tool_call: BaseToolCall
    finish_reason: Optional[Literal["tool_calls", "stop", "length", "content_filter"]]


# API request and response structures
class BaseAPIRequest(TypedDict, total=False):
    """Base type for API request parameters."""

    model: str
    messages: List[Union[BaseMessage, Dict[str, Any]]]
    max_tokens: int
    temperature: float
    tools: List[Dict[str, Any]]
    functions: List[Dict[str, Any]]  # For compatibility with older APIs
    stream: bool
    stop: Union[str, List[str]]
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    response_format: Optional[Dict[str, Any]]
    seed: Optional[int]
    timeout: Optional[int]
    tool_choice: Optional[Union[str, Dict[str, Any]]]
    function_call: Optional[Union[str, Dict[str, Any]]]  # For compatibility with older APIs


class BaseGenerationOptions(TypedDict, total=False):
    """Standardized generation options used across providers."""
    
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: Union[str, List[str]]
    timeout: int
    stream: bool
    response_format: Dict[str, Any]
    seed: int
    tools: List[Dict[str, Any]]
    tool_choice: Union[str, Dict[str, Any]]


class BaseAPIResponseChoice(TypedDict, total=False):
    """Base type for API response choice."""
    
    index: int
    message: Union[BaseMessage, Dict[str, Any]]
    finish_reason: Optional[str]
    logprobs: Optional[Dict[str, Any]]
    

class BaseAPIResponse(TypedDict, total=False):
    """Base type for API response."""

    id: str
    object: str
    created: int
    model: str
    choices: List[Union[BaseAPIResponseChoice, Dict[str, Any]]]
    usage: Dict[str, Any]
    system_fingerprint: Optional[str]


# Streaming types
class BaseDelta(TypedDict, total=False):
    """Base type for streaming delta updates."""
    
    role: Optional[RoleType]
    content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    function_call: Optional[Dict[str, Any]]  # For compatibility with older APIs
    tool_call_id: Optional[str]  # For tool responses


class BaseStreamingChunk(TypedDict, total=False):
    """Base type for streaming response chunks."""
    
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    delta: Optional[BaseDelta]  # For providers using delta format


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
    request_timestamp: Optional[float]
    response_timestamp: Optional[float]
    latency_ms: Optional[float]
    thinking_tokens: Optional[int]  # For models with thinking budget


# Embeddings types
class BaseEmbeddingRequest(TypedDict, total=False):
    """Base type for embedding request parameters."""
    
    model: str
    input: Union[str, List[str]]
    dimensions: Optional[int]
    user: Optional[str]
    encoding_format: Optional[str]


class BaseEmbeddingResponse(TypedDict, total=False):
    """Base type for embedding response."""
    
    id: str
    object: str
    model: str
    data: List[Dict[str, Any]]
    usage: Dict[str, int]


class BaseEmbeddingData(TypedDict):
    """Base type for embedding data."""
    
    object: str
    embedding: List[float]
    index: int


# Reranking types
class BaseRerankingRequest(TypedDict, total=False):
    """Base type for reranking request parameters."""
    
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int]
    return_documents: Optional[bool]


class BaseRerankingResponse(TypedDict, total=False):
    """Base type for reranking response."""
    
    id: str
    model: str
    results: List[Dict[str, Any]]
    usage: Dict[str, int]


class BaseRerankingResult(TypedDict, total=False):
    """Base type for reranking result."""
    
    document: Optional[str]
    index: int
    relevance_score: float
    document_id: Optional[str]


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
    timestamp: Optional[float]
    retryable: Optional[bool]
    rate_limit_reset: Optional[int]
    quota_reset: Optional[str]
    operation: Optional[str]


# Function call conversion protocols
class FunctionCallProcessor(Protocol):
    """Protocol for function call processing."""
    
    def format_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tool calls for provider API."""
        ...
    
    def parse_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse tool calls from provider API response."""
        ...


# Authentication types
class BaseAuthConfig(TypedDict, total=False):
    """Base type for authentication configuration."""
    
    api_key: Optional[str]
    organization_id: Optional[str]
    base_url: Optional[str]
    aws_region: Optional[str]
    aws_profile: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    secret_name: Optional[str]
    service_account_file: Optional[str]
    project_id: Optional[str]
    auth_token: Optional[str]
