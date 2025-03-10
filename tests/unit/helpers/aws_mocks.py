"""
Mock responses for AWS services used in testing.

This module provides standardized mock responses for AWS Bedrock and other
AWS services to ensure consistent test behavior across all test modules.
"""

from typing import Any, Dict, List, Optional


def create_bedrock_converse_response(
    text: str = "This is a test response",
    input_tokens: int = 20,
    output_tokens: int = 15,
    completion_reason: str = "end_turn",
) -> Dict[str, Any]:
    """
    Create a mock response for the Bedrock converse API.

    Args:
        text: The response text
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        completion_reason: Reason for completion (end_turn, stop_sequence, etc.)

    Returns:
        Dict representing a Bedrock converse API response
    """
    return {
        "message": {"role": "assistant", "content": [{"text": text}]},
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
        "completionReason": completion_reason,
        "stopSequences": [],
    }


def create_bedrock_converse_stream_response(
    chunks: List[str] = None,
    input_tokens: int = 20,
    output_tokens: int = 15,
) -> Dict[str, Any]:
    """
    Create a mock response for the Bedrock converse_stream API.

    Args:
        chunks: List of text chunks to include in the stream
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated

    Returns:
        Dict with a stream generator
    """
    if chunks is None:
        chunks = ["This is ", "a test ", "response"]

    class MockStreamIterator:
        def __init__(self, chunks: List[str]):
            self.chunks = chunks
            self.current = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.current >= len(self.chunks):
                if self.current == len(self.chunks):
                    self.current += 1
                    return {"completionReason": "end_turn"}
                raise StopIteration

            chunk = self.chunks[self.current]
            self.current += 1
            return {"message": {"content": chunk}}

    return {
        "stream": MockStreamIterator(chunks),
        "contentType": "application/json",
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
    }


def create_bedrock_converse_with_tools_response(
    text: str = "I'll help you with that",
    tool_use: Optional[Dict[str, Any]] = None,
    input_tokens: int = 30,
    output_tokens: int = 25,
) -> Dict[str, Any]:
    """
    Create a mock response for the Bedrock converse API with tool use.

    Args:
        text: The response text
        tool_use: Optional tool use dictionary
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated

    Returns:
        Dict representing a Bedrock converse API response with tool use
    """
    content = [{"text": text}]

    if tool_use:
        content.append(
            {
                "toolUse": {
                    "toolUseId": tool_use.get("id", "tool-123"),
                    "name": tool_use.get("name", "test_function"),
                    "input": tool_use.get("input", {}),
                }
            }
        )

    return {
        "message": {"role": "assistant", "content": content},
        "usage": {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        },
        "completionReason": "tool_use" if tool_use else "end_turn",
        "stopSequences": [],
    }


def create_bedrock_list_models_response() -> Dict[str, Any]:
    """
    Create a mock response for the Bedrock list_foundation_models API.

    Returns:
        Dict representing a Bedrock list_foundation_models API response
    """
    return {
        "modelSummaries": [
            {
                "modelId": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                "modelName": "Claude 3.5 Sonnet",
                "providerName": "Anthropic",
                "inputModalities": ["TEXT", "IMAGE"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
                "customizationsSupported": ["FINE_TUNING"],
            },
            {
                "modelId": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "modelName": "Claude 3.7 Sonnet",
                "providerName": "Anthropic",
                "inputModalities": ["TEXT", "IMAGE"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
                "customizationsSupported": ["FINE_TUNING"],
            },
            {
                "modelId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                "modelName": "Claude 3.5 Haiku",
                "providerName": "Anthropic",
                "inputModalities": ["TEXT", "IMAGE"],
                "outputModalities": ["TEXT"],
                "responseStreamingSupported": True,
                "customizationsSupported": [],
            },
        ]
    }


def create_aws_client_error(
    code: str, message: str, operation_name: str = "converse"
) -> Dict[str, Any]:
    """
    Create a structure that mimics a boto3 ClientError.

    Args:
        code: The error code (e.g., "ThrottlingException")
        message: The error message
        operation_name: The operation that caused the error

    Returns:
        Dict with the error response structure
    """
    return {
        "Error": {"Code": code, "Message": message},
        "ResponseMetadata": {
            "RequestId": "12345678-1234-5678-1234-567812345678",
            "HTTPStatusCode": (
                400
                if code in ["ValidationException", "InvalidRequestException"]
                else (
                    429
                    if code in ["ThrottlingException", "TooManyRequestsException"]
                    else (
                        403
                        if code in ["AccessDeniedException", "UnauthorizedException"]
                        else (
                            503
                            if code
                            in [
                                "ServiceUnavailableException",
                                "InternalServerException",
                            ]
                            else 400
                        )
                    )
                )
            ),
            "HTTPHeaders": {
                "x-amzn-requestid": "12345678-1234-5678-1234-567812345678",
                "content-type": "application/json",
                "content-length": "123",
                "date": "Fri, 07 Mar 2025 12:00:00 GMT",
            },
            "RetryAttempts": 0,
        },
        "Operation": operation_name,
    }
