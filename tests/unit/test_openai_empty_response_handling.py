"""
Tests for OpenAI provider empty response handling.

This module focuses specifically on testing the handling of empty
or invalid responses from the OpenAI API.
"""

from unittest.mock import MagicMock

import pytest

from lluminary.exceptions import InvalidResponseError
from lluminary.models.providers.openai import OpenAILLM


def test_empty_response_handling():
    """Test handling of empty response from OpenAI API."""
    provider = OpenAILLM("gpt-4o")

    # Mock client and response
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = {
        "choices": []  # Empty choices list
    }

    # Set up provider state
    provider.authenticated = True

    # Call generate and expect exception
    with pytest.raises(InvalidResponseError) as exc_info:
        provider.generate(messages=[{"role": "user", "content": "Test message"}])

    # Verify error message
    assert "Empty response received from OpenAI API" in str(exc_info.value)


def test_missing_message_content_handling():
    """Test handling of response with missing message content."""
    provider = OpenAILLM("gpt-4o")

    # Mock client and response with missing content
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = {
        "choices": [{"message": {}}]  # Missing content field
    }

    # Set up provider state
    provider.authenticated = True

    # Call generate and expect exception
    with pytest.raises(InvalidResponseError) as exc_info:
        provider.generate(messages=[{"role": "user", "content": "Test message"}])

    # Verify error message
    assert "Invalid response format" in str(exc_info.value)


def test_invalid_response_format_handling():
    """Test handling of response with invalid format."""
    provider = OpenAILLM("gpt-4o")

    # Mock client and response with completely invalid format
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = {
        "invalid_key": "invalid_value"  # Missing required structure
    }

    # Set up provider state
    provider.authenticated = True

    # Call generate and expect exception
    with pytest.raises(InvalidResponseError) as exc_info:
        provider.generate(messages=[{"role": "user", "content": "Test message"}])

    # Verify error message
    assert "Invalid response format" in str(exc_info.value)


def test_null_response_handling():
    """Test handling of null response from OpenAI API."""
    provider = OpenAILLM("gpt-4o")

    # Mock client and response
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = None  # Null response

    # Set up provider state
    provider.authenticated = True

    # Call generate and expect exception
    with pytest.raises(InvalidResponseError) as exc_info:
        provider.generate(messages=[{"role": "user", "content": "Test message"}])

    # Verify error message
    assert "Null response received from OpenAI API" in str(exc_info.value)


def test_empty_content_handling():
    """Test handling of empty content in response."""
    provider = OpenAILLM("gpt-4o")

    # Mock client and response with empty content
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = {
        "choices": [{"message": {"content": ""}}]  # Empty content
    }

    # Set up provider state
    provider.authenticated = True

    # Call generate - should return empty string without error
    response = provider.generate(messages=[{"role": "user", "content": "Test message"}])

    # Verify empty response is returned
    assert response == ""


def test_whitespace_only_content_handling():
    """Test handling of whitespace-only content in response."""
    provider = OpenAILLM("gpt-4o")

    # Mock client and response with whitespace-only content
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = {
        "choices": [{"message": {"content": "   \n   "}}]  # Whitespace-only content
    }

    # Set up provider state
    provider.authenticated = True

    # Call generate - should return whitespace without error
    response = provider.generate(messages=[{"role": "user", "content": "Test message"}])

    # Verify whitespace response is returned
    assert response == "   \n   "


def test_tool_call_without_content_handling():
    """Test handling of tool call response without content."""
    provider = OpenAILLM("gpt-4o")

    # Mock client and response with tool calls but no content
    provider.client = MagicMock()
    provider.client.chat.completions.create.return_value = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}',
                            },
                        }
                    ]
                }
            }
        ]
    }

    # Set up provider state
    provider.authenticated = True

    # Call generate with tools enabled - should process tool calls without error
    response = provider.generate(
        messages=[{"role": "user", "content": "What's the weather in SF?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
    )

    # Verify tool call is processed and returned
    assert "tool_calls" in response
    assert response["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "San Francisco" in response["tool_calls"][0]["function"]["arguments"]
