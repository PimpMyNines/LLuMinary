"""
Tests for OpenAI provider empty response handling with proper setup/teardown.

This module focuses specifically on testing the handling of empty
or invalid responses from the OpenAI API.
"""

from unittest.mock import MagicMock

import pytest

from lluminary.exceptions import InvalidResponseError
from lluminary.models.providers.openai import OpenAILLM


class TestOpenAIEmptyResponseHandling:
    """Test OpenAI empty response handling with proper setup/teardown."""

    def setup_method(self):
        """Set up the test environment."""
        # Create provider with mocked client
        self.provider = OpenAILLM("gpt-4o")
        self.provider.client = MagicMock()

        # Set provider as authenticated
        self.provider.authenticated = True

    def test_empty_response_handling(self):
        """Test handling of empty response from OpenAI API."""
        # Mock response with empty choices list
        self.provider.client.chat.completions.create.return_value = {
            "choices": []  # Empty choices list
        }

        # Call generate and expect exception
        with pytest.raises(InvalidResponseError) as exc_info:
            self.provider.generate(
                messages=[{"role": "user", "content": "Test message"}]
            )

        # Verify error message
        assert "Empty response received from OpenAI API" in str(exc_info.value)

    def test_missing_message_content_handling(self):
        """Test handling of response with missing message content."""
        # Mock response with missing content field
        self.provider.client.chat.completions.create.return_value = {
            "choices": [{"message": {}}]  # Missing content field
        }

        # Call generate and expect exception
        with pytest.raises(InvalidResponseError) as exc_info:
            self.provider.generate(
                messages=[{"role": "user", "content": "Test message"}]
            )

        # Verify error message
        assert "Invalid response format" in str(exc_info.value)

    def test_invalid_response_format_handling(self):
        """Test handling of response with invalid format."""
        # Mock response with completely invalid format
        self.provider.client.chat.completions.create.return_value = {
            "invalid_key": "invalid_value"  # Missing required structure
        }

        # Call generate and expect exception
        with pytest.raises(InvalidResponseError) as exc_info:
            self.provider.generate(
                messages=[{"role": "user", "content": "Test message"}]
            )

        # Verify error message
        assert "Invalid response format" in str(exc_info.value)

    def test_null_response_handling(self):
        """Test handling of null response from OpenAI API."""
        # Mock null response
        self.provider.client.chat.completions.create.return_value = None

        # Call generate and expect exception
        with pytest.raises(InvalidResponseError) as exc_info:
            self.provider.generate(
                messages=[{"role": "user", "content": "Test message"}]
            )

        # Verify error message
        assert "Null response received from OpenAI API" in str(exc_info.value)

    def test_empty_content_handling(self):
        """Test handling of empty content in response."""
        # Mock response with empty content
        self.provider.client.chat.completions.create.return_value = {
            "choices": [{"message": {"content": ""}}]  # Empty content
        }

        # Call generate - should return empty string without error
        response = self.provider.generate(
            messages=[{"role": "user", "content": "Test message"}]
        )

        # Verify empty response is returned
        assert response == ""

    def test_whitespace_only_content_handling(self):
        """Test handling of whitespace-only content in response."""
        # Mock response with whitespace-only content
        self.provider.client.chat.completions.create.return_value = {
            "choices": [{"message": {"content": "   \n   "}}]  # Whitespace-only content
        }

        # Call generate - should return whitespace without error
        response = self.provider.generate(
            messages=[{"role": "user", "content": "Test message"}]
        )

        # Verify whitespace response is returned
        assert response == "   \n   "

    def test_tool_call_without_content_handling(self):
        """Test handling of tool call response without content."""
        # Mock response with tool calls but no content
        self.provider.client.chat.completions.create.return_value = {
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

        # Call generate with tools enabled - should process tool calls without error
        response = self.provider.generate(
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

    def test_missing_choices_key_handling(self):
        """Test handling of response with missing choices key."""
        # Mock response with missing choices key
        self.provider.client.chat.completions.create.return_value = {
            "id": "response-123",
            "model": "gpt-4o",
            # No choices key
        }

        # Call generate and expect exception
        with pytest.raises(InvalidResponseError) as exc_info:
            self.provider.generate(
                messages=[{"role": "user", "content": "Test message"}]
            )

        # Verify error message
        assert "Invalid response format" in str(exc_info.value)

    def test_complex_response_validation(self):
        """Test validation of complex response structure."""
        # Mock response with multiple choices but no content
        self.provider.client.chat.completions.create.return_value = {
            "choices": [
                {"index": 0, "finish_reason": "stop"},  # No message key
                {"index": 1, "message": {"role": "assistant"}},  # No content key
            ]
        }

        # Call generate and expect exception
        with pytest.raises(InvalidResponseError) as exc_info:
            self.provider.generate(
                messages=[{"role": "user", "content": "Test message"}]
            )

        # Verify error message
        assert "Invalid response format" in str(exc_info.value)
