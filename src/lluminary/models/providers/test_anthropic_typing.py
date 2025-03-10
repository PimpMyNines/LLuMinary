"""
Type compatibility tests for the Anthropic provider.

This module tests that our TypedDict definitions are compatible with
the functions that use them, and that we can instantiate and manipulate
these types correctly.
"""

import unittest

from ...models.providers.anthropic import (
    AnthropicImageContent,
    AnthropicImageSource,
    AnthropicMessage,
    AnthropicTextContent,
    AnthropicThinkingContent,
    AnthropicTool,
    AnthropicToolResultContent,
    AnthropicToolUseContent,
    ToolCallData,
)


class TestAnthropicTyping(unittest.TestCase):
    """Test cases for Anthropic type compatibility."""

    def test_image_source_typing(self):
        """Test that we can create and use AnthropicImageSource objects."""
        image_source: AnthropicImageSource = {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "base64_encoded_string",
        }

        # Verify we can access fields
        self.assertEqual(image_source["type"], "base64")
        self.assertEqual(image_source["media_type"], "image/jpeg")
        self.assertEqual(image_source["data"], "base64_encoded_string")

    def test_content_part_typing(self):
        """Test that we can create and use different content part objects."""
        # Text content
        text_content: AnthropicTextContent = {"type": "text", "text": "Hello world"}

        # Image content
        image_content: AnthropicImageContent = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "base64_encoded_string",
            },
        }

        # Tool use content
        tool_use_content: AnthropicToolUseContent = {
            "type": "tool_use",
            "id": "tool-123",
            "name": "get_weather",
            "input": {"location": "San Francisco"},
        }

        # Tool result content
        tool_result_content: AnthropicToolResultContent = {
            "type": "tool_result",
            "tool_use_id": "tool-123",
            "content": "Tool Call Successful: 72°F and sunny",
        }

        # Thinking content
        thinking_content: AnthropicThinkingContent = {
            "type": "thinking",
            "thinking": "I need to consider all options",
            "signature": "abcd1234",
        }

        # Verify we can access fields
        self.assertEqual(text_content["text"], "Hello world")
        self.assertEqual(image_content["source"]["media_type"], "image/jpeg")
        self.assertEqual(tool_use_content["name"], "get_weather")
        self.assertEqual(
            tool_result_content["content"], "Tool Call Successful: 72°F and sunny"
        )
        self.assertEqual(thinking_content["thinking"], "I need to consider all options")

    def test_message_typing(self):
        """Test that we can create and use AnthropicMessage objects."""
        message: AnthropicMessage = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello Claude"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "base64_encoded_string",
                    },
                },
            ],
        }

        # Verify we can access fields
        self.assertEqual(message["role"], "user")
        self.assertEqual(len(message["content"]), 2)
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertEqual(message["content"][0]["text"], "Hello Claude")
        self.assertEqual(message["content"][1]["type"], "image")

    def test_tool_typing(self):
        """Test that we can create and use AnthropicTool objects."""
        tool: AnthropicTool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        }
                    },
                    "required": ["location"],
                },
            },
        }

        # Verify we can access fields
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "get_weather")
        self.assertEqual(tool["function"]["parameters"]["type"], "object")

    def test_tool_call_data_typing(self):
        """Test that we can create and use ToolCallData objects."""
        tool_call: ToolCallData = {
            "id": "call-123",
            "name": "get_weather",
            "arguments": '{"location": "San Francisco"}',
            "type": "function",
        }

        # Verify we can access fields
        self.assertEqual(tool_call["id"], "call-123")
        self.assertEqual(tool_call["name"], "get_weather")
        self.assertEqual(tool_call["type"], "function")


if __name__ == "__main__":
    unittest.main()
