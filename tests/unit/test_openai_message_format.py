"""
Tests for OpenAI provider message formatting.
"""

from unittest.mock import patch

from lluminary.models.providers.openai import OpenAILLM


def test_format_basic_messages():
    """Test formatting basic messages."""
    with patch.object(OpenAILLM, "auth"):
        # Create OpenAI LLM instance
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Test human/user message
        messages = [{"message_type": "human", "message": "hello"}]
        formatted = llm._format_messages_for_model(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"] == "hello"

        # Test AI/assistant message
        messages = [{"message_type": "ai", "message": "I can help with that"}]
        formatted = llm._format_messages_for_model(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "assistant"
        assert formatted[0]["content"] == "I can help with that"

        # Test system message
        messages = [
            {"message_type": "system", "message": "You are a helpful assistant"}
        ]
        formatted = llm._format_messages_for_model(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == "You are a helpful assistant"


def test_format_tool_messages():
    """Test formatting tool-related messages."""
    with patch.object(OpenAILLM, "auth"):
        # Create OpenAI LLM instance
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Test assistant message with tool use
        messages = [
            {
                "message_type": "ai",
                "message": "I'll use the calculator",
                "tool_use": {
                    "id": "tool-123",
                    "name": "calculator",
                    "input": {"x": 10, "y": 5},
                },
            }
        ]

        formatted = llm._format_messages_for_model(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "assistant"
        assert formatted[0]["content"] == "I'll use the calculator"
        assert "tool_calls" in formatted[0]
        assert formatted[0]["tool_calls"][0]["id"] == "tool-123"
        assert formatted[0]["tool_calls"][0]["function"]["name"] == "calculator"

        # Test tool result message
        messages = [
            {
                "message_type": "tool_result",
                "tool_result": {
                    "tool_id": "tool-123",
                    "success": True,
                    "result": "The result is 15",
                },
            }
        ]

        formatted = llm._format_messages_for_model(messages)

        assert len(formatted) == 1
        assert formatted[0]["role"] == "tool"
        assert "tool_call_id" in formatted[0]
        assert formatted[0]["tool_call_id"] == "tool-123"
        assert "Tool Call Successful" in formatted[0]["content"]
