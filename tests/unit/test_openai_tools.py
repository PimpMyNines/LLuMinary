"""
Tests for OpenAI provider tool handling.
"""

from unittest.mock import patch

from lluminary.models.providers.openai import OpenAILLM


def test_format_tools_for_model():
    """Test formatting of tools for OpenAI models."""
    with patch.object(OpenAILLM, "auth"):
        # Create OpenAI LLM instance
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Define test function
        def test_tool(x: int) -> int:
            """Test tool with docstring."""
            return x * 2

        # Format the function as a tool
        tools = [test_tool]
        formatted = llm._format_tools_for_model(tools)

        # Verify tool formatting
        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "test_tool"
        assert "description" in formatted[0]["function"]
        assert "parameters" in formatted[0]["function"]

        # Verify schema format
        assert "type" in formatted[0]["function"]["parameters"]
        assert formatted[0]["function"]["parameters"]["additionalProperties"] is False


def test_format_dict_tools():
    """Test formatting of dictionary-based tools for OpenAI models."""
    with patch.object(OpenAILLM, "auth"):
        # Create OpenAI LLM instance
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Define dictionary-based tool
        dict_tool = {
            "name": "dict_test_tool",
            "description": "A test tool defined as a dictionary",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter"},
                },
                "required": ["param1"],
            },
        }

        # Format the dictionary as a tool
        formatted = llm._format_tools_for_model([dict_tool])

        # Verify tool formatting
        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "dict_test_tool"
        assert (
            formatted[0]["function"]["description"]
            == "A test tool defined as a dictionary"
        )

        # Verify schema format
        assert "properties" in formatted[0]["function"]["parameters"]
        assert "param1" in formatted[0]["function"]["parameters"]["properties"]
        assert "param2" in formatted[0]["function"]["parameters"]["properties"]
        assert formatted[0]["function"]["parameters"]["additionalProperties"] is False
