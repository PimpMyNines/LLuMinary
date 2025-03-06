"""
Unit tests for Anthropic function calling functionality.

This module tests the function calling/tool use capabilities of the Anthropic provider.
"""

import inspect
import json
from unittest.mock import MagicMock, patch

import pytest

from lluminary.exceptions import FormatError
from lluminary.models.providers.anthropic import AnthropicLLM


@pytest.fixture
def anthropic_llm():
    """Fixture for Anthropic LLM instance."""
    with patch("anthropic.Anthropic"):
        # Create the LLM instance with mock API key
        llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

        # Ensure config exists
        llm.config = {"api_key": "test-key"}

        yield llm


def test_function_to_tool_conversion(anthropic_llm):
    """Test conversion of Python function to Anthropic tool format."""

    def test_function(param1: str, param2: int = 42) -> dict:
        """Test function with docstring."""
        return {"result": f"{param1}_{param2}"}

    # Convert function to tool
    tool = anthropic_llm._convert_function_to_tool(test_function)

    # Verify tool structure
    assert tool["type"] == "function"
    assert "function" in tool

    # Verify function properties
    assert tool["function"]["name"] == "test_function"
    assert "description" in tool["function"]
    assert "Test function with docstring" in tool["function"]["description"]

    # Verify parameters
    assert "parameters" in tool["function"]
    assert tool["function"]["parameters"]["type"] == "object"
    assert "properties" in tool["function"]["parameters"]

    # Verify parameter details
    properties = tool["function"]["parameters"]["properties"]
    assert "param1" in properties
    assert "param2" in properties
    assert properties["param1"]["type"] == "string"
    assert properties["param2"]["type"] == "integer"
    assert properties["param2"]["default"] == 42

    # Verify required parameters
    assert "required" in tool["function"]["parameters"]
    assert "param1" in tool["function"]["parameters"]["required"]
    assert "param2" not in tool["function"]["parameters"]["required"]


def test_functions_to_tools_conversion(anthropic_llm):
    """Test conversion of multiple functions to Anthropic tools."""

    def func1(x: int) -> int:
        """Function 1."""
        return x * 2

    def func2(text: str, uppercase: bool = False) -> str:
        """Function 2."""
        return text.upper() if uppercase else text.lower()

    # Convert functions to tools
    tools = anthropic_llm._convert_functions_to_tools([func1, func2])

    # Verify tools list
    assert len(tools) == 2
    assert tools[0]["function"]["name"] == "func1"
    assert tools[1]["function"]["name"] == "func2"

    # Verify tools have correct structure
    for tool in tools:
        assert tool["type"] == "function"
        assert "function" in tool
        assert "parameters" in tool["function"]


def test_generate_with_functions(anthropic_llm):
    """Test generating with available functions."""

    def get_weather(location: str, unit: str = "celsius") -> dict:
        """Get the current weather for a location."""
        return {"location": location, "temperature": 22, "unit": unit}

    def calculate(expression: str) -> float:
        """Calculate a mathematical expression."""
        return eval(expression)

    # Mock API response with tool use
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": "I'll check the weather for you."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {"location": "San Francisco", "unit": "fahrenheit"},
                },
            ],
            "usage": {"input_tokens": 15, "output_tokens": 10},
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Call generate with functions
        response, usage, tool_calls = anthropic_llm.generate(
            event_id="test-functions",
            system_prompt="You are a helpful assistant with access to weather data.",
            messages=[
                {
                    "message_type": "human",
                    "message": "What's the weather in San Francisco?",
                }
            ],
            functions=[get_weather, calculate],
        )

        # Verify response
        assert "check the weather" in response

        # Verify tool calls
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["arguments"]["location"] == "San Francisco"
        assert tool_calls[0]["arguments"]["unit"] == "fahrenheit"

        # Verify API request contained tools
        call_args = mock_post.call_args[1]
        request_data = json.loads(call_args["data"])
        assert "tools" in request_data
        assert len(request_data["tools"]) == 2

        # Verify tool definitions
        tools = request_data["tools"]
        weather_tool = next(t for t in tools if t["function"]["name"] == "get_weather")
        assert (
            weather_tool["function"]["description"]
            == "Get the current weather for a location."
        )
        assert "location" in weather_tool["function"]["parameters"]["properties"]
        assert "unit" in weather_tool["function"]["parameters"]["properties"]


def test_tool_use_response_formatting(anthropic_llm):
    """Test formatting of tool use responses from the model."""
    # Define a test message with tool use
    messages = [
        {"message_type": "human", "message": "What's 5 + 10?"},
        {
            "message_type": "ai",
            "message": "I'll calculate that for you.",
            "tool_use": {
                "id": "tool_123",
                "name": "calculate",
                "input": {"expression": "5 + 10"},
            },
        },
        {
            "message_type": "tool",
            "tool_result": {"tool_id": "tool_123", "success": True, "result": 15},
        },
    ]

    # Format messages
    formatted = anthropic_llm._format_messages_for_model(messages)

    # Verify structure (should have 3 messages)
    assert len(formatted) == 3

    # Verify human message
    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"][0]["type"] == "text"
    assert formatted[0]["content"][0]["text"] == "What's 5 + 10?"

    # Verify assistant message with tool use
    assert formatted[1]["role"] == "assistant"
    assert len(formatted[1]["content"]) == 2  # Text + tool_use

    # Find text and tool_use parts
    text_part = next(p for p in formatted[1]["content"] if p["type"] == "text")
    tool_part = next(p for p in formatted[1]["content"] if p["type"] == "tool_use")

    assert text_part["text"] == "I'll calculate that for you."
    assert tool_part["id"] == "tool_123"
    assert tool_part["name"] == "calculate"
    assert tool_part["input"] == {"expression": "5 + 10"}

    # Verify tool result message
    assert formatted[2]["role"] == "user"
    assert formatted[2]["content"][0]["type"] == "tool_result"
    assert formatted[2]["content"][0]["tool_use_id"] == "tool_123"
    assert "Tool Call Successful: 15" in formatted[2]["content"][0]["content"]


def test_tool_failure_formatting(anthropic_llm):
    """Test formatting of tool failure responses."""
    # Define a test message with tool failure
    messages = [
        {"message_type": "human", "message": "What's the square root of -1?"},
        {
            "message_type": "ai",
            "message": "I'll calculate that for you.",
            "tool_use": {
                "id": "tool_456",
                "name": "calculate",
                "input": {"expression": "math.sqrt(-1)"},
            },
        },
        {
            "message_type": "tool",
            "tool_result": {
                "tool_id": "tool_456",
                "success": False,
                "error": "Cannot calculate square root of negative number",
            },
        },
    ]

    # Format messages
    formatted = anthropic_llm._format_messages_for_model(messages)

    # Verify tool result message
    tool_result_message = formatted[2]
    assert tool_result_message["role"] == "user"
    tool_result_content = tool_result_message["content"][0]
    assert tool_result_content["type"] == "tool_result"
    assert tool_result_content["tool_use_id"] == "tool_456"
    assert "Tool Call Failed:" in tool_result_content["content"]
    assert (
        "Cannot calculate square root of negative number"
        in tool_result_content["content"]
    )


def test_invalid_tool_result_format(anthropic_llm):
    """Test error handling for invalid tool result format."""
    # Define a message with invalid tool_result (missing tool_id)
    messages = [
        {
            "message_type": "tool",
            "tool_result": {
                # Missing tool_id field
                "success": True,
                "result": "some result",
            },
        }
    ]

    # Try to format and expect error
    with pytest.raises(FormatError) as excinfo:
        anthropic_llm._format_messages_for_model(messages)

    # Verify error details
    assert "Invalid 'tool_result' structure" in str(excinfo.value)
    assert "tool_id" in str(excinfo.value)
    assert excinfo.value.provider == "anthropic"


def test_invalid_tool_use_format(anthropic_llm):
    """Test error handling for invalid tool use format."""
    # Define a message with invalid tool_use (missing name)
    messages = [
        {
            "message_type": "ai",
            "message": "I'll help with that.",
            "tool_use": {
                "id": "tool_123",
                # Missing name field
                "input": {"param": "value"},
            },
        }
    ]

    # Try to format and expect error
    with pytest.raises(FormatError) as excinfo:
        anthropic_llm._format_messages_for_model(messages)

    # Verify error details
    assert "Invalid 'tool_use' structure" in str(excinfo.value)
    assert "name" in str(excinfo.value)
    assert excinfo.value.provider == "anthropic"


def test_complex_function_conversion(anthropic_llm):
    """Test conversion of complex function with nested types."""

    def complex_function(
        query: str,
        filters: dict,
        limit: int = 10,
        options: list = None,
        enable_feature: bool = False,
    ) -> dict:
        """Search for items matching the query and filters.

        Args:
            query: The search query string
            filters: Dictionary of filter criteria
            limit: Maximum number of results
            options: Additional options for the search
            enable_feature: Whether to enable a special feature

        Returns:
            Dictionary containing search results
        """
        return {"results": [], "count": 0}

    # Get function signature and docstring
    signature = inspect.signature(complex_function)
    docstring = inspect.getdoc(complex_function)

    # Convert function to tool
    tool = anthropic_llm._convert_function_to_tool(complex_function)

    # Verify tool structure
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "complex_function"
    assert (
        "Search for items matching the query and filters"
        in tool["function"]["description"]
    )

    # Verify parameters
    params = tool["function"]["parameters"]["properties"]
    assert "query" in params
    assert params["query"]["type"] == "string"
    assert "filters" in params
    assert params["filters"]["type"] == "object"
    assert "limit" in params
    assert params["limit"]["type"] == "integer"
    assert params["limit"]["default"] == 10
    assert "options" in params
    assert params["options"]["type"] == "array"
    assert "enable_feature" in params
    assert params["enable_feature"]["type"] == "boolean"
    assert params["enable_feature"]["default"] is False

    # Verify required parameters
    required = tool["function"]["parameters"]["required"]
    assert "query" in required
    assert "filters" in required
    assert "limit" not in required
    assert "options" not in required
    assert "enable_feature" not in required


def test_duplicate_function_detection(anthropic_llm):
    """Test handling of functions with duplicate names."""

    # Define two functions with the same name but different signatures
    def test_func(a: int) -> int:
        """First function."""
        return a * 2

    # Using a lambda to create a function with the same name
    duplicate_func = lambda x: x + 1
    duplicate_func.__name__ = "test_func"
    duplicate_func.__doc__ = "Second function with same name."

    # Convert functions to tools
    tools = anthropic_llm._convert_functions_to_tools([test_func, duplicate_func])

    # Since we can't have duplicate function names, the second one should be renamed
    # or otherwise differentiated
    function_names = [tool["function"]["name"] for tool in tools]
    assert len(set(function_names)) == len(function_names)  # No duplicates


def test_tool_use_in_response_parsing(anthropic_llm):
    """Test parsing of tool use in model responses."""
    # Mock API response with tool use
    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [
                {"type": "text", "text": "Let me calculate that for you."},
                {
                    "type": "tool_use",
                    "id": "tool_789",
                    "name": "calculate",
                    "input": {"expression": "123 * 456"},
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Define a test function
        def calculate(expression: str) -> float:
            """Calculate a mathematical expression."""
            return eval(expression)

        # Call generate
        response, usage, tool_calls = anthropic_llm.generate(
            event_id="test-tool-response",
            system_prompt="You are a helpful assistant.",
            messages=[{"message_type": "human", "message": "What's 123 × 456?"}],
            functions=[calculate],
        )

        # Verify the tool call was extracted correctly
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "calculate"
        assert tool_calls[0]["arguments"]["expression"] == "123 * 456"
        assert tool_calls[0]["id"] == "tool_789"


def test_tool_call_execution_flow(anthropic_llm):
    """Test the full flow of tool calling and response handling."""

    # Define a test function
    def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
        """Convert a value from one unit to another."""
        # Just a simple example - not actual conversion
        if from_unit == "celsius" and to_unit == "fahrenheit":
            result = value * 9 / 5 + 32
        elif from_unit == "fahrenheit" and to_unit == "celsius":
            result = (value - 32) * 5 / 9
        else:
            raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")

        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
        }

    # Test the full conversation flow with tool calling
    with patch("requests.post") as mock_post:
        # First response - model calls the tool
        first_response = MagicMock()
        first_response.json.return_value = {
            "content": [
                {"type": "text", "text": "I'll convert that for you."},
                {
                    "type": "tool_use",
                    "id": "tool_temp",
                    "name": "convert_units",
                    "input": {
                        "value": 25,
                        "from_unit": "celsius",
                        "to_unit": "fahrenheit",
                    },
                },
            ],
            "usage": {"input_tokens": 15, "output_tokens": 10},
        }

        # Second response - model responds to tool result
        second_response = MagicMock()
        second_response.json.return_value = {
            "content": [{"type": "text", "text": "25°C is equal to 77°F."}],
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }

        # Set up the sequence of responses
        mock_post.side_effect = [first_response, second_response]

        # First generate call - model should call the tool
        response1, usage1, tool_calls = anthropic_llm.generate(
            event_id="test-tool-flow-1",
            system_prompt="You are a helpful assistant.",
            messages=[
                {
                    "message_type": "human",
                    "message": "Convert 25 degrees Celsius to Fahrenheit.",
                }
            ],
            functions=[convert_units],
        )

        # Verify tool call
        assert "I'll convert that for you" in response1
        assert tool_calls is not None
        assert tool_calls[0]["name"] == "convert_units"
        assert tool_calls[0]["arguments"]["value"] == 25

        # Execute the tool (in a real scenario, this would be done by the application)
        tool_result = convert_units(**tool_calls[0]["arguments"])

        # Second generate call - send tool result back to model
        response2, usage2, _ = anthropic_llm.generate(
            event_id="test-tool-flow-2",
            system_prompt="You are a helpful assistant.",
            messages=[
                {
                    "message_type": "human",
                    "message": "Convert 25 degrees Celsius to Fahrenheit.",
                },
                {
                    "message_type": "ai",
                    "message": "I'll convert that for you.",
                    "tool_use": {
                        "id": "tool_temp",
                        "name": "convert_units",
                        "input": {
                            "value": 25,
                            "from_unit": "celsius",
                            "to_unit": "fahrenheit",
                        },
                    },
                },
                {
                    "message_type": "tool",
                    "tool_result": {
                        "tool_id": "tool_temp",
                        "success": True,
                        "result": tool_result,
                    },
                },
            ],
            functions=[convert_units],
        )

        # Verify final response
        assert "25°C is equal to 77°F" in response2
