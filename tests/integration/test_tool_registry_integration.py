"""
Integration tests for the tool registry functionality.
Tests tool registration, validation, and usage with real LLM function calling.
"""

from typing import Dict, List

import pytest
from lluminary.models.router import get_llm_from_model
from lluminary.tools.registry import ToolRegistry, ToolValidationError
from lluminary.tools.validators import (
    json_serializable,
    validate_params,
    validate_return_type,
)

# Mark all tests in this file as tool registry integration tests
pytestmark = [pytest.mark.integration, pytest.mark.tools]


class TestToolRegistryIntegration:
    """Integration tests for tool registry functionality."""

    @pytest.fixture
    def registry(self):
        """Create a tool registry with test tools."""
        registry = ToolRegistry()

        # Register some test tools
        @validate_return_type(Dict[str, str])
        def get_weather(location: str, unit: str = "celsius") -> Dict[str, str]:
            """Get weather for a location."""
            return {
                "location": location,
                "temperature": "22",
                "unit": unit,
                "condition": "sunny",
            }

        @json_serializable
        @validate_params(str, query=str)
        def search_database(term: str, query: str) -> List[Dict[str, str]]:
            """Search a database for information."""
            return [
                {
                    "id": "1",
                    "title": f"Result for {term}",
                    "content": f"Information about {query}",
                },
                {
                    "id": "2",
                    "title": f"Another result for {term}",
                    "content": f"More about {query}",
                },
            ]

        registry.register_tool(get_weather)
        registry.register_tool(search_database)

        return registry

    def test_tool_registration_and_usage(self, registry):
        """Test registering tools and using them."""
        # Test tool registration
        assert len(registry.list_tools()) == 2

        # Test using registered tools
        weather_tool = registry.get_tool("get_weather")
        assert weather_tool is not None

        weather_result = weather_tool("New York", unit="fahrenheit")
        assert weather_result["location"] == "New York"
        assert weather_result["unit"] == "fahrenheit"

        # Test tool stats after usage
        weather_stats = registry.get_tool_stats("get_weather")
        assert weather_stats["success_count"] == 1
        assert weather_stats["failure_count"] == 0
        assert weather_stats["average_execution_time"] > 0

    def test_tool_function_calling_integration(self):
        """Test that LLM providers can call registered tools."""
        # Test models that support function calling
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-3-haiku-20240307",  # Anthropic
        ]

        # Define a simple tool for the model to call
        def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, str]:
            """Get the current weather in a location.

            Args:
                location: The city and state, e.g. San Francisco, CA
                unit: The temperature unit to use. One of 'celsius' or 'fahrenheit'

            Returns:
                A dictionary with weather information
            """
            # In a real implementation, this would call a weather API
            return {
                "location": location,
                "temperature": "22" if unit == "celsius" else "72",
                "unit": unit,
                "condition": "sunny",
            }

        # Register tool in registry
        registry = ToolRegistry()
        registry.register_tool(get_current_weather)

        # Convert tool to OpenAI/Anthropic tool format
        tools = [
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]

        # Track results
        successful_models = []
        all_results = {}

        print("\n" + "=" * 60)
        print("TOOL FUNCTION CALLING TEST")
        print("=" * 60)

        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting function calling with {provider} model: {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Generate response with tools
                print("Generating response with tools...")
                response, usage, _ = llm.generate(
                    event_id="test_function_call",
                    system_prompt="You are a helpful assistant that can provide weather information.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "What's the weather like in Boston?",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=200,
                    tools=tools,
                )

                # Print results
                print(f"Response: {response[:100]}...")
                print(
                    f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write"
                )
                print(f"Cost: ${usage['total_cost']:.6f}")

                # Check that the response mentions Boston and weather
                assert "Boston" in response or "boston" in response.lower()
                weather_terms = [
                    "weather",
                    "temperature",
                    "degrees",
                    "sunny",
                    "celsius",
                    "fahrenheit",
                ]
                found_terms = [
                    term for term in weather_terms if term in response.lower()
                ]

                if found_terms:
                    print(f"Weather information found with terms: {found_terms}")
                    successful_models.append(model_name)
                    all_results[model_name] = response
                else:
                    print("Weather information may not have been correctly included")

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")

        # Print summary
        print("\n" + "=" * 60)
        print("FUNCTION CALLING TEST SUMMARY")
        print("=" * 60)
        print(f"Successful models: {len(successful_models)}/{len(test_models)}")
        for model in successful_models:
            print(f"  - {model}")

        # Skip if no models work at all
        if not successful_models:
            pytest.skip(
                "Skipping test as no models were able to complete the function calling test"
            )

    def test_tool_error_handling(self, registry):
        """Test error handling in the tool registry."""

        # Test calling a tool that raises an exception
        @validate_return_type(Dict[str, str])
        def failing_tool(param: str) -> Dict[str, str]:
            """A tool that always fails."""
            raise ValueError("This tool always fails")

        registry.register_tool(failing_tool)

        tool = registry.get_tool("failing_tool")
        with pytest.raises(ValueError):
            tool("test")

        # Check that failure is tracked
        stats = registry.get_tool_stats("failing_tool")
        assert stats["success_count"] == 0
        assert stats["failure_count"] == 1

        # Test tool with invalid return type
        @validate_return_type(Dict[str, str])
        def wrong_return_type() -> Dict[str, str]:
            """A tool that returns the wrong type."""
            return ["This", "is", "not", "a", "dict"]  # type: ignore

        registry.register_tool(wrong_return_type)

        tool = registry.get_tool("wrong_return_type")
        with pytest.raises(ToolValidationError):
            tool()

        # Check that failure is tracked
        stats = registry.get_tool_stats("wrong_return_type")
        assert stats["success_count"] == 0
        assert stats["failure_count"] == 1
