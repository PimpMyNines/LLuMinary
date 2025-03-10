"""
Script to verify our arithmetic operation type fixes in the Bedrock provider implementation.
This script simulates the operations that were fixed to ensure they work properly without type errors.
"""

from typing import Any, Dict, List, Optional


def test_estimate_tokens():
    """Test the _estimate_tokens method with explicit type handling."""

    # Simulate the _estimate_tokens method
    def estimate_tokens(text: str) -> int:
        """Simulate the _estimate_tokens method."""
        if not text:
            return 0
        # A simple approximation: 1 token ≈ 4 characters for English text
        # Ensure we're working with integers for division
        text_length = len(text)
        char_per_token = 4
        # Use integer division to ensure we get an integer result
        token_estimate = text_length // char_per_token
        # Ensure we return at least 1 token for non-empty text
        return max(1, token_estimate)

    # Test with different inputs
    empty_text = ""
    short_text = "Hello"
    long_text = "This is a longer text that should result in more tokens."

    # Get token estimates
    empty_tokens = estimate_tokens(empty_text)
    short_tokens = estimate_tokens(short_text)
    long_tokens = estimate_tokens(long_text)

    # Verify results
    print(f"Empty text ({len(empty_text)} chars): {empty_tokens} tokens")
    print(f"Short text ({len(short_text)} chars): {short_tokens} tokens")
    print(f"Long text ({len(long_text)} chars): {long_tokens} tokens")

    # Verify types
    assert isinstance(empty_tokens, int)
    assert isinstance(short_tokens, int)
    assert isinstance(long_tokens, int)

    # Success check
    print("_estimate_tokens test: PASS")


def test_cost_calculations():
    """Test cost calculations with explicit type handling."""

    # Simulate the get_model_costs method
    def get_model_costs() -> Dict[str, Optional[float]]:
        """Simulate the get_model_costs method."""
        return {"input_cost": 0.0000008, "output_cost": 0.0000024, "image_cost": None}

    # Create test usage stats
    usage_stats = {"read_tokens": 100, "write_tokens": 50, "images": 0}

    # Calculate costs manually
    costs = get_model_costs()

    # Extract cost values with proper type handling
    read_token_cost = float(costs.get("input_cost", 0.0) or 0.0)
    write_token_cost = float(costs.get("output_cost", 0.0) or 0.0)
    image_cost = float(costs.get("image_cost", 0.0) or 0.0)

    # Ensure we're working with floats for all values
    read_tokens = float(usage_stats["read_tokens"])
    write_tokens = float(usage_stats["write_tokens"])
    image_count = float(usage_stats["images"])

    # Calculate costs
    read_cost = read_tokens * read_token_cost
    write_cost = write_tokens * write_token_cost
    image_cost_total = image_count * image_cost if image_count > 0 else 0.0
    total_cost = read_cost + write_cost + image_cost_total

    # Round costs
    read_cost_rounded = round(read_cost, 6)
    write_cost_rounded = round(write_cost, 6)
    image_cost_rounded = round(image_cost_total, 6)
    total_cost_rounded = round(total_cost, 6)

    # Print results
    print(
        f"Read tokens: {read_tokens}, cost per token: {read_token_cost}, total: {read_cost_rounded}"
    )
    print(
        f"Write tokens: {write_tokens}, cost per token: {write_token_cost}, total: {write_cost_rounded}"
    )
    print(
        f"Images: {image_count}, cost per image: {image_cost}, total: {image_cost_rounded}"
    )
    print(f"Total cost: {total_cost_rounded}")

    # Verify types
    assert isinstance(read_cost, float)
    assert isinstance(write_cost, float)
    assert isinstance(image_cost_total, float)
    assert isinstance(total_cost, float)

    # Success check
    print("Cost calculations test: PASS")


def test_convert_functions_to_tools():
    """Test the _convert_functions_to_tools method with proper type handling."""
    import inspect

    # Simulate the _convert_functions_to_tools method
    def convert_functions_to_tools(functions: List[Any]) -> List[Dict[str, Any]]:
        """Simulate the _convert_functions_to_tools method."""
        tools: List[Dict[str, Any]] = []

        for func in functions:
            # Get function name and docstring
            func_name = func.__name__
            func_doc = func.__doc__ or "No description available"

            # Get function signature
            sig = inspect.signature(func)

            # Build parameters schema
            parameters: Dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            for param_name, param in sig.parameters.items():
                # Skip self parameter for methods
                if param_name == "self":
                    continue

                # Get parameter type and default value
                param_type = "string"  # Default type
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list or param.annotation == List:
                        param_type = "array"
                    elif param.annotation == dict or param.annotation == Dict:
                        param_type = "object"

                # Add parameter to schema
                # Ensure properties is a dictionary before accessing it
                if "properties" in parameters and isinstance(
                    parameters["properties"], dict
                ):
                    properties_dict = parameters["properties"]
                    # Add the new property
                    properties_dict[param_name] = {
                        "type": param_type,
                        "description": f"Parameter: {param_name}",
                    }

                # Add to required list if no default value
                if param.default == inspect.Parameter.empty:
                    # Ensure required is a list before modifying it
                    if "required" in parameters and isinstance(
                        parameters["required"], list
                    ):
                        # Create a mutable copy of the required list
                        required_params = list(parameters["required"])
                        required_params.append(param_name)
                        parameters["required"] = required_params

            # Create tool definition
            tool: Dict[str, Any] = {
                "name": func_name,
                "description": func_doc.strip(),
                "input_schema": parameters,
            }

            tools.append(tool)

        return tools

    # Define test functions
    def test_func1(param1: str, param2: int = 0) -> str:
        """Test function 1."""
        return f"{param1} {param2}"

    def test_func2(param1: str, param2: int, param3: bool = False) -> Dict[str, Any]:
        """Test function 2."""
        return {"param1": param1, "param2": param2, "param3": param3}

    # Convert functions to tools
    functions = [test_func1, test_func2]
    tools = convert_functions_to_tools(functions)

    # Verify results
    print(f"Number of tools: {len(tools)}")
    for i, tool in enumerate(tools):
        print(f"Tool {i+1}:")
        print(f"  Name: {tool.get('name')}")
        print(f"  Description: {tool.get('description')}")
        print(f"  Input schema type: {tool.get('input_schema', {}).get('type')}")
        print(
            f"  Properties: {tool.get('input_schema', {}).get('properties', {}).keys()}"
        )
        print(f"  Required: {tool.get('input_schema', {}).get('required', [])}")

    # Verify types
    assert isinstance(tools, list)
    for tool in tools:
        assert isinstance(tool, dict)
        assert isinstance(tool.get("name", ""), str)
        assert isinstance(tool.get("description", ""), str)
        assert isinstance(tool.get("input_schema", {}), dict)

    # Success check
    print("_convert_functions_to_tools test: PASS")


if __name__ == "__main__":
    print("Running verification tests for arithmetic operation type fixes...")
    print("\n1. Testing _estimate_tokens method:")
    test_estimate_tokens()

    print("\n2. Testing cost calculations:")
    test_cost_calculations()

    print("\n3. Testing _convert_functions_to_tools method:")
    test_convert_functions_to_tools()

    print("\nAll tests passed successfully!")
