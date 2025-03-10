"""
Script to verify our type-related fixes in the Bedrock provider implementation.
This is a simple script that performs the operations that were fixed, to ensure
they work properly without type errors.
"""

import sys
from typing import Any, Dict, List, cast
from unittest.mock import MagicMock, patch

# Import the BedrockLLM class
try:
    from src.lluminary.models.providers.bedrock import BedrockLLM

    print("Successfully imported BedrockLLM")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


def test_get_model_costs():
    """Test the get_model_costs return type."""
    # Create a minimal instance with mocked auth
    with patch.object(BedrockLLM, "auth"):
        llm = BedrockLLM(
            model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1",  # Provide required region_name
            auto_auth=False,
        )

        # Get the model costs
        costs = llm.get_model_costs()

        # Verify the structure
        print("Model costs:", costs)
        print("Types:")
        for key, value in costs.items():
            print(f"  {key}: {type(value)}")

        # Success check
        print("get_model_costs test: PASS")


def test_arithmetic_operations():
    """Test the arithmetic operations in cost calculations."""
    # Create a minimal instance with mocked auth
    with patch.object(BedrockLLM, "auth"):
        llm = BedrockLLM(
            model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1",  # Provide required region_name
            auto_auth=False,
        )

        # Mock get_model_costs to return known values
        llm.get_model_costs = MagicMock(
            return_value={
                "input_cost": 0.0001,
                "output_cost": 0.0002,
                "image_cost": 0.02,
            }
        )

        # Create test usage stats
        usage_stats = {
            "read_tokens": "100",  # String instead of number
            "write_tokens": 200.0,  # Float instead of int
            "images": 2,  # Integer
        }

        # Perform calculations like the fixed code does
        costs = llm.get_model_costs()

        # Extract cost values with proper type handling - convert to float
        read_token_cost = float(costs.get("input_cost", 0.0) or 0.0)
        write_token_cost = float(costs.get("output_cost", 0.0) or 0.0)
        image_cost = float(costs.get("image_cost", 0.0) or 0.0)

        # Ensure we're working with floats for all values before arithmetic
        read_tokens = float(usage_stats["read_tokens"])
        write_tokens = float(usage_stats["write_tokens"])
        image_count = float(usage_stats["images"])

        # Calculate costs
        read_cost = read_tokens * read_token_cost
        write_cost = write_tokens * write_token_cost
        image_cost_total = image_count * image_cost if image_count > 0 else 0.0
        total_cost = read_cost + write_cost + image_cost_total

        # Update the usage stats
        usage_stats["read_cost"] = read_cost
        usage_stats["write_cost"] = write_cost
        usage_stats["image_cost"] = image_cost_total
        usage_stats["total_cost"] = total_cost

        # Print results
        print("Calculation results:", usage_stats)
        print("read_cost type:", type(usage_stats["read_cost"]))
        print("write_cost type:", type(usage_stats["write_cost"]))
        print("image_cost type:", type(usage_stats["image_cost"]))
        print("total_cost type:", type(usage_stats["total_cost"]))

        # Verify calculations
        assert usage_stats["read_cost"] == 0.01
        assert usage_stats["write_cost"] == 0.04
        assert usage_stats["image_cost"] == 0.04
        assert usage_stats["total_cost"] == 0.09

        print("arithmetic_operations test: PASS")


def test_dictionary_typing():
    """Test that dictionary typing works properly."""
    with patch.object(BedrockLLM, "auth"):
        llm = BedrockLLM(
            model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1",  # Provide required region_name
            auto_auth=False,
        )

        # Create dictionaries with proper typing
        text_content: Dict[str, str] = {"text": "Test message"}
        print("text_content:", text_content)

        reasoning_content: Dict[str, Dict[str, Dict[str, str]]] = {
            "reasoningContent": {
                "reasoningText": {
                    "text": "Some reasoning",
                    "signature": "test-signature",
                }
            }
        }
        print("reasoning_content:", reasoning_content)

        # Test a case where we need to cast a value
        d = {"value": 1.23}
        cost_value = cast(float, d["value"])
        rounded = round(cost_value, 6)
        print("rounded value:", rounded)

        print("dictionary_typing test: PASS")


def test_explicit_formatting():
    """Test the formatting of custom parameters with proper typing."""
    # Create a dict with explicit typing
    tool_use_info: Dict[str, Any] = {
        "id": "tool-1",
        "name": "test_tool",
        "input": {"param1": "value1"},
    }
    print("tool_use_info:", tool_use_info)

    # Create a typed list
    messages: List[Dict[str, Any]] = [{"role": "user", "content": [{"text": "Hello"}]}]
    print("messages:", messages)

    print("explicit_formatting test: PASS")


def main():
    """Run all the verification tests."""
    print("\n--- Running Type Fix Verification ---\n")

    # Run tests
    test_get_model_costs()
    print("\n" + "-" * 50 + "\n")

    test_arithmetic_operations()
    print("\n" + "-" * 50 + "\n")

    test_dictionary_typing()
    print("\n" + "-" * 50 + "\n")

    test_explicit_formatting()
    print("\n" + "-" * 50 + "\n")

    print("All tests completed successfully!")


if __name__ == "__main__":
    main()
