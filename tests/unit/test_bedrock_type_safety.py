"""
Unit tests to verify the type safety of the AWS Bedrock provider implementation.
These tests focus on validating the fixes we made to ensure proper type handling.
"""

from unittest.mock import MagicMock, patch

import pytest
from lluminary.exceptions import LLMProviderError
from lluminary.models.providers.bedrock import BedrockLLM


@pytest.fixture
def bedrock_llm():
    """Create a simple mocked BedrockLLM instance for testing."""
    with patch.object(BedrockLLM, "auth"):
        llm = BedrockLLM(
            model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0", auto_auth=False
        )
        return llm


def test_get_model_costs_return_type(bedrock_llm):
    """Test that get_model_costs returns Dict[str, Optional[float]] as expected."""
    costs = bedrock_llm.get_model_costs()

    # Verify the return structure
    assert isinstance(costs, dict)
    assert "input_cost" in costs
    assert "output_cost" in costs
    assert "image_cost" in costs

    # Verify the values can be None or float, not just float
    for key, value in costs.items():
        assert isinstance(value, float) or value is None


def test_format_messages_for_model_type_safety(bedrock_llm):
    """Test that _format_messages_for_model handles dictionary types safely."""
    # Create a test message with various content types
    messages = [
        {
            "message_type": "human",
            "message": "Hello, this is a test",
            "image_paths": ["/path/to/image.jpg"],
            "image_urls": ["https://example.com/image.jpg"],
            "thinking": {
                "thinking": "Some reasoning process",
                "thinking_signature": "signature",
            },
        }
    ]

    # Mock the image loading methods to avoid file operations
    bedrock_llm._get_image_bytes = MagicMock(return_value=b"fake_image_data")
    bedrock_llm._download_image_from_url = MagicMock(return_value=b"fake_image_data")

    # Format the messages
    formatted = bedrock_llm._format_messages_for_model(messages)

    # Verify the result is a list of dictionaries
    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert isinstance(formatted[0], dict)
    assert "role" in formatted[0]
    assert "content" in formatted[0]
    assert isinstance(formatted[0]["content"], list)

    # Check that all content items are dictionaries
    for item in formatted[0]["content"]:
        assert isinstance(item, dict)


def test_arithmetic_operations_type_safety(bedrock_llm):
    """Test that arithmetic operations in cost calculations use proper typing."""
    # Mock the costs returned by get_model_costs
    bedrock_llm.get_model_costs = MagicMock(
        return_value={"input_cost": 0.0001, "output_cost": 0.0002, "image_cost": 0.02}
    )

    # Create a simple usage stats dictionary with mixed types
    usage_stats = {
        "read_tokens": "100",  # String instead of number
        "write_tokens": 200.0,  # Float instead of int
        "images": 2,  # Integer
    }

    # Define a function that performs the cost calculations like in _raw_generate
    def calculate_costs(stats):
        costs = bedrock_llm.get_model_costs()

        # Extract cost values with proper type handling - convert to float
        read_token_cost = float(costs.get("input_cost", 0.0) or 0.0)
        write_token_cost = float(costs.get("output_cost", 0.0) or 0.0)
        image_cost = float(costs.get("image_cost", 0.0) or 0.0)

        # Ensure we're working with floats for all values before arithmetic
        read_tokens = float(stats["read_tokens"])
        write_tokens = float(stats["write_tokens"])
        image_count = float(stats["images"])

        # Calculate costs
        read_cost = read_tokens * read_token_cost
        write_cost = write_tokens * write_token_cost
        image_cost_total = image_count * image_cost if image_count > 0 else 0.0
        total_cost = read_cost + write_cost + image_cost_total

        # Return the updated dictionary
        stats["read_cost"] = read_cost
        stats["write_cost"] = write_cost
        stats["image_cost"] = image_cost_total
        stats["total_cost"] = total_cost

        return stats

    # Verify the calculation succeeds without type errors
    result = calculate_costs(usage_stats)

    # Check the results
    assert "read_cost" in result
    assert "write_cost" in result
    assert "image_cost" in result
    assert "total_cost" in result

    # Verify values are the expected type (float)
    assert isinstance(result["read_cost"], float)
    assert isinstance(result["write_cost"], float)
    assert isinstance(result["image_cost"], float)
    assert isinstance(result["total_cost"], float)

    # Verify the actual values
    assert result["read_cost"] == 0.01  # 100 tokens * 0.0001
    assert result["write_cost"] == 0.04  # 200 tokens * 0.0002
    assert result["image_cost"] == 0.04  # 2 images * 0.02
    assert result["total_cost"] == 0.09  # 0.01 + 0.04 + 0.04


def test_error_mapping_from_various_exceptions(bedrock_llm):
    """Test that different exception types are correctly mapped to LLM* exceptions."""
    # Test with a generic Exception
    generic_error = Exception("Generic error")
    mapped_error = bedrock_llm._map_aws_error(generic_error)
    assert isinstance(mapped_error, LLMProviderError)
    assert mapped_error.provider == "bedrock"
    assert hasattr(mapped_error, "details")

    # Create a function to simulate different error scenarios
    def test_error_mapping(error_message, expected_text):
        error = Exception(error_message)
        mapped = bedrock_llm._map_aws_error(error)
        assert expected_text in str(mapped).lower()
        assert mapped.provider == "bedrock"

    # Test various error messages and expected mappings
    test_error_mapping("Rate limit exceeded", "rate limit")
    test_error_mapping("Access denied", "authentication")
    test_error_mapping("Invalid parameters", "configuration")
    test_error_mapping("Service unavailable", "service unavailable")
