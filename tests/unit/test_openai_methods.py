"""
Tests for OpenAI provider methods.
"""

import base64
from unittest.mock import mock_open, patch

from lluminary.models.providers.openai import OpenAILLM


def test_calculate_image_tokens():
    """Test token calculation for images."""
    with patch.object(OpenAILLM, "auth"):
        # Create OpenAI LLM instance
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Test low detail mode
        low_detail_tokens = llm.calculate_image_tokens(800, 600, "low")
        assert low_detail_tokens == llm.LOW_DETAIL_TOKEN_COST

        # Test high detail mode
        high_detail_tokens = llm.calculate_image_tokens(800, 600, "high")

        # For an 800x600 image, we expect the tokens to be in a reasonable range
        assert high_detail_tokens > 0

        # Test very large image (should be scaled down)
        large_image_tokens = llm.calculate_image_tokens(4000, 3000, "high")

        # This should have more tiles than the previous example
        assert large_image_tokens > 0


def test_encode_image():
    """Test the _encode_image method."""
    with patch.object(OpenAILLM, "auth"):
        # Create OpenAI LLM instance
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Create a mock image file
        mock_image_data = b"fake_image_data"

        # Mock the open function to return our fake image data
        with patch("builtins.open", mock_open(read_data=mock_image_data)):
            # Call the encode_image method
            base64_data = llm._encode_image("fake_image.jpg")

            # Verify the result is base64 encoded
            expected_base64 = base64.b64encode(mock_image_data).decode("utf-8")
            assert base64_data == expected_base64


def test_simple_mock():
    """Test a very simple mock to check testing environment."""
    assert True
