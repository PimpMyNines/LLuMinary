"""
Debug script to help diagnose test issues
"""

import traceback
from unittest.mock import MagicMock, patch

from lluminary.models.providers.anthropic import AnthropicLLM


def debug_anthropic_initialization():
    """Test Anthropic provider initialization."""
    print("Starting debug of Anthropic initialization...")

    try:
        with patch("anthropic.Anthropic"), patch("requests.post") as mock_post:
            # Configure mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "content": [{"type": "text", "text": "test response"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            # Create the LLM instance
            print("Creating AnthropicLLM instance...")
            llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

            print("Testing AnthropicLLM attributes...")
            # Test attributes
            print(f"SUPPORTED_MODELS: {llm.SUPPORTED_MODELS}")
            print(f"THINKING_MODELS: {llm.THINKING_MODELS}")
            print(f"CONTEXT_WINDOW: {llm.CONTEXT_WINDOW}")
            print(f"COST_PER_MODEL: {llm.COST_PER_MODEL}")

            # Test expected model presence
            assert "claude-3-5-sonnet-20241022" in llm.SUPPORTED_MODELS
            assert "claude-3-7-sonnet-20250219" in llm.SUPPORTED_MODELS
            assert all(model in llm.CONTEXT_WINDOW for model in llm.SUPPORTED_MODELS)
            assert all(model in llm.COST_PER_MODEL for model in llm.SUPPORTED_MODELS)

            print("AnthropicLLM initialization test passed!")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


def debug_message_formatting():
    """Test Anthropic message formatting."""
    print("\nStarting debug of message formatting...")

    try:
        with patch("anthropic.Anthropic"), patch("requests.post") as mock_post:
            # Create the LLM instance
            llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

            # Test basic message
            messages = [{"message_type": "human", "message": "test message"}]
            print(f"Input message: {messages}")

            formatted = llm._format_messages_for_model(messages)
            print(f"Formatted message: {formatted}")

            assert formatted[0]["role"] == "user"
            assert isinstance(formatted[0]["content"], list)
            assert formatted[0]["content"][0]["type"] == "text"
            assert formatted[0]["content"][0]["text"] == "test message"

            print("Message formatting test passed!")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


def debug_convert_function_to_tool():
    """Test converting a function to a tool."""
    print("\nStarting debug of function to tool conversion...")

    try:
        with patch("anthropic.Anthropic"), patch("requests.post") as mock_post:
            # Create the LLM instance
            llm = AnthropicLLM("claude-3-5-sonnet-20241022", api_key="test-key")

            def test_func(param1: str, param2: int = 0) -> str:
                """Test function docstring"""
                return f"{param1} {param2}"

            print("Converting function to tool...")
            tools = llm._convert_functions_to_tools([test_func])

            print(f"Converted tool: {tools}")

            assert len(tools) > 0
            assert "type" in tools[0]
            assert tools[0]["type"] == "function"
            assert "function" in tools[0]
            assert "name" in tools[0]["function"]
            assert tools[0]["function"]["name"] == "test_func"

            print("Function to tool conversion test passed!")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    debug_anthropic_initialization()
    debug_message_formatting()
    debug_convert_function_to_tool()
