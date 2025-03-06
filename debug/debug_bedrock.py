"""
Debug Bedrock provider tests
"""

import traceback
from unittest.mock import MagicMock, patch

from lluminary.models.providers.bedrock import BedrockLLM


def debug_bedrock_initialization():
    """Debug the Bedrock initialization test"""
    print("\nDebugging Bedrock initialization...")

    try:
        # Mock the auth method
        with patch.object(BedrockLLM, "auth") as mock_auth:
            mock_auth.return_value = None

            # Create a BedrockLLM instance with correct model name
            llm = BedrockLLM(
                model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
                region_name="us-east-1",
            )

            # Check key attributes
            print(f"Model name: {llm.model_name}")
            print(f"SUPPORTED_MODELS length: {len(llm.SUPPORTED_MODELS)}")
            print(f"SUPPORTED_MODELS (first few): {llm.SUPPORTED_MODELS[:3]}")
            print(f"CONTEXT_WINDOW (sample): {list(llm.CONTEXT_WINDOW.keys())[:3]}")
            print(f"COST_PER_MODEL (sample): {list(llm.COST_PER_MODEL.keys())[:3]}")

            # Print all model attributes
            print("\nAll model attributes:")
            for attr_name in dir(llm):
                if not attr_name.startswith("_"):  # Skip private attributes
                    attr = getattr(llm, attr_name)
                    if not callable(attr):  # Skip methods
                        print(f"{attr_name}: {type(attr)}")

    except Exception as e:
        print(f"ERROR in initialization: {e!s}")
        traceback.print_exc()


def debug_message_formatting():
    """Debug the message formatting test"""
    print("\nDebugging message formatting...")

    try:
        # Create a BedrockLLM instance with mocked auth
        with patch.object(BedrockLLM, "auth") as mock_auth:
            mock_auth.return_value = None

            # Create a BedrockLLM instance with correct model name
            llm = BedrockLLM(
                model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
                region_name="us-east-1",
            )

            # Test basic message formatting
            messages = [{"message_type": "human", "message": "test message"}]

            try:
                print("Formatting message...")
                formatted = llm._format_messages_for_model(messages)
                print(f"Formatted message: {formatted}")
            except Exception as e:
                print(f"ERROR in message formatting: {e!s}")
                traceback.print_exc()

    except Exception as e:
        print(f"ERROR: {e!s}")
        traceback.print_exc()


def debug_raw_generation():
    """Debug the raw generation test"""
    print("\nDebugging raw generation...")

    try:
        # Create a BedrockLLM instance with mocked auth
        with patch.object(BedrockLLM, "auth") as mock_auth:
            mock_auth.return_value = None

            # Create a BedrockLLM instance
            llm = BedrockLLM(
                model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
                region_name="us-east-1",
            )

            # Create and configure mock
            llm.bedrock_client = MagicMock()
            if not hasattr(llm, "config"):
                llm.config = {}
            llm.config["client"] = llm.bedrock_client

            # Set up mock response
            claude_response = {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "This is a test response from Claude"}],
                    }
                },
                "usage": {"inputTokens": 15, "outputTokens": 10, "totalTokens": 25},
            }
            llm.bedrock_client.converse.return_value = claude_response

            # Test the raw generate method
            try:
                print("Calling _raw_generate...")
                response, usage = llm._raw_generate(
                    event_id="test_event",
                    system_prompt="You are a helpful assistant",
                    messages=[
                        {"message_type": "human", "message": "Tell me a short joke"}
                    ],
                    max_tokens=100,
                    temp=0.7,
                )
                print(f"Success! Response: {response}")
                print(f"Usage: {usage}")
            except Exception as e:
                print(f"ERROR in _raw_generate: {e!s}")
                traceback.print_exc()

            # Print client calls to see what was called
            print(f"Client calls: {llm.bedrock_client.mock_calls}")

    except Exception as e:
        print(f"ERROR: {e!s}")
        traceback.print_exc()


if __name__ == "__main__":
    debug_bedrock_initialization()
    debug_message_formatting()
    debug_raw_generation()
