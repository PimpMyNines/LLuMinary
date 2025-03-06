"""
Debug script for testing Google LLM tools support.
"""

import inspect
from unittest.mock import MagicMock, patch

from lluminary.models.providers.google import GoogleLLM


def main():
    """Run the debug test."""
    try:
        print("Testing tools support...")

        # Define simple test function
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Weather in {location}: Sunny"

        # Create the LLM instance with mocked auth
        with patch.object(GoogleLLM, "auth"):
            llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

            # Setup client and mock response
            mock_response = MagicMock()
            mock_response.text = "I'll check the weather"
            mock_response.candidates = [MagicMock()]
            mock_response.usage_metadata.prompt_token_count = 10
            mock_response.usage_metadata.candidates_token_count = 5
            mock_response.usage_metadata.total_token_count = 15

            # Mock client
            llm.client = MagicMock()
            models_mock = MagicMock()
            llm.client.models = models_mock
            models_mock.generate_content.return_value = mock_response

            # Mock types for config
            with patch("google.genai.types") as mock_types, patch.object(
                llm, "_format_messages_for_model"
            ):

                # Create mock config object
                mock_config = MagicMock()
                mock_types.GenerateContentConfig.return_value = mock_config

                # Print tool details for debugging
                print(f"Tool function: {get_weather}")
                print(f"Tool signature: {inspect.signature(get_weather)}")
                print(f"Tool docstring: {get_weather.__doc__}")

                # Call generate with tools
                print("Calling generate with tools...")
                response, usage, _ = llm.generate(
                    event_id="test_event",
                    system_prompt="You are a helpful assistant",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "What's the weather in New York?",
                        }
                    ],
                    tools=[get_weather],
                )

                # Check if config was created with tools
                print("Checking GenerateContentConfig calls")
                print(
                    f"GenerateContentConfig called: {mock_types.GenerateContentConfig.called}"
                )
                print(f"Call args: {mock_types.GenerateContentConfig.call_args}")

                if mock_types.GenerateContentConfig.call_args:
                    config_args = mock_types.GenerateContentConfig.call_args[1]
                    print(f"Config args: {config_args}")
                    print(f"Tools in config_args: {'tools' in config_args}")
                    if "tools" in config_args:
                        print(f"Tools value: {config_args['tools']}")

                # Verify model was called
                print(f"generate_content called: {models_mock.generate_content.called}")

                # Verify response
                print(f"Response: {response}")
                print(f"Usage: {usage}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
