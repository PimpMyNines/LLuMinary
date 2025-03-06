"""
Debug script for testing Google LLM generation.
"""

from unittest.mock import MagicMock, patch

from lluminary.models.providers.google import GoogleLLM


def main():
    """Run the debug test."""
    try:
        print("Testing simple generation...")
        with patch("google.genai.GenerativeModel") as mock_generative_model:
            # Create the LLM instance
            print("Creating GoogleLLM...")
            llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

            # Create test message
            messages = [{"message_type": "human", "message": "Hello, assistant!"}]

            # Create mock model instance
            print("Creating mock model...")
            mock_model = MagicMock()
            mock_generative_model.return_value = mock_model

            # Create mock response
            print("Creating mock response...")
            mock_response = MagicMock()
            mock_response.text = "Hello, human!"
            mock_response.candidates = [MagicMock()]
            mock_response.usage_metadata.prompt_token_count = 5
            mock_response.usage_metadata.candidates_token_count = 3
            mock_response.usage_metadata.total_token_count = 8

            # Setup client and mock response
            print("Setting up client and mock response...")
            llm.client = MagicMock()
            mock_model.generate_content.return_value = mock_response

            # Mock the _format_messages_for_model method
            print("Patching _format_messages_for_model...")
            with patch.object(llm, "_format_messages_for_model") as mock_format:
                print("Setting up mock_format return value...")
                mock_format.return_value = [MagicMock()]

                # Call generate method
                print("Calling generate method...")
                response, usage, _ = llm.generate(
                    event_id="test_event",
                    system_prompt="You are a helpful assistant",
                    messages=messages,
                    max_tokens=100,
                )

                # Verify response
                print(f"Response: {response}")
                print(f"Usage: {usage}")

                # Debug output
                print(f"mock_format.called: {mock_format.called}")
                print(f"mock_generative_model.called: {mock_generative_model.called}")
                print(
                    f"mock_model.generate_content.called: {mock_model.generate_content.called}"
                )

                if mock_model.generate_content.called:
                    call_args = mock_model.generate_content.call_args
                    print(f"Call args: {call_args}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
