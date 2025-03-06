"""
Debug script for testing Google LLM message formatting.
"""

from unittest.mock import MagicMock, patch

from lluminary.models.providers.google import GoogleLLM


def main():
    """Run the debug test."""
    try:
        print("Creating GoogleLLM instance...")
        with patch.object(GoogleLLM, "auth"):
            llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")
            print(f"LLM created: {llm}, model_name={llm.model_name}")

            # Create a simple message to test formatting
            message = {"message_type": "human", "message": "Hello, assistant!"}
            print(f"Message: {message}")

            # Mock the _create_content method
            print("Patching _create_content method...")
            with patch.object(llm, "_create_content") as mock_create_content:
                print("Creating mock content...")
                mock_content = MagicMock()
                mock_content.role = "user"
                mock_content.parts = [MagicMock(text="Hello, assistant!")]
                print(f"Mock content: {mock_content}, role={mock_content.role}")
                mock_create_content.return_value = mock_content

                # Format the message
                print("Formatting message...")
                formatted = llm._format_messages_for_model([message])
                print(f"Formatted result: {formatted}")

                # Check results
                print("Verifying results...")
                print(f"mock_create_content.called: {mock_create_content.called}")
                print(f"len(formatted): {len(formatted)}")
                print(f"formatted[0].role: {formatted[0].role}")
                print(f"len(formatted[0].parts): {len(formatted[0].parts)}")
                print(f"formatted[0].parts[0].text: {formatted[0].parts[0].text}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
