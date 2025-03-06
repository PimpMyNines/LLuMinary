"""
Debug script for Google LLM provider.
"""

from unittest.mock import patch

from lluminary.models.providers.google import GoogleLLM


def main():
    """Main debug function."""
    try:
        print("Testing GoogleLLM class attributes...")
        print(f"SUPPORTED_MODELS: {GoogleLLM.SUPPORTED_MODELS}")
        print(f"SUPPORTS_IMAGES: {GoogleLLM.SUPPORTS_IMAGES}")
        print(f"CONTEXT_WINDOW: {GoogleLLM.CONTEXT_WINDOW}")
        print(f"COST_PER_MODEL: {GoogleLLM.COST_PER_MODEL}")

        print("\nCreating GoogleLLM instance...")
        with patch.object(GoogleLLM, "auth"):
            llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")
            print(f"LLM created. model_name={llm.model_name}")
            # Check if api_key is available in kwargs:
            for key, value in llm.__dict__.items():
                print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
