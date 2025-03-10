"""
Simple test of a single provider to identify error handling issues.
"""

import os

# Map environment variables for testing
TOKEN_TO_KEY_MAPPING = {
    "OPENAI_API_TOKEN": "OPENAI_API_KEY",
    "ANTHROPIC_API_TOKEN": "ANTHROPIC_API_KEY",
    "COHERE_API_TOKEN": "COHERE_API_KEY",
    "GOOGLE_GEMINI_API_TOKEN": "GOOGLE_API_KEY",  # Add mapping for Google Gemini
}

# Map environment variables
for token_var, key_var in TOKEN_TO_KEY_MAPPING.items():
    if token_var in os.environ:
        os.environ[key_var] = os.environ[token_var]
        print(f"✓ Mapped {token_var} to {key_var}")

# Set the model to test - CHANGE THIS TO TEST DIFFERENT PROVIDERS
TEST_MODEL = "gemini-2.0-flash"
TEST_PROVIDER = "google"

print(f"\nTesting {TEST_PROVIDER} model {TEST_MODEL}...")

try:
    # Import specific provider class directly instead of via router
    if TEST_PROVIDER == "anthropic":
        from lluminary.models.providers.anthropic import AnthropicLLM

        # Create instance directly
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        # Create provider instance directly
        llm = AnthropicLLM(
            model_name="claude-3-5-haiku-20241022",  # Provider-specific model name format
            api_key=api_key,
        )

        # Test generation
        print("Generating text...")
        result = llm.generate(
            event_id="test-anthropic-direct",
            system_prompt="You are a helpful assistant.",
            messages=[
                {
                    "message_type": "human",
                    "message": "What is the capital of France? Keep your answer brief.",
                }
            ],
            max_tokens=50,
        )

        # Unpack results
        response, usage, messages = result

        # Print result
        print(f"\nResponse: {response}")
        print(f"Usage: {usage}")

    elif TEST_PROVIDER == "google":
        # Manually add src to path if needed
        import sys

        if "/Users/shawnlopresto/Projects/lluminary/src" not in sys.path:
            sys.path.insert(0, "/Users/shawnlopresto/Projects/lluminary/src")

        # Create a simple monkeypatch for the get_secret function
        # Monkeypatch the imports to avoid get_secret dependency
        import sys
        from unittest.mock import MagicMock

        sys.modules["lluminary.utils"] = MagicMock()
        sys.modules["lluminary.utils.get_secret"] = MagicMock()

        # Create the mock function
        def mock_get_secret(secret_id, required_keys=None):
            if "google" in secret_id.lower():
                return {"api_key": os.environ.get("GOOGLE_API_KEY")}
            return {}

        # Import Google provider with patched dependencies
        from lluminary.models.providers.google import GoogleLLM

        # Apply our mock get_secret to the module
        sys.modules["lluminary.utils"].get_secret = mock_get_secret

        # Create instance directly
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set (should be mapped from GOOGLE_GEMINI_API_TOKEN)"
            )

        print(
            f"Using API key: {api_key[:5]}...{api_key[-5:] if len(api_key) > 10 else ''}"
        )

        # Create provider instance directly
        llm = GoogleLLM(
            model_name=TEST_MODEL,
            api_key=api_key,
        )

        # Test authentication
        print("Authenticating with Google API...")
        llm.auth()
        print("✓ Authentication successful")

        # Test generation
        print("Generating text...")
        response, usage = llm._raw_generate(
            event_id="test-google-direct",
            system_prompt="You are a helpful assistant.",
            messages=[
                {
                    "message_type": "human",
                    "message": "What is the capital of France? Keep your answer brief.",
                }
            ],
            max_tokens=50,
        )

        # Print result
        print(f"\nResponse: {response}")
        print(f"Usage: {usage}")

    else:
        print(f"Provider {TEST_PROVIDER} testing not implemented yet")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

    # Print more error details if available
    if hasattr(e, "details"):
        print(f"Details: {e.details}")
    if hasattr(e, "provider"):
        print(f"Provider: {e.provider}")
