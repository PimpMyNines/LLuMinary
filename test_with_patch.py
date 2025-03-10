"""
Test individual providers with environment variable tokens.
"""

import os


# Map environment variables for testing
def map_tokens_to_keys():
    """Map TOKEN environment variables to KEY environment variables."""
    TOKEN_TO_KEY_MAPPING = {
        "OPENAI_API_TOKEN": "OPENAI_API_KEY",
        "ANTHROPIC_API_TOKEN": "ANTHROPIC_API_KEY",
        "COHERE_API_TOKEN": "COHERE_API_KEY",
    }

    for token_var, key_var in TOKEN_TO_KEY_MAPPING.items():
        if token_var in os.environ:
            os.environ[key_var] = os.environ[token_var]
            print(f"✓ Mapped {token_var} to {key_var}")


# Apply our monkey patch

# Map environment variables
map_tokens_to_keys()


# Define provider tests
def test_anthropic():
    """Test the Anthropic provider."""
    print("\nTESTING ANTHROPIC PROVIDER")
    print("=========================")

    try:
        # Import the provider class directly
        from lluminary.models.providers.anthropic import AnthropicLLM

        # Create instance with API key
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("✗ ANTHROPIC_API_KEY environment variable not set")
            return False

        print(f"✓ Found API key: {api_key[:5]}...{api_key[-5:]}")

        # Create provider instance directly
        llm = AnthropicLLM(
            model_name="claude-3-5-haiku-20241022",  # Provider-specific model name format
            api_key=api_key,
        )

        print("✓ Successfully created Anthropic LLM instance")

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
        print(f"✓ Response received: {response}")
        print(f"✓ Usage: {usage}")

        return True
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

        # Print more error details if available
        if hasattr(e, "details"):
            print(f"Details: {e.details}")
        if hasattr(e, "provider"):
            print(f"Provider: {e.provider}")

        return False


def test_openai():
    """Test the OpenAI provider."""
    print("\nTESTING OPENAI PROVIDER")
    print("=====================")

    try:
        # Import the provider class directly
        from lluminary.models.providers.openai import OpenAILLM

        # Create instance with API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("✗ OPENAI_API_KEY environment variable not set")
            return False

        print(f"✓ Found API key: {api_key[:5]}...{api_key[-5:]}")

        # Create provider instance directly
        llm = OpenAILLM(
            model_name="gpt-4o-mini",  # Provider-specific model name format
            api_key=api_key,
        )

        print("✓ Successfully created OpenAI LLM instance")

        # Test generation
        print("Generating text...")
        result = llm.generate(
            event_id="test-openai-direct",
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
        print(f"✓ Response received: {response}")
        print(f"✓ Usage: {usage}")

        return True
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

        # Print more error details if available
        if hasattr(e, "details"):
            print(f"Details: {e.details}")
        if hasattr(e, "provider"):
            print(f"Provider: {e.provider}")

        return False


def test_cohere():
    """Test the Cohere provider."""
    print("\nTESTING COHERE PROVIDER")
    print("=====================")

    try:
        # Import the provider class directly
        from lluminary.models.providers.cohere import CohereLLM

        # Create instance with API key
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            print("✗ COHERE_API_KEY environment variable not set")
            return False

        print(f"✓ Found API key: {api_key[:5]}...{api_key[-5:]}")

        # Create provider instance directly
        llm = CohereLLM(
            model_name="command-light",  # Provider-specific model name format
            api_key=api_key,
        )

        print("✓ Successfully created Cohere LLM instance")

        # Test generation
        print("Generating text...")
        result = llm.generate(
            event_id="test-cohere-direct",
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
        print(f"✓ Response received: {response}")
        print(f"✓ Usage: {usage}")

        return True
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

        # Print more error details if available
        if hasattr(e, "details"):
            print(f"Details: {e.details}")
        if hasattr(e, "provider"):
            print(f"Provider: {e.provider}")

        return False


def test_google():
    """Test the Google provider."""
    print("\nTESTING GOOGLE PROVIDER")
    print("====================")

    try:
        # Import the provider class directly
        from lluminary.models.providers.google import GoogleLLM

        # Create instance with API key
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("✗ GOOGLE_API_KEY environment variable not set")
            return False

        print(f"✓ Found API key: {api_key[:5]}...{api_key[-5:]}")

        # Create provider instance directly
        llm = GoogleLLM(
            model_name="gemini-2.0-flash-lite-preview-02-05",  # Provider-specific model name format
            api_key=api_key,
        )

        print("✓ Successfully created Google LLM instance")

        # Test generation
        print("Generating text...")
        result = llm.generate(
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

        # Unpack results
        response, usage, messages = result

        # Print result
        print(f"✓ Response received: {response}")
        print(f"✓ Usage: {usage}")

        return True
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")

        # Print more error details if available
        if hasattr(e, "details"):
            print(f"Details: {e.details}")
        if hasattr(e, "provider"):
            print(f"Provider: {e.provider}")

        return False


# Run all tests
test_results = {
    "anthropic": test_anthropic(),
    "openai": test_openai(),
    "cohere": test_cohere(),
    # "google": test_google(),  # Uncomment if Google API key is set
}

# Print summary
print("\nTEST RESULTS SUMMARY")
print("===================")
for provider, success in test_results.items():
    if success:
        print(f"✓ {provider.upper()}: SUCCESS")
    else:
        print(f"✗ {provider.upper()}: FAILED")

# Record issues for documentation
with open("provider_test_results.md", "w") as f:
    f.write("# Provider Test Results\n\n")
    f.write("Testing performed on: " + os.popen("date").read().strip() + "\n\n")

    for provider, success in test_results.items():
        if success:
            f.write(f"## {provider.capitalize()}: ✅ SUCCESS\n\n")
        else:
            f.write(f"## {provider.capitalize()}: ❌ FAILED\n\n")
            f.write("Details:\n\n")
            f.write("- Issue with provider implementation or credentials\n")
            f.write("- Needs further investigation\n\n")

print("\nResults written to provider_test_results.md")
