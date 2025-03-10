"""
Patch the get_secret missing function and run tests.
"""

import os
import sys
from typing import Optional

# First, add get_secret to utils module
import lluminary.utils


def get_secret(secret_name: str, default: Optional[str] = None) -> str:
    """Get a secret from environment variables."""
    return os.environ.get(secret_name, default or "")


# Patch the module
lluminary.utils.get_secret = get_secret
print("✓ Patched utils.get_secret function")

# Now force reload other modules that might have failed to import due to missing get_secret
import importlib

for module_name in [
    "lluminary.models.providers",
    "lluminary.models.providers.openai",
    "lluminary.models.providers.google",
    "lluminary.models.providers.cohere",
    "lluminary.models.providers.anthropic",
    "lluminary.models.providers.bedrock",
    "lluminary.models.router",
]:
    try:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)
        print(f"✓ Loaded {module_name}")
    except Exception as e:
        print(f"✗ Error loading {module_name}: {e}")

# Map environment variables for testing
TOKEN_TO_KEY_MAPPING = {
    "OPENAI_API_TOKEN": "OPENAI_API_KEY",
    "ANTHROPIC_API_TOKEN": "ANTHROPIC_API_KEY",
    "COHERE_API_TOKEN": "COHERE_API_KEY",
}

# Map environment variables
for token_var, key_var in TOKEN_TO_KEY_MAPPING.items():
    if token_var in os.environ:
        os.environ[key_var] = os.environ[token_var]
        print(f"✓ Mapped {token_var} to {key_var}")

# Set up test configuration
from lluminary.models.router import PROVIDER_REGISTRY, get_llm_from_model

# Print available providers
print(f"\nAvailable providers: {list(PROVIDER_REGISTRY.keys())}")

# Test a simple case with OpenAI
print("\nTesting OpenAI:")
try:
    openai_llm = get_llm_from_model("gpt-4o-mini")
    print("✓ Successfully created OpenAI LLM instance")

    response, usage, _ = openai_llm.generate(
        event_id="test_openai",
        system_prompt="You are a helpful assistant.",
        messages=[
            {
                "message_type": "human",
                "message": "What is the capital of France? Keep your answer very brief.",
            }
        ],
        max_tokens=50,
    )

    print(f"✓ Response: {response}")
    print(f"✓ Usage: {usage}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test Anthropic
print("\nTesting Anthropic:")
try:
    anthropic_llm = get_llm_from_model("claude-haiku-3.5")
    print("✓ Successfully created Anthropic LLM instance")

    response, usage, _ = anthropic_llm.generate(
        event_id="test_anthropic",
        system_prompt="You are a helpful assistant.",
        messages=[
            {
                "message_type": "human",
                "message": "What is the capital of France? Keep your answer very brief.",
            }
        ],
        max_tokens=50,
    )

    print(f"✓ Response: {response}")
    print(f"✓ Usage: {usage}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test Cohere
print("\nTesting Cohere:")
try:
    cohere_llm = get_llm_from_model("cohere-command-light")
    print("✓ Successfully created Cohere LLM instance")

    response, usage, _ = cohere_llm.generate(
        event_id="test_cohere",
        system_prompt="You are a helpful assistant.",
        messages=[
            {
                "message_type": "human",
                "message": "What is the capital of France? Keep your answer very brief.",
            }
        ],
        max_tokens=50,
    )

    print(f"✓ Response: {response}")
    print(f"✓ Usage: {usage}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test Google
print("\nTesting Google:")
try:
    google_llm = get_llm_from_model("gemini-2.0-flash-lite")
    print("✓ Successfully created Google LLM instance")

    response, usage, _ = google_llm.generate(
        event_id="test_google",
        system_prompt="You are a helpful assistant.",
        messages=[
            {
                "message_type": "human",
                "message": "What is the capital of France? Keep your answer very brief.",
            }
        ],
        max_tokens=50,
    )

    print(f"✓ Response: {response}")
    print(f"✓ Usage: {usage}")
except Exception as e:
    print(f"✗ Error: {e}")
