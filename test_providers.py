"""
Test script to verify all providers with real credentials.
This script uses actual API credentials to test each provider.
"""

import os
import sys
from typing import Any, Dict

# Ensure we can import from the package
sys.path.insert(0, os.path.abspath("."))

from lluminary.models.router import get_llm_from_model

# Setup environment variable mapping
TOKEN_TO_KEY_MAPPING = {
    "OPENAI_API_TOKEN": "OPENAI_API_KEY",
    "ANTHROPIC_API_TOKEN": "ANTHROPIC_API_KEY",
    "COHERE_API_TOKEN": "COHERE_API_KEY",
    # For Google, we'll handle it differently since the naming is different
}

# Map environment variables
for token_var, key_var in TOKEN_TO_KEY_MAPPING.items():
    if token_var in os.environ:
        os.environ[key_var] = os.environ[token_var]
        print(f"✓ Mapped {token_var} to {key_var}")

# Test configuration - using the correct model names from available models list
TEST_MODELS = [
    "gpt-4o-mini",  # OpenAI
    "claude-haiku-3.5",  # Anthropic
    "cohere-command-light",  # Cohere
    "gemini-2.0-flash-lite",  # Google
]


# Create a get_secret function to address the import error
def get_secret(secret_name: str) -> str:
    """Temporary function to mimic expected get_secret behavior."""
    import os

    return os.environ.get(secret_name, "")


TEST_PROMPT = "What is the capital of France? Keep your answer very brief."

test_results: Dict[str, Dict[str, Any]] = {}

print("\n" + "=" * 60)
print("TESTING ALL PROVIDERS WITH REAL CREDENTIALS")
print("=" * 60)

for model_name in TEST_MODELS:
    provider = model_name.split("-")[0] if "-" in model_name else model_name
    print(f"\nTesting {provider} model: {model_name}")

    test_results[model_name] = {
        "provider": provider,
        "success": False,
        "error": None,
        "response": None,
    }

    try:
        # Initialize model
        llm = get_llm_from_model(model_name)

        # Generate response
        print(f"Generating response with {model_name}...")
        response, usage, _ = llm.generate(
            event_id=f"test_{provider}",
            system_prompt="You are a helpful assistant.",
            messages=[{"message_type": "human", "message": TEST_PROMPT}],
            max_tokens=50,
        )

        # Record success
        test_results[model_name]["success"] = True
        test_results[model_name]["response"] = response
        test_results[model_name]["usage"] = usage

        print(f"✓ Success! Response: {response}")
        print(f"  Usage: {usage}")

    except Exception as e:
        # Record error
        test_results[model_name]["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }

        print(f"✗ Error: {type(e).__name__}: {e!s}")

print("\n" + "=" * 60)
print("TEST RESULTS SUMMARY")
print("=" * 60)

all_success = True
errors = []

for model_name, result in test_results.items():
    provider = result["provider"]
    if result["success"]:
        print(f"✓ {provider} ({model_name}): SUCCESS")
    else:
        all_success = False
        print(f"✗ {provider} ({model_name}): FAILED - {result['error']['type']}")
        errors.append(
            {
                "model": model_name,
                "provider": provider,
                "error_type": result["error"]["type"],
                "error_message": result["error"]["message"],
            }
        )

print("\n" + "=" * 60)
if all_success:
    print("ALL PROVIDERS TESTED SUCCESSFULLY!")
else:
    print("SOME PROVIDERS FAILED:")
    for error in errors:
        print(f"\n{error['provider']} ({error['model']}):")
        print(f"  Error type: {error['error_type']}")
        print(f"  Error message: {error['error_message']}")

print("=" * 60)
