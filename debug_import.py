"""
Debug script to trace provider module imports.
"""

import sys
import traceback

print("Current Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nAttempting to import from providers...")
try:
    from lluminary.models.providers import (
        AnthropicLLM,
        BedrockLLM,
        GoogleLLM,
        OpenAILLM,
    )

    print("✓ Successfully imported provider classes")
except ImportError as e:
    print(f"✗ Import error: {e}")
    traceback.print_exc()

print("\nTrying to import provider modules separately...")
for provider in ["openai", "anthropic", "google", "bedrock"]:
    try:
        print(f"Trying to import {provider}...")
        module = __import__(f"lluminary.models.providers.{provider}", fromlist=["*"])
        print(f"✓ Successfully imported {provider} module")
        print(f"  Module attributes: {dir(module)[:10]}...")
    except ImportError as e:
        print(f"✗ Error importing {provider}: {e}")

print("\nChecking for get_secret in utils...")
try:
    from lluminary.utils import get_secret

    print("✓ Successfully imported get_secret from utils")
except ImportError as e:
    print(f"✗ Error importing get_secret: {e}")

    try:
        from lluminary.utils import aws

        print("✓ Successfully imported aws from utils")
        print(f"  aws module attributes: {dir(aws)[:10]}...")
    except ImportError as e:
        print(f"✗ Error importing aws: {e}")

    print("\nTrying to add get_secret to utils...")
    try:
        import lluminary.utils

        # Adding get_secret function to utils module
        def get_secret(secret_name, default=None):
            """Get a secret from environment."""
            import os

            return os.environ.get(secret_name, default or "")

        # Add function to module
        lluminary.utils.get_secret = get_secret
        print("✓ Added get_secret function to utils module")
    except Exception as e:
        print(f"✗ Error adding get_secret: {e}")
