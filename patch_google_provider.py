"""
Patch script to update the Google LLM provider in LLuMinary.

This script applies the necessary fixes to the Google provider:
1. Updates environment variable handling to support GOOGLE_GEMINI_API_TOKEN
2. Fixes import path for get_secret function
3. Updates supported models list

Usage:
    python patch_google_provider.py
"""

import os
import re
import sys
from pathlib import Path

# Path to the Google provider file
GOOGLE_PROVIDER_PATH = Path("src/lluminary/models/providers/google.py")

def apply_patch():
    """Apply patches to fix Google provider issues."""
    if not GOOGLE_PROVIDER_PATH.exists():
        print(f"Error: Google provider file not found at {GOOGLE_PROVIDER_PATH}")
        return False

    # Read the original file
    with open(GOOGLE_PROVIDER_PATH, 'r') as f:
        content = f.read()

    # 1. Fix the import path
    content = re.sub(
        r'from \.\.\.utils import get_secret',
        'from ...utils.aws import get_secret',
        content
    )

    # 2. Fix environment variable handling
    content = re.sub(
        r'api_key = os\.environ\.get\("GOOGLE_API_KEY"\)',
        'api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GEMINI_API_TOKEN")',
        content
    )

    # 3. Update models list to include 1.5 models as fallback
    content = re.sub(
        r'SUPPORTED_MODELS = \[(.*?)\]',
        'SUPPORTED_MODELS = [\\1,\n        "gemini-1.5-flash",\n        "gemini-1.5-pro"]',
        content,
        flags=re.DOTALL
    )

    # 4. Update error details to mention both environment variables
    content = re.sub(
        r'"source_checked": \["GOOGLE_API_KEY",\s+"AWS Secrets Manager"\]',
        '"source_checked": ["GOOGLE_API_KEY", "GOOGLE_GEMINI_API_TOKEN", "AWS Secrets Manager"]',
        content
    )

    # Write the updated content
    with open(GOOGLE_PROVIDER_PATH, 'w') as f:
        f.write(content)

    print(f"✅ Successfully patched {GOOGLE_PROVIDER_PATH}")
    print("Changes made:")
    print("  1. Fixed import path for get_secret")
    print("  2. Added support for GOOGLE_GEMINI_API_TOKEN environment variable")
    print("  3. Updated supported models list to include Gemini 1.5 models")
    print("  4. Updated error messages to mention both environment variables")

    return True

def update_test_files():
    """Update test files for Google provider."""
    test_files = [
        Path("tests/unit/test_google_simplified.py"),
        Path("tests/unit/test_google_error_handling.py"),
    ]

    for test_file in test_files:
        if not test_file.exists():
            print(f"Warning: Test file not found at {test_file}")
            continue

        with open(test_file, 'r') as f:
            content = f.read()

        # Add test for GOOGLE_GEMINI_API_TOKEN if it's the error handling test
        if "test_google_error_handling.py" in str(test_file):
            if "test_auth_with_gemini_token" not in content:
                # Find the test_auth_with_env_var function
                match = re.search(r'def test_auth_with_env_var\(\):(.*?)def', content, re.DOTALL)
                if match:
                    # Create a new test function based on the existing one
                    new_test = match.group(1).replace(
                        'os.environ["GOOGLE_API_KEY"] = "env_test_key"',
                        'os.environ["GOOGLE_GEMINI_API_TOKEN"] = "env_gemini_token"'
                    )
                    new_test = new_test.replace(
                        'api_key="env_test_key"',
                        'api_key="env_gemini_token"'
                    )

                    # Add the new test function
                    new_function = f"""
def test_auth_with_gemini_token():
    """Test authentication using GOOGLE_GEMINI_API_TOKEN."""
    # Save original environment
    original_env = os.environ.get("GOOGLE_GEMINI_API_TOKEN")

    try:
        # Set environment variable
        os.environ["GOOGLE_GEMINI_API_TOKEN"] = "env_gemini_token"

        # Create GoogleLLM instance with mocked client
        with patch("google.genai.Client") as mock_client:
            llm = GoogleLLM("gemini-2.0-flash")
            llm.auth()

            # Verify client was initialized with the env var API key
            mock_client.assert_called_once_with(api_key="env_gemini_token")
    finally:
        # Restore original environment
        if original_env:
            os.environ["GOOGLE_GEMINI_API_TOKEN"] = original_env
        else:
            del os.environ["GOOGLE_GEMINI_API_TOKEN"]

{match.group(0)[match.group(0).find('def'):]}"""

                    content = content.replace(match.group(0), new_function)

                    with open(test_file, 'w') as f:
                        f.write(content)

                    print(f"✅ Added GOOGLE_GEMINI_API_TOKEN test to {test_file}")
                else:
                    print(f"Warning: Could not find test_auth_with_env_var in {test_file}")

if __name__ == "__main__":
    print("Applying patches to Google provider...")
    if apply_patch():
        update_test_files()
        print("\nPatches applied successfully. Please run the following to test:")
        print("python -m pytest tests/unit/test_google*.py -v")
    else:
        print("Failed to apply patches.")
        sys.exit(1)
