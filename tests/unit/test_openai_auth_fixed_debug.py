"""
Debug test for OpenAI provider authentication.

This module tests authentication mechanisms of the OpenAI provider
with comprehensive debug output.
"""

import json
import sys
import traceback
from unittest.mock import MagicMock, patch

from lluminary.models.providers.openai import OpenAILLM
from openai import AuthenticationError as OpenAIAuthError


def test_debug_openai_auth():
    """Debug test for OpenAI auth function."""
    print("\n=== Starting Debug Test for OpenAI Auth ===")

    try:
        # Create mock for get_secret function
        with patch(
            "src.lluminary.models.providers.openai.get_secret"
        ) as mock_get_secret:
            # Configure mock to return a valid API key
            mock_secret = {"api_key": "test-secret-key"}
            mock_get_secret.return_value = mock_secret

            # Create mock for OpenAI client
            with patch("lluminary.models.providers.openai.OpenAI") as mock_openai:
                # Set up mock client with models attribute
                mock_client = MagicMock()
                mock_models = MagicMock()
                mock_models.list.return_value = {"data": [{"id": "gpt-4o"}]}
                mock_client.models = mock_models
                mock_openai.return_value = mock_client

                # Mock time.sleep to speed up tests
                with patch("time.sleep"):
                    print("\n=== TEST 1: Basic Authentication ===")
                    # Create instance using __new__ to avoid __init__
                    openai_llm = OpenAILLM.__new__(OpenAILLM)
                    openai_llm.config = {}
                    openai_llm.api_base = None
                    openai_llm.model_name = "gpt-4o"

                    print(
                        f"Initial config: {json.dumps(openai_llm.config, default=str)}"
                    )
                    print(f"api_base: {openai_llm.api_base}")

                    # Call auth
                    print("Calling auth method...")
                    openai_llm.auth()
                    print("Auth method completed successfully")

                    # Print results
                    print(f"Final config: {json.dumps(openai_llm.config, default=str)}")
                    print(f"get_secret called with: {mock_get_secret.call_args}")
                    print(f"OpenAI client initialized with: {mock_openai.call_args}")
                    print(f"models.list called: {mock_models.list.called}")

                    # Reset mocks
                    mock_get_secret.reset_mock()
                    mock_openai.reset_mock()
                    mock_models.list.reset_mock()

                    print("\n=== TEST 2: Authentication with Organization ===")
                    # Create instance with organization in config
                    openai_llm = OpenAILLM.__new__(OpenAILLM)
                    openai_llm.config = {"organization": "test-org"}
                    openai_llm.api_base = None
                    openai_llm.model_name = "gpt-4o"

                    print(
                        f"Initial config: {json.dumps(openai_llm.config, default=str)}"
                    )

                    # Call auth
                    print("Calling auth method...")
                    openai_llm.auth()
                    print("Auth method completed successfully")

                    # Print results
                    print(f"Final config: {json.dumps(openai_llm.config, default=str)}")
                    print(f"OpenAI client initialized with: {mock_openai.call_args}")
                    print(
                        f"OpenAI client kwargs: {mock_openai.call_args[1] if len(mock_openai.call_args) > 1 else 'No kwargs'}"
                    )

                    # Reset mocks
                    mock_openai.reset_mock()
                    mock_models.list.reset_mock()

                    print("\n=== TEST 3: Credential Verification Failure ===")
                    # Create instance
                    openai_llm = OpenAILLM.__new__(OpenAILLM)
                    openai_llm.config = {}
                    openai_llm.api_base = None
                    openai_llm.model_name = "gpt-4o"

                    # Set up mock to raise exception during verification
                    mock_models.list.side_effect = OpenAIAuthError("Invalid API key")

                    # Call auth and expect exception
                    print("Calling auth method with invalid credentials...")
                    try:
                        openai_llm.auth()
                        print("WARNING: Auth did not raise exception as expected")
                    except Exception as auth_error:
                        print(
                            f"Auth raised exception as expected: {type(auth_error).__name__}: {auth_error!s}"
                        )
                        print(f"Exception type hierarchy: {type(auth_error).__mro__}")

                    # Print results
                    print(f"OpenAI client initialized with: {mock_openai.call_args}")

    except Exception as e:
        print(f"ERROR: {e!s}")
        print("Traceback:")
        traceback.print_exc(file=sys.stdout)

    print("=== Debug Test Complete ===")


if __name__ == "__main__":
    test_debug_openai_auth()
