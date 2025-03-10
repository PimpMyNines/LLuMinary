"""
Debug test for OpenAI provider to check auth implementation.
"""

import inspect

from lluminary.models.providers.openai import OpenAILLM


def test_print_auth_method():
    """Print the auth method implementation to debug."""
    print("\nAUTH METHOD IMPLEMENTATION:")
    print(inspect.getsource(OpenAILLM.auth))

    print("\nINIT METHOD IMPLEMENTATION:")
    print(inspect.getsource(OpenAILLM.__init__))

    assert True  # Always passes
