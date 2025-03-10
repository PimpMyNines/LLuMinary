"""
Module mocks for Google API and related classes.

This module provides mock implementations of Google's generativeai module
and associated classes to enable testing without requiring the actual
dependencies to be installed.
"""

import sys
import types
from unittest.mock import MagicMock, patch


# Create mock module structure
def create_mock_modules():
    """
    Create and inject mock versions of the Google modules into sys.modules.
    This allows tests to import these modules even if they're not installed.
    """
    # Create mock modules
    mock_google = types.ModuleType("google")
    mock_genai = types.ModuleType("google.genai")
    mock_generativeai = types.ModuleType("google.generativeai")
    mock_types = types.ModuleType("google.genai.types")
    mock_content_types = types.ModuleType("google.genai.types.content_types")
    mock_generation_types = types.ModuleType("google.genai.types.generation_types")

    # Create mock classes
    mock_genai.GenerativeModel = MagicMock()
    mock_generativeai.GenerativeModel = MagicMock()
    mock_types.Content = MagicMock()
    mock_types.Part = MagicMock()
    mock_types.GenerateContentConfig = MagicMock()

    # Setup Part.from_text and other methods
    mock_types.Part.from_text = MagicMock()
    mock_types.Part.from_function_call = MagicMock()
    mock_types.Part.from_function_response = MagicMock()

    # Setup content types
    mock_content_types.Content = MagicMock()
    mock_content_types.Part = MagicMock()
    mock_content_types.Part.from_text = MagicMock()

    # Add modules to sys.modules if they don't exist
    if "google" not in sys.modules:
        sys.modules["google"] = mock_google
    if "google.genai" not in sys.modules:
        sys.modules["google.genai"] = mock_genai
    if "google.generativeai" not in sys.modules:
        sys.modules["google.generativeai"] = mock_generativeai
    if "google.genai.types" not in sys.modules:
        sys.modules["google.genai.types"] = mock_types
    if "google.genai.types.content_types" not in sys.modules:
        sys.modules["google.genai.types.content_types"] = mock_content_types
    if "google.genai.types.generation_types" not in sys.modules:
        sys.modules["google.genai.types.generation_types"] = mock_generation_types

    # Return the mocks for use in tests
    return {
        "google": mock_google,
        "google.genai": mock_genai,
        "google.generativeai": mock_generativeai,
        "google.genai.types": mock_types,
        "google.genai.types.content_types": mock_content_types,
        "google.genai.types.generation_types": mock_generation_types,
    }


# Create a patch context manager for tests
def patch_google_modules():
    """
    Create a context manager that patches Google modules for tests.
    """
    mocks = create_mock_modules()

    patches = [
        patch.dict(
            "sys.modules",
            {
                "google": mocks["google"],
                "google.genai": mocks["google.genai"],
                "google.generativeai": mocks["google.generativeai"],
                "google.genai.types": mocks["google.genai.types"],
                "google.genai.types.content_types": mocks[
                    "google.genai.types.content_types"
                ],
                "google.genai.types.generation_types": mocks[
                    "google.genai.types.generation_types"
                ],
            },
        ),
    ]

    return patches, mocks
