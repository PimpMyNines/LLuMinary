"""
Verifies that the module mocking approach works correctly.
"""

import sys
from unittest.mock import MagicMock

# Import our module mock functions
from tests.unit.test_google_module_mock import create_mock_modules, patch_google_modules


def test_module_mock_creation():
    """Test that our module mocks can be created."""
    # Create mock modules
    mocks = create_mock_modules()

    # Verify that the mocks were created
    assert "google" in mocks
    assert "google.genai" in mocks
    assert "google.generativeai" in mocks
    assert "google.genai.types" in mocks
    assert "google.genai.types.content_types" in mocks
    assert "google.genai.types.generation_types" in mocks

    # Verify that key classes are available
    assert hasattr(mocks["google.genai"], "GenerativeModel")
    assert hasattr(mocks["google.generativeai"], "GenerativeModel")
    assert hasattr(mocks["google.genai.types"], "Content")
    assert hasattr(mocks["google.genai.types"], "Part")
    assert hasattr(mocks["google.genai.types"], "GenerateContentConfig")

    # Verify that the modules were added to sys.modules
    assert "google" in sys.modules
    assert "google.genai" in sys.modules
    assert "google.generativeai" in sys.modules
    assert "google.genai.types" in sys.modules
    assert "google.genai.types.content_types" in sys.modules
    assert "google.genai.types.generation_types" in sys.modules


def test_patching_context_manager():
    """Test that our patching context manager works."""
    # Get patches and mocks
    patches, mocks = patch_google_modules()

    # Verify that patches were created
    assert len(patches) > 0

    # Apply patches
    for p in patches:
        p.start()

    try:
        # Now try to import previously unimportable modules
        import google.generativeai
        from google.genai import GenerativeModel
        from google.genai.types import Content, Part
        from google.genai.types.content_types import Content as ContentType

        # Verify that the imports worked and are our mocks
        assert google.generativeai is not None
        assert GenerativeModel is not None
        assert Content is not None
        assert Part is not None
        assert ContentType is not None

        # Verify that these are indeed our mocks
        assert isinstance(GenerativeModel, MagicMock)
        assert isinstance(Content, MagicMock)
        assert isinstance(Part, MagicMock)
        assert isinstance(ContentType, MagicMock)

    finally:
        # Clean up patches
        for p in patches:
            p.stop()


def test_streaming_import_mock():
    """Test that we can import the streaming-related modules."""
    # Apply patches
    patches, mocks = patch_google_modules()
    for p in patches:
        p.start()

    try:
        # This part fails in the real tests - try to mock it here
        # and see if it works
        import google.generativeai

        # Setup mock model
        mock_model = MagicMock()
        mock_model.generate_content.return_value = [
            MagicMock(text="Hello"),
            MagicMock(text=" World"),
            MagicMock(text=""),
        ]
        google.generativeai.GenerativeModel.return_value = mock_model

        # Now try to create and use a GenerativeModel
        model = google.generativeai.GenerativeModel(model_name="test-model")

        # Call generate_content with stream=True
        chunks = list(model.generate_content("Hello", stream=True))

        # Verify chunks
        assert len(chunks) == 3
        assert chunks[0].text == "Hello"
        assert chunks[1].text == " World"
        assert chunks[2].text == ""

    finally:
        # Clean up patches
        for p in patches:
            p.stop()
