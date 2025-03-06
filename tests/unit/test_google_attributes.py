"""
Tests for the Google provider's class attributes and basic methods.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.google import GoogleLLM


def test_supported_models():
    """Test that the supported models list is correct."""
    expected_models = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-pro-exp-02-05",
        "gemini-2.0-flash-thinking-exp-01-21",
    ]
    assert set(GoogleLLM.SUPPORTED_MODELS) == set(expected_models)


def test_context_window():
    """Test that context windows are defined for all models."""
    for model in GoogleLLM.SUPPORTED_MODELS:
        assert model in GoogleLLM.CONTEXT_WINDOW
        assert isinstance(GoogleLLM.CONTEXT_WINDOW[model], int)
        assert GoogleLLM.CONTEXT_WINDOW[model] > 0

        # Specific checks for known values
        if model == "gemini-2.0-flash":
            assert GoogleLLM.CONTEXT_WINDOW[model] == 128000

        if model == "gemini-2.0-flash-lite-preview-02-05":
            assert GoogleLLM.CONTEXT_WINDOW[model] == 128000


def test_cost_per_model():
    """Test that cost info is defined for all models."""
    for model in GoogleLLM.SUPPORTED_MODELS:
        assert model in GoogleLLM.COST_PER_MODEL

        model_costs = GoogleLLM.COST_PER_MODEL[model]
        assert "read_token" in model_costs
        assert "write_token" in model_costs
        assert "image_cost" in model_costs

        assert model_costs["read_token"] > 0
        assert model_costs["write_token"] > 0
        assert model_costs["image_cost"] > 0


def test_image_support():
    """Test that image support is defined correctly."""
    assert GoogleLLM.SUPPORTS_IMAGES is True


def test_init():
    """Test basic initialization without authentication."""
    with patch.object(GoogleLLM, "auth"):
        llm = GoogleLLM("gemini-2.0-flash")
        assert llm.model_name == "gemini-2.0-flash"
        assert llm.client is None  # Client not initialized until auth


def test_init_with_kwargs():
    """Test initialization with additional kwargs."""
    with patch.object(GoogleLLM, "auth"):
        llm = GoogleLLM("gemini-2.0-flash", api_key="test-key", custom_param="value")
        assert llm.model_name == "gemini-2.0-flash"
        assert llm.config["api_key"] == "test-key"
        assert llm.config["custom_param"] == "value"


def test_supports_image_input():
    """Test the supports_image_input method."""
    with patch.object(GoogleLLM, "auth"):
        llm = GoogleLLM("gemini-2.0-flash")
        assert llm.supports_image_input() is True
        assert llm.supports_image_input() == GoogleLLM.SUPPORTS_IMAGES


def test_get_supported_models():
    """Test the get_supported_models method."""
    with patch.object(GoogleLLM, "auth"):
        llm = GoogleLLM("gemini-2.0-flash")
        models = llm.get_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gemini-2.0-flash" in models


def test_get_model_costs():
    """Test the get_model_costs method."""
    with patch.object(GoogleLLM, "auth"):
        llm = GoogleLLM("gemini-2.0-flash")
        costs = llm.get_model_costs()
        assert "read_token" in costs
        assert "write_token" in costs
        assert "image_cost" in costs
        assert costs["read_token"] > 0
        assert costs["write_token"] > 0
        assert costs["image_cost"] > 0

        # Test with different model
        llm.model_name = "gemini-2.0-pro-exp-02-05"
        costs = llm.get_model_costs()
        assert costs["read_token"] > 0
        assert costs["write_token"] > 0
        assert costs["image_cost"] > 0


def test_process_image_local():
    """Test processing a local image."""
    with patch.object(GoogleLLM, "auth"):
        llm = GoogleLLM("gemini-2.0-flash")

        mock_image = MagicMock()
        with patch("PIL.Image.open", return_value=mock_image) as mock_open:
            result = llm._process_image("/path/to/image.jpg")

            # Verify Image.open was called with correct path
            mock_open.assert_called_once_with("/path/to/image.jpg")
            assert result == mock_image


def test_process_image_url():
    """Test processing an image from URL."""
    with patch.object(GoogleLLM, "auth"):
        llm = GoogleLLM("gemini-2.0-flash")

        # Create mock response for URL fetch
        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"

        # Create mock image
        mock_image = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            with patch("PIL.Image.open", return_value=mock_image) as mock_open:
                result = llm._process_image(
                    "https://example.com/image.jpg", is_url=True
                )

                # Verify HTTP request was made
                mock_get.assert_called_once_with(
                    "https://example.com/image.jpg", timeout=10
                )

                # Verify image was processed
                assert mock_open.call_count == 1
                assert result == mock_image


def test_process_image_error_local():
    """Test error handling for local image processing."""
    with patch.object(GoogleLLM, "auth"):
        llm = GoogleLLM("gemini-2.0-flash")

        # Make Image.open raise an exception
        with patch("PIL.Image.open", side_effect=Exception("Failed to open image")):
            with pytest.raises(Exception) as exc_info:
                llm._process_image("/path/to/bad_image.jpg")

            assert "Failed to process image file" in str(exc_info.value)
            assert "/path/to/bad_image.jpg" in str(exc_info.value)


def test_process_image_error_url():
    """Test error handling for URL image processing."""
    with patch.object(GoogleLLM, "auth"):
        llm = GoogleLLM("gemini-2.0-flash")

        # Make requests.get raise an exception
        with patch("requests.get", side_effect=Exception("Failed to fetch image")):
            with pytest.raises(Exception) as exc_info:
                llm._process_image("https://example.com/bad_image.jpg", is_url=True)

            assert "Failed to process image URL" in str(exc_info.value)
            assert "https://example.com/bad_image.jpg" in str(exc_info.value)
