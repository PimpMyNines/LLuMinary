"""
Tests for OpenAI provider image generation functionality.

This module focuses on testing the image generation capabilities
of the OpenAI provider, including DALL-E model integration.
"""
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import requests

from src.llmhandler.models.providers.openai import OpenAILLM


@pytest.fixture
def openai_llm():
    """Fixture for OpenAI LLM instance."""
    with patch.object(OpenAILLM, "auth") as mock_auth:
        # Mock authentication to avoid API errors
        mock_auth.return_value = None

        # Create the LLM instance with mock API key
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Initialize client attribute directly for tests
        llm.client = MagicMock()
        
        yield llm


@pytest.fixture
def mock_image_response():
    """Create a mock image generation response with realistic structure."""
    # Create mock image data
    data = [
        MagicMock(url="https://example.com/image1.png"),
        MagicMock(b64_json="base64encodedimagedata")
    ]
    
    # Create the complete response object
    response = MagicMock()
    response.data = data
    
    return response


def test_generate_image_basic(openai_llm, mock_image_response):
    """Test basic image generation functionality."""
    # Create a simpler mock response
    simple_response = MagicMock()
    simple_response.data = [MagicMock(url="https://example.com/image1.png")]
    
    # Mock the OpenAI images.generate method
    openai_llm.client.images.generate.return_value = simple_response
    
    # Call generate_image with basic parameters
    prompt = "A beautiful sunset over mountains"
    images = openai_llm.generate_image(prompt)
    
    # Verify OpenAI client was called correctly
    openai_llm.client.images.generate.assert_called_once()
    call_args = openai_llm.client.images.generate.call_args[1]
    assert call_args["prompt"] == prompt
    assert call_args["model"] == "dall-e-3"  # Default model
    
    # Verify image structure
    assert len(images) == 1
    assert "url" in images[0]
    assert images[0]["url"] == "https://example.com/image1.png"


def test_generate_image_with_options(openai_llm, mock_image_response):
    """Test image generation with custom options."""
    # Mock the OpenAI images.generate method
    openai_llm.client.images.generate.return_value = mock_image_response
    
    # Call generate_image with custom parameters
    prompt = "A futuristic city"
    options = {
        "model": "dall-e-2",
        "n": 2,
        "size": "1024x1024",
        "quality": "standard",
        "style": "natural",
        "response_format": "url"
    }
    
    images = openai_llm.generate_image(prompt, **options)
    
    # Verify OpenAI client was called with the correct options
    call_args = openai_llm.client.images.generate.call_args[1]
    assert call_args["model"] == "dall-e-2"
    assert call_args["n"] == 2
    assert call_args["size"] == "1024x1024"
    assert call_args["quality"] == "standard"
    assert call_args["style"] == "natural"
    assert call_args["response_format"] == "url"


def test_generate_image_response_formats(openai_llm):
    """Test different response formats for image generation."""
    # Test URL response format
    url_response = MagicMock()
    url_response.data = [MagicMock(url="https://example.com/image.png")]
    
    # Test b64_json response format
    b64_response = MagicMock()
    b64_response.data = [MagicMock(b64_json="base64data")]
    
    # Set up the client to return different responses based on parameters
    def mock_generate(**kwargs):
        if kwargs.get("response_format") == "url":
            return url_response
        else:
            return b64_response
    
    openai_llm.client.images.generate.side_effect = mock_generate
    
    # Test URL format
    url_images = openai_llm.generate_image("Test prompt", response_format="url")
    assert "url" in url_images[0]
    assert url_images[0]["url"] == "https://example.com/image.png"
    
    # Test b64_json format
    b64_images = openai_llm.generate_image("Test prompt", response_format="b64_json")
    assert "data" in b64_images[0]
    assert b64_images[0]["data"] == "base64data"


def test_generate_image_error_handling(openai_llm):
    """Test error handling in image generation."""
    # Test API error
    openai_llm.client.images.generate.side_effect = Exception("API error")
    
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("Test prompt")
    
    assert "Error generating image" in str(excinfo.value)
    assert "API error" in str(excinfo.value)
    
    # Test content policy violation
    openai_llm.client.images.generate.side_effect = Exception("Your request was rejected as a result of our safety system")
    
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("Something inappropriate")
    
    assert "safety system" in str(excinfo.value)


def test_generate_image_cost_tracking(openai_llm):
    """Test cost tracking for image generation."""
    # Create a mock response with a single image
    mock_response = MagicMock()
    mock_response.data = [MagicMock(url="https://example.com/image.png")]
    openai_llm.client.images.generate.return_value = mock_response
    
    # Define image generation costs
    image_costs = {
        "dall-e-3": {
            "1024x1024": 0.040,
            "1024x1792": 0.080
        },
        "dall-e-2": {
            "1024x1024": 0.020,
            "512x512": 0.018
        }
    }
    
    # Patch the cost tracking attribute
    with patch.object(openai_llm, "IMAGE_GENERATION_COSTS", image_costs):
        # Generate image with dall-e-3 at 1024x1024
        images, usage = openai_llm.generate_image("Test prompt", model="dall-e-3", size="1024x1024", return_usage=True)
        
        # Verify cost calculation
        assert "cost" in usage
        assert usage["cost"] == 0.040  # Cost for 1 image
        
        # Reset mock
        openai_llm.client.images.generate.reset_mock()
        
        # Generate image with dall-e-2 at 512x512
        images, usage = openai_llm.generate_image("Test prompt", model="dall-e-2", size="512x512", return_usage=True)
        
        # Verify cost calculation
        assert usage["cost"] == 0.018  # Cost for 1 image
        
        # Reset mock and test with multiple images
        openai_llm.client.images.generate.reset_mock()
        
        # Create a new mock response with multiple images
        multi_response = MagicMock()
        multi_response.data = [MagicMock(url=f"https://example.com/image{i}.png") for i in range(3)]
        openai_llm.client.images.generate.return_value = multi_response
        
        # Generate multiple images
        images, usage = openai_llm.generate_image("Test prompt", model="dall-e-3", size="1024x1024", n=3, return_usage=True)
        
        # Verify cost calculation for multiple images
        assert usage["cost"] == 0.040 * 3  # Cost for 3 images


def test_generate_image_with_different_sizes(openai_llm, mock_image_response):
    """Test image generation with different size options."""
    # Mock the OpenAI images.generate method
    openai_llm.client.images.generate.return_value = mock_image_response
    
    # Test various sizes with DALL-E 3
    sizes = ["1024x1024", "1792x1024", "1024x1792"]
    
    for size in sizes:
        # Reset mock
        openai_llm.client.images.generate.reset_mock()
        
        # Generate image with specific size
        openai_llm.generate_image("Test prompt", model="dall-e-3", size=size)
        
        # Verify size parameter was passed correctly
        call_args = openai_llm.client.images.generate.call_args[1]
        assert call_args["size"] == size


def test_generate_image_with_file_output(openai_llm, mock_image_response):
    """Test saving generated images to file."""
    # Set up mock URL in the response
    mock_url_response = MagicMock()
    mock_url_response.data = [MagicMock(url="https://example.com/image.png")]
    openai_llm.client.images.generate.return_value = mock_url_response
    
    # Set up mock HTTP response for downloading the image
    mock_http_response = MagicMock()
    mock_http_response.content = b"test image data"
    
    # Create temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_image.png")
        
        # Mock the requests.get function that would download the image
        with patch("requests.get", return_value=mock_http_response) as mock_get:
            # Generate image with output path
            result = openai_llm.generate_image("Test prompt", output_path=output_path)
            
            # Verify HTTP request was made to download the image
            mock_get.assert_called_once_with("https://example.com/image.png")
            
            # Verify file was created with correct content
            assert os.path.exists(output_path)
            with open(output_path, "rb") as f:
                assert f.read() == b"test image data"
            
            # Verify the function still returns the expected result
            assert result[0]["url"] == "https://example.com/image.png"


def test_generate_image_with_file_output_b64(openai_llm):
    """Test saving base64 encoded images to file."""
    # Set up mock response with base64 data
    mock_b64_response = MagicMock()
    mock_b64_response.data = [MagicMock(b64_json="dGVzdCBpbWFnZSBkYXRh")]  # base64 for "test image data"
    openai_llm.client.images.generate.return_value = mock_b64_response
    
    # Create temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_image_b64.png")
        
        # Generate image with output path and b64_json response format
        with patch("base64.b64decode", return_value=b"test image data") as mock_b64decode:
            result = openai_llm.generate_image(
                "Test prompt", 
                output_path=output_path,
                response_format="b64_json"
            )
            
            # Verify base64 decoding was called
            mock_b64decode.assert_called_once_with("dGVzdCBpbWFnZSBkYXRh")
            
            # Verify file should have been created with correct content
            assert os.path.exists(output_path)
            with open(output_path, "rb") as f:
                assert f.read() == b"test image data"
            
            # Verify the function still returns the expected result
            assert result[0]["data"] == "dGVzdCBpbWFnZSBkYXRh"


def test_generate_multiple_images(openai_llm):
    """Test generating multiple images in one request."""
    # Create a mock response with multiple images
    multi_response = MagicMock()
    multi_response.data = [
        MagicMock(url=f"https://example.com/image{i}.png") 
        for i in range(3)
    ]
    
    # Mock the API call
    openai_llm.client.images.generate.return_value = multi_response
    
    # Generate multiple images
    images = openai_llm.generate_image("Test prompt", n=3)
    
    # Verify the API was called with the correct n parameter
    call_args = openai_llm.client.images.generate.call_args[1]
    assert call_args["n"] == 3
    
    # Verify the result contains all 3 images
    assert len(images) == 3
    for i, image in enumerate(images):
        assert image["url"] == f"https://example.com/image{i}.png"


def test_generate_multiple_images_with_file_output(openai_llm):
    """Test generating and saving multiple images."""
    # Create a mock response with multiple images
    multi_response = MagicMock()
    multi_response.data = [
        MagicMock(url=f"https://example.com/image{i}.png") 
        for i in range(3)
    ]
    
    # Mock the API call
    openai_llm.client.images.generate.return_value = multi_response
    
    # Mock HTTP responses for downloading the images
    mock_http_responses = [MagicMock() for _ in range(3)]
    for i, resp in enumerate(mock_http_responses):
        resp.content = f"test image data {i}".encode()
    
    # Create temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_image_{index}.png")
        
        # Mock the requests.get function with different responses
        with patch("requests.get", side_effect=mock_http_responses) as mock_get:
            # Generate multiple images with output path
            images = openai_llm.generate_image("Test prompt", n=3, output_path=output_path)
            
            # Verify HTTP request was made for each image
            assert mock_get.call_count == 3
            for i in range(3):
                mock_get.assert_any_call(f"https://example.com/image{i}.png")
                
                # Verify all files were created with correct content
                file_path = output_path.replace("{index}", str(i))
                assert os.path.exists(file_path)
                with open(file_path, "rb") as f:
                    assert f.read() == f"test image data {i}".encode()


def test_generate_image_invalid_parameters(openai_llm):
    """Test handling of invalid parameters in image generation."""
    # Test invalid size
    openai_llm.client.images.generate.side_effect = Exception("Invalid size. Must be one of 1024x1024, 1024x1792, 1792x1024")
    
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("Test prompt", size="invalid_size")
    assert "size" in str(excinfo.value).lower()
    
    # Reset mock for next test
    openai_llm.client.images.generate.reset_mock()
    openai_llm.client.images.generate.side_effect = Exception("Invalid model. Must be one of dall-e-2, dall-e-3")
    
    # Test invalid model
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("Test prompt", model="not-a-real-model")
    assert "model" in str(excinfo.value).lower()
    
    # Reset mock for next test
    openai_llm.client.images.generate.reset_mock()
    openai_llm.client.images.generate.side_effect = Exception("Your request was rejected due to safety concerns")
    
    # Test content policy violation
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("Test prompt with potential policy violation")
    assert "safety" in str(excinfo.value).lower()
    
    # Reset mock for next test
    openai_llm.client.images.generate.reset_mock()
    openai_llm.client.images.generate.side_effect = Exception("n must be between 1 and 10")
    
    # Test invalid n parameter
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("Test prompt", n=20)
    assert "n must be between" in str(excinfo.value)


def test_generate_image_empty_prompt(openai_llm):
    """Test image generation with empty prompt."""
    # Set up error for empty prompt
    openai_llm.client.images.generate.side_effect = Exception("Prompt cannot be empty")
    
    # Test with empty prompt
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("")
    assert "prompt" in str(excinfo.value).lower()
    
    # Test with only whitespace
    openai_llm.client.images.generate.reset_mock()
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("   ")
    assert "prompt" in str(excinfo.value).lower()


def test_generate_image_rate_limiting(openai_llm):
    """Test handling of rate limiting errors."""
    # Simulate rate limit error
    openai_llm.client.images.generate.side_effect = Exception("Rate limit exceeded")
    
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("Test prompt")
    assert "rate limit" in str(excinfo.value).lower()
    
    # Reset mock and test with API timeout
    openai_llm.client.images.generate.reset_mock()
    openai_llm.client.images.generate.side_effect = Exception("Request timed out")
    
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("Test prompt")
    assert "timed out" in str(excinfo.value).lower()


def test_generate_image_handler_integration():
    """Test integration with LLMHandler class."""
    from src.llmhandler.handler import LLMHandler
    
    # Create handler mock
    with patch("src.llmhandler.handler.LLMHandler") as mock_handler_class:
        # Create mock instance
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        
        # Create mock LLM
        mock_llm = MagicMock()
        mock_llm.generate_image.return_value = [{"url": "https://example.com/image.png"}]
        
        # Set up handler to return our mock LLM
        mock_handler.get_llm.return_value = mock_llm
        
        # Create handler
        handler = LLMHandler()
        
        # Use handler to generate image
        handler.generate_image("Test prompt", model="dall-e-3")
        
        # Verify LLM was obtained with correct model
        mock_handler.get_llm.assert_called_once_with("dall-e-3")
        
        # Verify generate_image was called on the LLM
        mock_llm.generate_image.assert_called_once_with("Test prompt")


def test_generate_image_network_error(openai_llm):
    """Test handling of network errors during image generation."""
    # Simulate network error during API call
    openai_llm.client.images.generate.side_effect = Exception("Connection error")
    
    with pytest.raises(Exception) as excinfo:
        openai_llm.generate_image("Test prompt")
    assert "connection error" in str(excinfo.value).lower()
    
    # Simulate network error during image download with URL response
    mock_url_response = MagicMock()
    mock_url_response.data = [MagicMock(url="https://example.com/image.png")]
    
    openai_llm.client.images.generate.reset_mock()
    openai_llm.client.images.generate.side_effect = None
    openai_llm.client.images.generate.return_value = mock_url_response
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_image.png")
        
        # Simulate network error during download
        with patch("requests.get", side_effect=requests.exceptions.RequestException("Download failed")):
            with pytest.raises(Exception) as excinfo:
                openai_llm.generate_image("Test prompt", output_path=output_path)
            assert "download failed" in str(excinfo.value).lower()