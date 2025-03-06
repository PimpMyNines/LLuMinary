"""
Unit tests for the Provider Template utility methods.
"""
import base64
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests

from src.llmhandler.models.providers.provider_template import ProviderNameLLM


@pytest.fixture
def provider_instance():
    """Create a basic provider instance for testing."""
    # Mock the auth method to avoid actual authentication
    with patch.object(ProviderNameLLM, "auth", return_value=None):
        provider = ProviderNameLLM(
            model_name="provider-model-2",  # Use model-2 which supports images
            timeout=30
        )
    return provider


class TestProviderTemplateImageProcessing:
    """Test image processing methods of ProviderNameLLM."""
    
    def test_process_image_file(self, provider_instance):
        """Test processing a local image file."""
        # Skip because _process_image_file is not implemented in the template
        pytest.skip("Skip image file processing test as _process_image_file is not implemented in template")
        
        # The actual implementation would look like this:
        # with patch("builtins.open", new_callable=mock_open, read_data=b"image_binary_data"):
        #     with patch("base64.b64encode") as mock_b64encode:
        #         # Configure mock
        #         mock_b64encode.return_value = b"encoded_image_data"
        #         
        #         # Add implementation to the pass method
        #         def process_image_file(path):
        #             with open(path, "rb") as img_file:
        #                 return base64.b64encode(img_file.read()).decode("utf-8")
        #         
        #         provider_instance._process_image_file = process_image_file
        #         
        #         # Call the method
        #         result = provider_instance._process_image_file("/path/to/image.jpg")
        #         
        #         # Verify result
        #         assert result == "encoded_image_data"
    
    def test_process_image_url(self, provider_instance):
        """Test processing an image from URL."""
        # Skip because _process_image_url is not implemented in the template
        pytest.skip("Skip image URL processing test as _process_image_url is not implemented in template")
    
    def test_process_image_url_error(self, provider_instance):
        """Test error handling when processing an image from URL."""
        # Skip because _process_image_url is not implemented in the template
        pytest.skip("Skip image URL error test as _process_image_url is not implemented in template")


class TestProviderTemplateToolHandling:
    """Test tool handling methods of ProviderNameLLM."""
    
    @patch("src.llmhandler.models.providers.provider_template.ProviderNameLLM._format_messages_for_model")
    def test_raw_generate_with_tools(self, mock_format_messages, provider_instance):
        """Test raw_generate with tools parameter."""
        # Set up mocks
        mock_format_messages.return_value = [{"role": "user", "content": "Hello"}]
        
        # Define test tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        # Call the method under test
        result, usage = provider_instance._raw_generate(
            event_id="test-event",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "What's the weather in Seattle?"}],
            tools=tools
        )
        
        # In a real implementation, we would verify that tools were passed correctly
        # to the provider API. For the template, we just verify basic behavior.
        assert result == "This is a placeholder response."
        assert "tool_use" in usage
    
    def test_tool_result_message_formatting(self, provider_instance):
        """Test formatting of tool result messages."""
        messages = [
            {"message_type": "human", "message": "What's the weather in Seattle?"},
            {
                "message_type": "tool_result", 
                "message": '{"temperature": 72, "condition": "sunny"}',
                "tool_name": "get_weather"
            }
        ]
        
        formatted = provider_instance._format_messages_for_model(messages)
        
        assert len(formatted) == 2
        assert formatted[0]["role"] == "user"
        assert formatted[1]["role"] == "tool"
        assert formatted[1]["content"] == '{"temperature": 72, "condition": "sunny"}'