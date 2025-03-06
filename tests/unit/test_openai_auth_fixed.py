"""
Fixed tests for OpenAI provider authentication functionality.

This module focuses specifically on testing the authentication
mechanisms of the OpenAI provider in isolation, with a different patching approach.
"""
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from src.llmhandler.models.providers.openai import OpenAILLM


class TestOpenAIAuth:
    """Test class for OpenAI authentication functionality."""

    def test_auth_with_aws_secrets(self):
        """Test authentication using AWS Secrets Manager."""
        # Create mock for get_secret function
        mock_secret = {"api_key": "test-secret-key"}
        
        # Mock both get_secret and OpenAI class using context manager
        with patch("src.llmhandler.models.providers.openai.get_secret", return_value=mock_secret):
            # Mock the OpenAI class constructor
            openai_mock = MagicMock()
            
            # Patch the OpenAI class
            with patch.dict("sys.modules", {"openai": MagicMock()}):
                with patch("src.llmhandler.models.providers.openai.OpenAI", return_value=openai_mock):
                    # Create instance and call auth
                    openai_llm = OpenAILLM("gpt-4o")
                    openai_llm.auth()
                    
                    # Verify API key was properly stored
                    assert openai_llm.config["api_key"] == "test-secret-key"

    def test_auth_with_environment_variables(self):
        """Test authentication using environment variables instead of AWS Secrets Manager."""
        # Original environment
        original_env = os.environ.copy()
        
        try:
            # Set up test environment
            os.environ["OPENAI_API_KEY"] = "env-api-key"
            
            # Mock get_secret to raise an exception
            with patch("src.llmhandler.models.providers.openai.get_secret", 
                      side_effect=Exception("Secret not found")):
                # Mock the OpenAI class constructor
                openai_mock = MagicMock()
                
                # Patch the OpenAI class
                with patch.dict("sys.modules", {"openai": MagicMock()}):
                    with patch("src.llmhandler.models.providers.openai.OpenAI", return_value=openai_mock):
                        # Create instance and call auth
                        openai_llm = OpenAILLM("gpt-4o")
                        openai_llm.auth()
                        
                        # Verify API key was properly stored from environment variable
                        assert openai_llm.config["api_key"] == "env-api-key"
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_auth_failure_handling(self):
        """Test handling of authentication failures."""
        # Test different error scenarios
        error_messages = [
            "Secret not found",
            "Access denied",
            "Invalid parameters",
            "Network error"
        ]
        
        for error_msg in error_messages:
            # Original environment
            original_env = os.environ.copy()
            
            try:
                # Clear OPENAI_API_KEY from environment
                if "OPENAI_API_KEY" in os.environ:
                    del os.environ["OPENAI_API_KEY"]
                
                # Mock get_secret to raise exception
                with patch("src.llmhandler.models.providers.openai.get_secret", 
                          side_effect=Exception(error_msg)):
                    
                    # Create instance
                    openai_llm = OpenAILLM("gpt-4o")
                    
                    # Call auth and expect exception
                    with pytest.raises(Exception) as excinfo:
                        openai_llm.auth()
                    
                    # Verify error message
                    assert "OpenAI authentication failed" in str(excinfo.value)
                    assert error_msg in str(excinfo.value)
            finally:
                # Restore original environment
                os.environ.clear()
                os.environ.update(original_env)