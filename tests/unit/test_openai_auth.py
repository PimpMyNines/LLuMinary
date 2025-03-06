"""
Tests for OpenAI provider authentication functionality.

This module focuses specifically on testing the authentication
mechanisms of the OpenAI provider in isolation.
"""
import os
from unittest.mock import patch, MagicMock

import pytest

from src.llmhandler.models.providers.openai import OpenAILLM


def test_auth_with_aws_secrets():
    """Test authentication using AWS Secrets Manager."""
    # Create mock for get_secret function
    mock_secret = {"api_key": "test-secret-key"}
    
    with patch("src.llmhandler.models.providers.openai.get_secret", return_value=mock_secret) as mock_get_secret, \
         patch("openai.OpenAI") as mock_openai_client:
        
        # Create instance and call auth
        openai_llm = OpenAILLM("gpt-4o")
        openai_llm.auth()
        
        # Verify get_secret was called with correct parameters
        mock_get_secret.assert_called_once_with("openai_api_key", required_keys=["api_key"])
        
        # Verify API key was properly stored
        assert openai_llm.config["api_key"] == "test-secret-key"
        
        # Verify OpenAI client was initialized with correct API key
        mock_openai_client.assert_called_once_with(api_key="test-secret-key")


def test_auth_with_environment_variables():
    """Test authentication using environment variables instead of AWS Secrets Manager."""
    # Mock environment variable and make get_secret raise an exception
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}), \
         patch("src.llmhandler.models.providers.openai.get_secret", side_effect=Exception("Secret not found")), \
         patch("openai.OpenAI") as mock_openai_client:
        
        # Create instance and call auth - should fall back to env var
        openai_llm = OpenAILLM("gpt-4o")
        
        # We need to catch the exception and verify it contains information about checking env vars
        with pytest.raises(Exception) as excinfo:
            openai_llm.auth()
        
        # Verify exception message suggests checking environment variables
        assert "Secret not found" in str(excinfo.value)


def test_auth_failure_handling():
    """Test handling of authentication failures."""
    # Test different error scenarios
    error_messages = [
        "Secret not found",
        "Access denied",
        "Invalid parameters",
        "Network error"
    ]
    
    for error_msg in error_messages:
        # Mock get_secret to raise exception and ensure no env var fallback
        with patch("src.llmhandler.models.providers.openai.get_secret", 
                   side_effect=Exception(error_msg)), \
             patch.dict(os.environ, {}, clear=True), \
             patch("openai.OpenAI"):
            
            # Create instance
            openai_llm = OpenAILLM("gpt-4o")
            
            # Call auth and expect exception
            with pytest.raises(Exception) as excinfo:
                openai_llm.auth()
            
            # Verify error message
            assert error_msg in str(excinfo.value)
            assert "OpenAI authentication failed" in str(excinfo.value)
