"""
Unit tests for the AWS Bedrock provider implementation.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.bedrock import BedrockLLM


@pytest.fixture
def bedrock_llm():
    """Fixture for Bedrock LLM instance."""
    # Patch both auth and boto3 client
    with patch.object(BedrockLLM, "auth") as mock_auth, patch(
        "boto3.client"
    ) as mock_boto3:
        # Mock the auth method to avoid actual AWS calls
        mock_auth.return_value = None

        # Create the client with a valid model
        llm = BedrockLLM(
            model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
        )

        # Create mock client directly
        llm.bedrock_client = MagicMock()

        # Ensure config exists
        if not hasattr(llm, "config"):
            llm.config = {}

        # Add client to config as expected by implementation
        llm.config["client"] = llm.bedrock_client

        yield llm


def test_bedrock_initialization(bedrock_llm):
    """Test Bedrock provider initialization."""
    # Make sure model lists are properly populated
    assert len(bedrock_llm.SUPPORTED_MODELS) > 0
    assert bedrock_llm.model_name in bedrock_llm.SUPPORTED_MODELS

    # Check that model settings are properly configured
    assert bedrock_llm.model_name in bedrock_llm.CONTEXT_WINDOW
    assert bedrock_llm.model_name in bedrock_llm.COST_PER_MODEL


def test_message_formatting(bedrock_llm):
    """Test Bedrock message formatting."""
    # Test basic single message formatting
    messages = [
        {
            "message_type": "human",
            "message": "test message",
            "image_paths": [],
            "image_urls": [],
        }
    ]

    formatted = bedrock_llm._format_messages_for_model(messages)

    # Verify basic structure for anthropic models
    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert isinstance(formatted[0]["content"], list)

    # Verify the text content is directly in the content list
    assert len(formatted[0]["content"]) == 1
    assert "text" in formatted[0]["content"][0]
    assert formatted[0]["content"][0]["text"] == "test message"


def test_get_supported_models(bedrock_llm):
    """Test that we can get supported models list."""
    models = bedrock_llm.get_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "us.anthropic.claude-3-5-sonnet-20241022-v2:0" in models


def test_validate_model(bedrock_llm):
    """Test model validation works correctly."""
    # Test with valid model
    assert (
        bedrock_llm.validate_model("us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        is True
    )

    # Test with invalid model
    assert bedrock_llm.validate_model("invalid-model-name") is False


def test_supports_image_input(bedrock_llm):
    """Test Bedrock image support flag."""
    # The Claude 3.5 models support images
    assert bedrock_llm.supports_image_input() is True
