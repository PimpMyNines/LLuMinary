"""
Debug file for Bedrock retry mechanism tests.
"""

from unittest.mock import MagicMock, patch

import pytest
from lluminary.models.providers.bedrock import BedrockLLM


@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client with pre-configured responses."""
    client = MagicMock()
    return client


@pytest.fixture
def bedrock_llm(mock_bedrock_client):
    """Fixture for Bedrock LLM instance with mocked dependencies."""
    # Patch needed AWS components
    with patch("boto3.session.Session") as mock_session, patch.object(
        BedrockLLM, "auth"
    ) :

        # Configure session mock
        mock_session_instance = MagicMock()
        mock_session_instance.client.return_value = mock_bedrock_client
        mock_session.return_value = mock_session_instance

        # Create a BedrockLLM instance with standard test parameters
        llm = BedrockLLM(
            model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            auto_auth=False,  # Disable auto auth to control it in tests
            region_name="us-east-1",
            profile_name="ai-dev",  # Include profile name as per best practices
        )

        # Configure the instance with our mock client
        llm.config["runtime_client"] = mock_bedrock_client
        llm.config["bedrock_client"] = mock_bedrock_client

        # Set model lists to ensure consistency in tests
        llm.SUPPORTED_MODELS = [
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-instant-v1",
        ]

        yield llm


def test_retry_function_exists(bedrock_llm):
    """Verify that _call_with_retry exists and is callable."""
    assert hasattr(bedrock_llm, "_call_with_retry")
    assert callable(bedrock_llm._call_with_retry)
    # Print the function signature to check parameters
    import inspect

    print(inspect.signature(bedrock_llm._call_with_retry))


def test_simple_retry(bedrock_llm):
    """Test a very simple retry case."""
    mock_func = MagicMock()
    mock_func.side_effect = [Exception("Error"), "Success"]

    # First test with modified parameters for simplicity
    result = bedrock_llm._call_with_retry(
        mock_func, retryable_errors=[Exception], max_retries=1, base_delay=0.01
    )

    assert mock_func.call_count == 2
    assert result == "Success"
