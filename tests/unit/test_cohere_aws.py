"""
Isolated test for Cohere provider AWS integration.

This separate test file avoids import and patching issues that might occur in
the main test_cohere_provider.py file.
"""

import os
from unittest.mock import patch

from lluminary.models.providers.cohere import CohereLLM


def test_get_api_key_from_aws_direct():
    """Test the _get_api_key_from_aws method with direct mocking."""
    # Create a test instance
    llm = CohereLLM("command")
    llm.config = {"aws_profile": "test-profile", "aws_region": "us-west-2"}

    # Mock the get_secret function directly
    with patch("lluminary.utils.aws.get_secret") as mock_get_secret:
        # Configure the mock
        mock_get_secret.return_value = {"api_key": "test-api-key"}

        # Call the method
        result = llm._get_api_key_from_aws("test-secret")

        # Verify the result
        assert result == "test-api-key"

        # Verify get_secret was called correctly
        mock_get_secret.assert_called_once()
        args, kwargs = mock_get_secret.call_args
        assert kwargs["secret_id"] == "test-secret"
        assert kwargs["required_keys"] == ["api_key"]
        assert kwargs["aws_profile"] == "test-profile"
        assert kwargs["aws_region"] == "us-west-2"


def test_auth_with_aws_secret_integration():
    """Test the complete authentication flow with AWS secret."""
    # Create a test instance
    llm = CohereLLM("command")
    llm.config = {"aws_secret_name": "test-secret"}

    # Clear environment variables to ensure AWS is used
    with patch.dict(os.environ, {}, clear=True):
        # Mock the _get_api_key_from_aws method
        with patch.object(CohereLLM, "_get_api_key_from_aws") as mock_get_key:
            # Configure the mock
            mock_get_key.return_value = "test-api-key-from-aws"

            # Call auth method
            llm.auth()

            # Verify API key was set
            assert llm.api_key == "test-api-key-from-aws"
            mock_get_key.assert_called_once()

            # Verify HTTP session was configured
            assert (
                llm.session.headers["Authorization"] == "Bearer test-api-key-from-aws"
            )
