"""
Unit tests for the authentication functionality.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from lluminary.models.utils.aws import get_secret


class TestAuthentication:
    """Tests for the authentication system."""

    def test_get_secret_from_env_vars(self, mock_env_vars):
        """Test retrieving API keys from environment variables."""
        # Test OpenAI secret
        secret = get_secret("openai_api_key", required_keys=["api_key"])
        assert secret["api_key"] == "test-openai-key"

        # Test Anthropic secret
        secret = get_secret("anthropic_api_key", required_keys=["api_key"])
        assert secret["api_key"] == "test-anthropic-key"

        # Test Google secret
        secret = get_secret("google_api_key", required_keys=["api_key"])
        assert secret["api_key"] == "test-google-key"

    @patch("boto3.session.Session")
    def test_get_secret_from_aws_fallback(self, mock_session):
        """Test fallback to AWS Secrets Manager when environment variables are not set."""
        # Mock AWS Secrets Manager response
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        mock_response = {
            "SecretString": json.dumps({"api_key": "test-secret-from-aws"})
        }
        mock_client.get_secret_value.return_value = mock_response

        # Clear environment variable to force AWS fallback
        with patch.dict(os.environ, {"OPENAI_API_KEY_API_KEY": ""}, clear=True):
            secret = get_secret("openai_api_key", required_keys=["api_key"])

            # Verify the secret was fetched from AWS
            assert secret["api_key"] == "test-secret-from-aws"
            mock_client.get_secret_value.assert_called_once_with(
                SecretId="openai_api_key"
            )

    def test_get_secret_missing_required_key(self, mock_env_vars):
        """Test exception raised when required key is missing."""
        # Test with environment variable missing required key
        # First clear the environment variable to avoid getting it from env
        with patch.dict(os.environ, {"OPENAI_API_KEY_API_KEY": ""}, clear=True), patch(
            "boto3.session.Session"
        ) as mock_session:
            # Mock AWS response with missing key
            mock_client = MagicMock()
            mock_session.return_value.client.return_value = mock_client

            mock_response = {
                "SecretString": json.dumps(
                    {
                        "api_key": "test-secret-from-aws"
                        # missing_key is not included
                    }
                )
            }
            mock_client.get_secret_value.return_value = mock_response

            with pytest.raises(Exception) as excinfo:
                get_secret("openai_api_key", required_keys=["api_key", "missing_key"])

            assert "is missing required keys" in str(excinfo.value)
