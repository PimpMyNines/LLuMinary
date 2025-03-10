"""
Tests for AWS utility functions.

This module tests the AWS utility functions in src/lluminary/utils/aws.py.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_boto3_session():
    """Create mock boto3 session."""
    mock_session = MagicMock()
    mock_client = MagicMock()
    mock_session.client.return_value = mock_client

    # Configure the mock client to return a valid secret
    mock_secret_response = {
        "SecretString": json.dumps({"api_key": "test-api-key-value"})
    }
    mock_client.get_secret_value.return_value = mock_secret_response

    return mock_session


@pytest.fixture
def mock_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TEST_API_KEY", "test-env-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")


class TestAWSUtils:
    """Test AWS utility functions."""

    def test_get_aws_session(self):
        """Test get_aws_session function."""
        from src.lluminary.utils.aws import get_aws_session

        with patch("boto3.session.Session") as mock_session:
            # Test with default parameters
            get_aws_session()
            mock_session.assert_called_once_with()
            mock_session.reset_mock()

            # Test with profile name
            get_aws_session(profile_name="test-profile")
            mock_session.assert_called_once_with(profile_name="test-profile")
            mock_session.reset_mock()

            # Test with region name
            get_aws_session(region_name="us-west-2")
            mock_session.assert_called_once_with(region_name="us-west-2")
            mock_session.reset_mock()

            # Test with both parameters
            get_aws_session(profile_name="test-profile", region_name="us-west-2")
            mock_session.assert_called_once_with(
                profile_name="test-profile", region_name="us-west-2"
            )

    def test_get_secret(self, mock_boto3_session):
        """Test get_secret function."""
        from src.lluminary.utils.aws import get_secret

        with patch("src.lluminary.utils.aws.get_aws_session") as mock_get_session:
            mock_get_session.return_value = mock_boto3_session

            # Test basic functionality
            result = get_secret("test-secret", required_keys=["api_key"])
            assert "api_key" in result
            assert result["api_key"] == "test-api-key-value"

            # Test with profile and region
            result = get_secret(
                "test-secret",
                required_keys=["api_key"],
                aws_profile="test-profile",
                aws_region="us-west-2",
            )
            assert "api_key" in result

            # Verify session was created with correct parameters
            mock_get_session.assert_called_with(
                profile_name="test-profile", region_name="us-west-2"
            )

    def test_get_secret_from_environment(self, mock_environment):
        """Test getting secrets from environment variables."""
        from src.lluminary.utils.aws import get_secret

        # Test with environment variable
        result = get_secret("test_api_key", required_keys=["api_key"])
        assert "api_key" in result
        assert result["api_key"] == "test-env-api-key"

    def test_get_api_key_from_config(self, mock_environment):
        """Test get_api_key_from_config function."""
        from src.lluminary.utils.aws import get_api_key_from_config

        # Test with api_key in config
        config = {"api_key": "config-api-key"}
        result = get_api_key_from_config(config, "test")
        assert result == "config-api-key"

        # Test with environment variable
        config = {}
        result = get_api_key_from_config(config, "openai")
        assert result == "openai-test-key"

        # Test with custom environment key
        config = {}
        result = get_api_key_from_config(config, "test", env_key="TEST_API_KEY")
        assert result == "test-env-api-key"

    def test_aws_secret_retrieval(self, mock_environment):
        """Test getting API key from AWS Secrets Manager."""
        from src.lluminary.utils.aws import get_api_key_from_config

        with patch(
            "src.lluminary.utils.aws.get_secret", autospec=True
        ) as mock_get_secret:
            # Configure mock to return a valid secret
            mock_get_secret.return_value = {"api_key": "aws-api-key"}

            # Test with AWS secret name in config
            config = {"aws_secret_name": "test-secret"}
            result = get_api_key_from_config(config, "test")
            assert result == "aws-api-key"

            # Verify it was called with the right parameters
            mock_get_secret.assert_called_once()
            args, kwargs = mock_get_secret.call_args
            assert kwargs["secret_id"] == "test-secret"
            assert kwargs["required_keys"] == ["api_key"]
            assert kwargs.get("aws_profile") is None
            assert kwargs.get("aws_region") is None

            # Reset mock
            mock_get_secret.reset_mock()

            # Test with AWS profile and region
            config = {
                "aws_secret_name": "test-secret",
                "aws_profile": "test-profile",
                "aws_region": "us-west-2",
            }
            result = get_api_key_from_config(config, "test")
            assert result == "aws-api-key"

            # Verify it was called with the right parameters
            mock_get_secret.assert_called_once()
            args, kwargs = mock_get_secret.call_args
            assert kwargs["secret_id"] == "test-secret"
            assert kwargs["required_keys"] == ["api_key"]
            assert kwargs["aws_profile"] == "test-profile"
            assert kwargs["aws_region"] == "us-west-2"

    def test_profile_name_compatibility(self):
        """Test compatibility between profile_name and aws_profile."""
        from src.lluminary.utils.aws import get_api_key_from_config

        with patch("src.lluminary.utils.aws.get_secret") as mock_get_secret:
            mock_get_secret.return_value = {"api_key": "aws-api-key"}

            # Test with profile_name (bedrock style)
            config = {
                "aws_secret_name": "test-secret",
                "profile_name": "bedrock-profile",
            }
            get_api_key_from_config(config, "test")
            mock_get_secret.assert_called_with(
                secret_id="test-secret",
                required_keys=["api_key"],
                aws_profile="bedrock-profile",
                aws_region=None,
            )

            # Reset mock
            mock_get_secret.reset_mock()

            # Test with aws_profile (cohere style)
            config = {"aws_secret_name": "test-secret", "aws_profile": "cohere-profile"}
            get_api_key_from_config(config, "test")
            mock_get_secret.assert_called_with(
                secret_id="test-secret",
                required_keys=["api_key"],
                aws_profile="cohere-profile",
                aws_region=None,
            )

    def test_aws_secret_retrieval_error_conditions(self):
        """Test error conditions in AWS secret retrieval."""
        from src.lluminary.utils.aws import get_api_key_from_config

        # Test when get_secret raises an exception but environment variable is available
        with patch("src.lluminary.utils.aws.get_secret") as mock_get_secret, patch.dict(
            os.environ, {"TEST_API_KEY": "fallback-env-key"}
        ):

            # Make get_secret raise an exception
            mock_get_secret.side_effect = Exception("AWS Secrets Manager error")

            # Config with aws_secret_name but should fall back to env var
            config = {"aws_secret_name": "test-secret"}
            result = get_api_key_from_config(config, "test")

            # Should have tried AWS but fallen back to env var
            assert result == "fallback-env-key"
            mock_get_secret.assert_called_once()

        # Test when both AWS and environment variables are missing
        with patch("src.lluminary.utils.aws.get_secret") as mock_get_secret, patch.dict(
            os.environ, {}, clear=True
        ):

            # Make get_secret raise an exception
            mock_get_secret.side_effect = Exception("AWS Secrets Manager error")

            # Config with aws_secret_name but no fallback
            config = {"aws_secret_name": "test-secret"}

            # Should raise exception
            with pytest.raises(Exception) as exc_info:
                get_api_key_from_config(config, "test")

            assert "API key for test not found" in str(exc_info.value)
            mock_get_secret.assert_called_once()

    def test_get_aws_session(self):
        """Test the get_aws_session function with various parameters."""
        from src.lluminary.utils.aws import get_aws_session

        with patch("boto3.session.Session") as mock_session:
            # Test with default parameters (no profile or region)
            session = get_aws_session()
            mock_session.assert_called_once_with(profile_name=None, region_name=None)
            mock_session.reset_mock()

            # Test with profile only
            session = get_aws_session(profile_name="test-profile")
            mock_session.assert_called_once_with(
                profile_name="test-profile", region_name=None
            )
            mock_session.reset_mock()

            # Test with region only
            session = get_aws_session(region_name="us-west-2")
            mock_session.assert_called_once_with(
                profile_name=None, region_name="us-west-2"
            )
            mock_session.reset_mock()

            # Test with both profile and region
            session = get_aws_session(
                profile_name="test-profile", region_name="us-west-2"
            )
            mock_session.assert_called_once_with(
                profile_name="test-profile", region_name="us-west-2"
            )

    def test_aws_secret_with_provider_integration(self):
        """Test integration with provider classes using AWS secrets."""
        from src.lluminary.utils.aws import get_api_key_from_config

        # First test an implementation that works directly with our utils
        config = {
            "aws_secret_name": "test-secret",
            "aws_profile": "test-profile",
            "aws_region": "us-west-2",
        }

        # Mock both get_secret to isolate the test
        with patch("src.lluminary.utils.aws.get_secret") as mock_get_secret:
            mock_get_secret.return_value = {"api_key": "test-api-key"}

            # Call the function we exported
            result = get_api_key_from_config(config, "test")

            # Verify results
            assert result == "test-api-key"
            mock_get_secret.assert_called_once()

            # Verify parameters were correctly passed
            args, kwargs = mock_get_secret.call_args
            assert kwargs["secret_id"] == "test-secret"
            assert kwargs["required_keys"] == ["api_key"]
            assert kwargs["aws_profile"] == "test-profile"
            assert kwargs["aws_region"] == "us-west-2"
