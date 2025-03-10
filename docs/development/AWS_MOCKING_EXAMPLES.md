# AWS Mocking Examples for Testing

This guide provides examples of proper AWS mocking techniques for testing AWS-dependent components in the LLuminary library, with a focus on Bedrock provider.

## Basic AWS Mocking Pattern

The basic pattern for mocking AWS services in tests involves:

1. Patching the boto3 Session constructor
2. Creating mock clients for AWS services
3. Configuring the mock clients to return appropriate responses
4. Verifying the correct behavior with assertions

## Authentication Mocking Examples

### Mocking AWS Session for Authentication Tests

```python
import boto3
import pytest
from unittest.mock import MagicMock, patch
from lluminary.models.providers.bedrock import BedrockLLM

@pytest.fixture
def mock_aws_session():
    """Create a mock AWS session with configurable clients."""
    with patch("boto3.session.Session") as mock_session:
        # Create a mock session instance
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        # Create a mock bedrock client
        mock_bedrock_client = MagicMock()
        mock_session_instance.client.return_value = mock_bedrock_client

        # Return both the session mock and client mock for configuration
        yield mock_session, mock_bedrock_client
```

### Testing Successful Authentication

```python
def test_successful_authentication(mock_aws_session):
    """Test successful authentication with AWS credentials."""
    mock_session, mock_client = mock_aws_session

    # Create a BedrockLLM instance with test credentials
    llm = BedrockLLM(
        model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        aws_profile="test-profile"
    )

    # Verify the AWS session was created with correct parameters
    mock_session.assert_called_once_with(
        profile_name="test-profile",
        region_name="us-east-1"  # Default region
    )

    # Verify client was created for the right service
    mock_session.return_value.client.assert_called_once_with(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        config=pytest.ANY  # Use pytest.ANY for config since it's complex
    )

    # Verify service property was set
    assert llm.service == "bedrock"
```

### Testing Authentication Failures

```python
def test_authentication_failure_with_invalid_profile():
    """Test authentication failure with invalid AWS profile."""
    with patch("boto3.session.Session") as mock_session:
        # Configure session to raise ProfileNotFound
        mock_session.side_effect = botocore.exceptions.ProfileNotFound(
            profile="nonexistent-profile"
        )

        # Attempt to create a BedrockLLM instance
        with pytest.raises(LLMAuthenticationError) as exc_info:
            BedrockLLM(
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
                aws_profile="nonexistent-profile"
            )

        # Verify exception details
        exception = exc_info.value
        assert exception.provider == "bedrock"
        assert "profile" in str(exception).lower()
        assert "nonexistent-profile" in str(exception).lower()
```

## API Error Mocking Examples

### Creating Mock ClientError Responses

When testing AWS API errors, you need to create properly structured error responses:

```python
def create_aws_client_error(code, message):
    """Create a properly formatted error response for boto3 ClientError."""
    return {
        "Error": {
            "Code": code,
            "Message": message
        },
        "ResponseMetadata": {
            "RequestId": "test-request-id"
        }
    }

# Example usage:
error_response = create_aws_client_error("ThrottlingException", "Rate exceeded")
client_error = ClientError(error_response=error_response, operation_name="invoke_model")
```

### Testing Rate Limiting

```python
def test_rate_limiting_handling():
    """Test handling of rate limiting errors."""
    with patch.object(BedrockLLM, "auth"):
        # Create a BedrockLLM instance without authentication
        llm = BedrockLLM(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            auto_auth=False
        )
        llm.service = "bedrock"

        # Create a mock client
        mock_client = MagicMock()
        llm.client = mock_client

        # Configure the client to raise a ThrottlingException
        error_response = create_aws_client_error(
            "ThrottlingException", "Rate exceeded"
        )
        mock_client.invoke_model.side_effect = ClientError(
            error_response=error_response,
            operation_name="invoke_model"
        )

        # Attempt to generate a response
        with pytest.raises(LLMRateLimitError) as exc_info:
            llm._raw_generate(
                event_id="test-event",
                system_prompt="You are a helpful assistant.",
                messages=[{"message_type": "human", "message": "Hello"}]
            )

        # Verify exception details
        exception = exc_info.value
        assert exception.provider == "bedrock"
        assert "rate" in str(exception).lower()
        assert "error" in exception.details
```

## Mocking Successful API Responses

### Mocking invoke_model Responses

```python
import json
from io import BytesIO

def test_successful_generation():
    """Test successful text generation."""
    with patch.object(BedrockLLM, "auth"):
        # Create a BedrockLLM instance without authentication
        llm = BedrockLLM(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            auto_auth=False
        )
        llm.service = "bedrock"

        # Create a mock client
        mock_client = MagicMock()
        llm.client = mock_client

        # Create a mock response
        mock_response = {
            "body": BytesIO(json.dumps({
                "id": "resp-123",
                "content": [{"type": "text", "text": "This is a test response"}],
                "usage": {"input_tokens": 10, "output_tokens": 20},
                "stop_reason": "stop"
            }).encode())
        }
        mock_client.invoke_model.return_value = mock_response

        # Call the generate method
        result, metadata = llm._raw_generate(
            event_id="test-event",
            system_prompt="You are a helpful assistant.",
            messages=[{"message_type": "human", "message": "Hello"}]
        )

        # Verify the result
        assert result == "This is a test response"
        assert metadata["input_tokens"] == 10
        assert metadata["output_tokens"] == 20
        assert metadata["total_tokens"] == 30  # Computed value
```

## Mocking AWS Secrets Manager

### Basic Secrets Manager Mocking

```python
import json
from unittest.mock import patch, MagicMock

def test_get_api_key_from_aws():
    """Test retrieving API key from AWS Secrets Manager."""
    with patch("boto3.session.Session") as mock_session:
        # Create mock session and client
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        mock_client = MagicMock()
        mock_session_instance.client.return_value = mock_client

        # Configure get_secret_value response
        mock_client.get_secret_value.return_value = {
            'SecretString': json.dumps({"api_key": "test-api-key"})
        }

        # Call get_secret function
        from lluminary.utils.aws import get_secret
        result = get_secret("test-secret", aws_profile="test-profile")

        # Verify the result
        assert result == {"api_key": "test-api-key"}

        # Verify session was created with correct profile
        mock_session.assert_called_once_with(
            profile_name="test-profile",
            region_name=None
        )

        # Verify client was created for Secrets Manager
        mock_session_instance.client.assert_called_once_with(
            service_name="secretsmanager"
        )

        # Verify get_secret_value was called with correct parameters
        mock_client.get_secret_value.assert_called_once_with(
            SecretId="test-secret"
        )
```

## Complex Scenarios

### Testing Retry Mechanism with Exponential Backoff

```python
import time
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError

def test_exponential_backoff_retry():
    """Test exponential backoff retry mechanism."""
    with patch.object(BedrockLLM, "auth"):
        # Create a BedrockLLM instance without authentication
        llm = BedrockLLM(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            auto_auth=False
        )
        llm.service = "bedrock"

        # Create mock function that initially fails with throttling errors
        mock_func = MagicMock()
        throttling_error = {
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Rate exceeded"
            },
            "ResponseMetadata": {
                "RequestId": "test-request-id"
            }
        }

        # Configure to fail twice then succeed
        mock_func.side_effect = [
            ClientError(error_response=throttling_error, operation_name="invoke_model"),
            ClientError(error_response=throttling_error, operation_name="invoke_model"),
            "Success!",  # Third call succeeds
        ]

        # Record start time
        start_time = time.time()

        # Call with retry with exponential backoff
        result = llm._call_with_retry(
            mock_func,
            retryable_errors=[ClientError],
            max_retries=5,
            base_delay=0.1  # Small for test but allows measurement
        )

        # Calculate elapsed time
        elapsed = time.time() - start_time

        # Verify exponential backoff occurred (at least 0.1 + 0.2 delay)
        assert elapsed >= 0.3
        assert result == "Success!"
        assert mock_func.call_count == 3
```

### Testing STS Role Assumption

```python
def test_assume_role_functionality():
    """Test assuming an IAM role for authentication."""
    with patch("boto3.session.Session") as mock_session:
        # Create mock session and clients
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        # Mock STS client
        mock_sts_client = MagicMock()

        # Mock Bedrock client
        mock_bedrock_client = MagicMock()

        # Configure client factory to return appropriate client
        def get_client(service_name, **kwargs):
            if service_name == "sts":
                return mock_sts_client
            elif service_name == "bedrock-runtime":
                return mock_bedrock_client
            else:
                return MagicMock()

        mock_session_instance.client.side_effect = get_client

        # Configure assume_role response
        mock_sts_client.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "ASIAIOSFODNN7EXAMPLE",
                "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "SessionToken": "FwoGZXIvYXdzEHkaDHVV6n5Xw1moAxPu3SLrASDDG4GiJ4hrgbp/G2s=",
                "Expiration": "2025-01-01T00:00:00Z"
            }
        }

        # Create a BedrockLLM instance with role assumption
        llm = BedrockLLM(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            aws_profile="base-profile",
            assume_role_arn="arn:aws:iam::123456789012:role/service-role/test-role"
        )

        # Verify assume_role was called with correct parameters
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::123456789012:role/service-role/test-role",
            RoleSessionName="BedrockLLMSession"
        )

        # Verify a new session was created with the temporary credentials
        assert mock_session.call_count == 2  # Initial session + session with assumed role
```

## Best Practices

1. **Use fixtures for common mocking setup** to reduce duplication and improve readability
2. **Verify error handling explicitly** by testing exception types, messages, and details
3. **Capture all possible error conditions** from AWS services
4. **Test retry mechanisms** with various error types and retry counts
5. **Verify credential handling** for all authentication methods (profile, direct credentials, role assumption)
6. **Test region configuration** and fallback behavior
7. **Use proper typing** in your mocks to help with type checking
8. **Implement cross-provider consistency tests** to ensure consistent behavior across all AWS-based providers

By following these patterns, you can ensure robust testing of AWS-dependent components without relying on actual AWS infrastructure.
