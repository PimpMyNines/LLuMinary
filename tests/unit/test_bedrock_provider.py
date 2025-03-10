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


def test_auth_error_handling():
    """Test authentication error handling."""
    from lluminary.exceptions import AuthenticationError

    # Test auth error handling by forcing an exception
    with patch("boto3.session.Session") as mock_session:
        # Set up mock to raise exception
        mock_session.side_effect = Exception("AWS credentials not found")

        # Create a new LLM instance that will use our mocked boto3
        llm = BedrockLLM(model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0")

        # Call auth directly and verify it raises the right exception
        with pytest.raises(AuthenticationError) as exc:
            llm.auth()

        # Verify exception details
        exception = exc.value
        assert isinstance(exception, AuthenticationError)
        assert "Bedrock authentication failed" in str(exception)
        assert exception.provider == "BedrockLLM"
        assert "original_error" in exception.details


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


def test_supported_model_lists(bedrock_llm):
    """Test that the model lists are properly configured."""
    # Make sure appropriate lists are populated
    assert len(bedrock_llm.SUPPORTED_MODELS) > 0
    assert len(bedrock_llm.THINKING_MODELS) > 0

    # Check the model lists contain appropriate entries
    assert "us.anthropic.claude-3-7-sonnet-20250219-v1:0" in bedrock_llm.THINKING_MODELS
    assert (
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0" in bedrock_llm.SUPPORTED_MODELS
    )

    # Verify that cost and context window data is properly configured for each model
    for model_name in bedrock_llm.SUPPORTED_MODELS:
        assert model_name in bedrock_llm.CONTEXT_WINDOW
        assert model_name in bedrock_llm.COST_PER_MODEL

        # Verify cost structure is correct
        model_costs = bedrock_llm.COST_PER_MODEL[model_name]
        assert "read_token" in model_costs
        assert "write_token" in model_costs
        assert "image_cost" in model_costs

        # Verify values are of the expected type
        assert isinstance(model_costs["read_token"], (int, float))
        assert isinstance(model_costs["write_token"], (int, float))
        assert isinstance(model_costs["image_cost"], (int, float))

        # Verify context window is a number
        assert isinstance(bedrock_llm.CONTEXT_WINDOW[model_name], int)


def test_error_handling(bedrock_llm):
    """Test basic error handling."""

    # Test validation error when trying to use an unsupported model
    unsupported_model = "invalid-model-name"
    with pytest.raises(ValueError) as excinfo:
        # This should fail validation
        BedrockLLM(
            model_name=unsupported_model,
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
        )

    # Verify error message contains useful information
    assert unsupported_model in str(excinfo.value)
    assert "not supported" in str(excinfo.value).lower()

    # Test configuration error
    with pytest.raises(LLMMistake):
        # Remove the client from config to cause an error
        bedrock_llm.config.pop("client", None)

        # This should fail since client is not configured
        bedrock_llm.generate(
            event_id="test",
            system_prompt="You are a helpful assistant",
            messages=[{"message_type": "human", "message": "Hello"}],
        )


def test_boto_client_errors(bedrock_llm):
    """Test AWS boto client error handling."""
    from botocore.exceptions import ClientError
    from lluminary.exceptions import (
        AuthenticationError,
        RateLimitError,
        ServiceUnavailableError,
    )

    # Create a mock response for boto3 client error
    def create_client_error(code, message):
        return ClientError(
            error_response={"Error": {"Code": code, "Message": message}},
            operation_name="converse",
        )

    # Define error test cases
    error_test_cases = [
        # Rate limit errors
        {
            "error": create_client_error("ThrottlingException", "Request throttled"),
            "expected_exception": RateLimitError,
            "expected_message": "rate limit exceeded",
        },
        {
            "error": create_client_error(
                "TooManyRequestsException", "Too many requests"
            ),
            "expected_exception": RateLimitError,
            "expected_message": "rate limit exceeded",
        },
        # Authentication errors
        {
            "error": create_client_error("AccessDeniedException", "Access denied"),
            "expected_exception": AuthenticationError,
            "expected_message": "authentication failed",
        },
        # Service errors
        {
            "error": create_client_error(
                "ServiceUnavailableException", "Service unavailable"
            ),
            "expected_exception": ServiceUnavailableError,
            "expected_message": "service unavailable",
        },
        # Generic API errors
        {
            "error": create_client_error("ValidationException", "Invalid request"),
            "expected_exception": LLMMistake,
            "expected_message": "Bedrock API error",
        },
    ]

    # Replace the mock client with one that raises errors
    for test_case in error_test_cases:
        # Configure the mock to raise our error
        bedrock_llm.config["client"] = MagicMock()
        bedrock_llm.config["client"].converse.side_effect = test_case["error"]

        # Make the API call and check for the expected exception
        with pytest.raises(test_case["expected_exception"]) as exc:
            bedrock_llm._raw_generate(
                event_id="test",
                system_prompt="You are a helpful assistant",
                messages=[{"message_type": "human", "message": "Test error handling"}],
                max_tokens=100,
                temp=0.7,
                top_k=40,
            )

        # Check exception properties
        exception = exc.value
        assert test_case["expected_message"] in str(exception).lower()
        assert exception.provider == "BedrockLLM"

        # Check specific exception properties
        if isinstance(exception, RateLimitError):
            assert exception.retry_after is not None
            assert exception.retry_after > 0

        if isinstance(exception, LLMMistake):
            assert exception.error_type == "api_error"

        assert "details" in dir(exception)
        assert "original_error" in exception.details


def test_image_processing_errors():
    """Test image processing error handling."""

    # Create a fresh instance with mocked dependencies
    with patch.object(BedrockLLM, "auth"):
        llm = BedrockLLM(model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    # Test file not found error
    with patch("open", side_effect=FileNotFoundError("File not found")):
        with pytest.raises(LLMMistake) as exc:
            llm._get_image_bytes("/path/to/nonexistent.jpg")

        exception = exc.value
        assert isinstance(exception, LLMMistake)
        assert "Failed to read image" in str(exception)
        assert exception.error_type == "image_processing_error"
        assert exception.provider == "BedrockLLM"
        assert "path" in exception.details

    # Test general image processing error
    with patch("open") as mock_open, patch(
        "PIL.Image.open", side_effect=Exception("Invalid image format")
    ):
        # Mock the file open but fail on image processing
        mock_open.return_value.__enter__.return_value = MagicMock()

        with pytest.raises(LLMMistake) as exc:
            llm._get_image_bytes("/path/to/corrupted.jpg")

        exception = exc.value
        assert isinstance(exception, LLMMistake)
        assert "Failed to process image" in str(exception)
        assert exception.error_type == "image_processing_error"

    # Test URL download error
    with patch("requests.get", side_effect=Exception("Failed to download")):
        with pytest.raises(LLMMistake) as exc:
            llm._download_image_from_url("https://example.com/image.jpg")

        exception = exc.value
        assert isinstance(exception, LLMMistake)
        assert "Failed to download image" in str(exception)
        assert exception.error_type == "image_url_error"
        assert "url" in exception.details


def test_tool_formatting(bedrock_llm):
    """Test Bedrock tool formatting."""
    # Create mock tools in format that would be passed to the function
    tools = [
        {
            "name": "test_tool1",
            "description": "Multiply x by the length of y",
            "input_schema": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "string"},
                },
                "required": ["x", "y"],
            },
        },
        {
            "name": "test_tool2",
            "description": "Create a user profile with optional fields",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "active": {"type": "boolean"},
                },
                "required": ["name"],
            },
        },
    ]

    formatted = bedrock_llm._format_tools_for_model(tools)

    # Check basic tool structure - should use Bedrock's toolSpec format
    assert "tools" in formatted
    assert len(formatted["tools"]) == 2

    # Check first tool
    assert "toolSpec" in formatted["tools"][0]
    assert formatted["tools"][0]["toolSpec"]["name"] == "test_tool1"
    assert (
        formatted["tools"][0]["toolSpec"]["description"]
        == "Multiply x by the length of y"
    )
    assert "inputSchema" in formatted["tools"][0]["toolSpec"]
    assert "json" in formatted["tools"][0]["toolSpec"]["inputSchema"]

    # Check first tool schema
    schema_json = formatted["tools"][0]["toolSpec"]["inputSchema"]["json"]
    assert schema_json["type"] == "object"
    assert "x" in schema_json["properties"]
    assert "y" in schema_json["properties"]
    assert schema_json["properties"]["x"]["type"] == "integer"
    assert schema_json["properties"]["y"]["type"] == "string"
    assert "x" in schema_json["required"]
    assert "y" in schema_json["required"]

    # Check second tool with optional parameters
    assert formatted["tools"][1]["toolSpec"]["name"] == "test_tool2"

    # Check second tool schema
    schema_json = formatted["tools"][1]["toolSpec"]["inputSchema"]["json"]
    assert "name" in schema_json["properties"]
    assert "age" in schema_json["properties"]
    assert "active" in schema_json["properties"]
    assert schema_json["properties"]["active"]["type"] == "boolean"
    assert "name" in schema_json["required"]
    assert "age" not in schema_json["required"]  # Should be optional
    assert "active" not in schema_json["required"]  # Should be optional
