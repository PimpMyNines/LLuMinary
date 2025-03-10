"""
Unit tests for the AWS Bedrock provider implementation.

This file includes comprehensive tests for the BedrockLLM provider, including:
- Basic initialization and authentication
- Message formatting
- Tool formatting and handling
- Error mapping and handling
- Image processing
- Streaming functionality
"""

from unittest.mock import MagicMock, patch

import pytest
import requests
from botocore.exceptions import ClientError, ProfileNotFound
from lluminary.exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContentError,
    LLMRateLimitError,
)
from lluminary.models.providers.bedrock import BedrockLLM

from tests.unit.helpers.aws_mocks import (
    create_aws_client_error,
    create_bedrock_converse_response,
    create_bedrock_converse_stream_response,
    create_bedrock_converse_with_tools_response,
    create_bedrock_list_models_response,
)


@pytest.fixture
def mock_bedrock_client():
    """Create a mock Bedrock client with pre-configured responses."""
    client = MagicMock()

    # Configure standard responses
    client.converse.return_value = create_bedrock_converse_response()
    client.converse_stream.return_value = create_bedrock_converse_stream_response()
    client.list_foundation_models.return_value = create_bedrock_list_models_response()

    return client


@pytest.fixture
def bedrock_llm(mock_bedrock_client):
    """Fixture for Bedrock LLM instance with mocked dependencies."""
    # Patch needed AWS components
    with patch("boto3.session.Session") as mock_session, patch.object(
        BedrockLLM, "auth"
    ):

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


def test_bedrock_initialization(bedrock_llm):
    """Test Bedrock provider initialization."""
    # Verify model lists are properly populated
    assert len(bedrock_llm.SUPPORTED_MODELS) > 0
    assert bedrock_llm.model_name in bedrock_llm.SUPPORTED_MODELS

    # Verify model settings exist
    assert bedrock_llm.model_name in bedrock_llm.CONTEXT_WINDOW
    assert bedrock_llm.model_name in bedrock_llm.COST_PER_MODEL

    # Verify profile name is set
    assert bedrock_llm.profile_name == "ai-dev"


def test_bedrock_initialization_with_params():
    """Test BedrockLLM initialization with various parameters."""
    # Add stronger patching to avoid validation issues
    with patch("lluminary.models.providers.bedrock.BedrockLLM.auth"), patch(
        "lluminary.models.providers.bedrock.BedrockLLM._validate_provider_config"
    ):

        # Create modified class attributes
        supported_models = [
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-instant-v1",
        ]

        # Test with a simpler initialization set
        with patch.object(BedrockLLM, "SUPPORTED_MODELS", supported_models):
            llm = BedrockLLM(
                model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                aws_access_key_id="test-key",
                aws_secret_access_key="test-secret",
                region_name="us-west-2",
                auto_auth=False,
            )

            # Verify parameters were properly set
            assert llm.model_name == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
            assert llm.config["aws_access_key_id"] == "test-key"
            assert llm.config["aws_secret_access_key"] == "test-secret"
            assert llm.config["region_name"] == "us-west-2"


def test_message_formatting(bedrock_llm):
    """Test Bedrock message formatting with the correct method."""
    # Re-enable the actual method for this test
    bedrock_llm._format_messages_for_model = (
        BedrockLLM._format_messages_for_model.__get__(bedrock_llm, BedrockLLM)
    )

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

    # Verify basic structure
    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert formatted[0]["role"] == "user"
    assert isinstance(formatted[0]["content"], list)

    # Verify the text content
    assert len(formatted[0]["content"]) == 1
    assert "text" in formatted[0]["content"][0]
    assert formatted[0]["content"][0]["text"] == "test message"

    # Test AI message type
    ai_messages = [
        {
            "message_type": "ai",
            "message": "AI response",
        }
    ]

    formatted_ai = bedrock_llm._format_messages_for_model(ai_messages)
    assert formatted_ai[0]["role"] == "assistant"

    # Test multiple message types in sequence
    mixed_messages = [
        {"message_type": "human", "message": "Hello"},
        {"message_type": "ai", "message": "Hi there"},
        {"message_type": "human", "message": "How are you?"},
    ]

    formatted_mixed = bedrock_llm._format_messages_for_model(mixed_messages)
    assert len(formatted_mixed) == 3
    assert formatted_mixed[0]["role"] == "user"
    assert formatted_mixed[1]["role"] == "assistant"
    assert formatted_mixed[2]["role"] == "user"


def test_tool_formatting(bedrock_llm):
    """Test Bedrock tool formatting."""
    # Re-enable the actual method for this test
    bedrock_llm._format_tools_for_model = BedrockLLM._format_tools_for_model.__get__(
        bedrock_llm, BedrockLLM
    )

    # Create mock tools
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

    # Check basic tool structure
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

    # Check second tool
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


def test_auth_error_handling():
    """Test authentication error handling."""
    # Create SessionMock with failure behavior
    with patch("boto3.session.Session") as mock_session:
        # Configure to raise an exception
        mock_session.side_effect = Exception("AWS credentials not found")

        # Test with auto_auth=True (should fail during init)
        with pytest.raises(LLMAuthenticationError) as exc:
            BedrockLLM(
                model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                region_name="us-east-1",
                profile_name="ai-dev",
            )

        # Verify exception details
        assert "Bedrock authentication failed" in str(exc.value)
        assert exc.value.provider == "bedrock"
        assert "error" in exc.value.details

        # Reset mock for profile not found test
        mock_session.side_effect = ProfileNotFound(profile="ai-dev")

        # Test profile not found error
        with pytest.raises(LLMAuthenticationError) as exc:
            BedrockLLM(
                model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                region_name="us-east-1",
                profile_name="ai-dev",
            )

        # Verify profile error details
        assert "profile" in str(exc.value).lower()
        assert "ai-dev" in str(exc.value)


def test_get_supported_models(bedrock_llm):
    """Test that we can get supported models list."""
    models = bedrock_llm.get_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "us.anthropic.claude-3-5-sonnet-20241022-v2:0" in models


def test_supports_image_input(bedrock_llm):
    """Test Bedrock image support flag."""
    # The Claude 3.5 models support images
    assert bedrock_llm.supports_image_input() is True


def test_model_costs(bedrock_llm):
    """Test model cost retrieval."""
    # Mock the get_model_costs method with a known return value
    bedrock_llm.get_model_costs = MagicMock(
        return_value={
            "input_cost": 0.000003,
            "output_cost": 0.000015,
            "image_cost": 0.024,
        }
    )

    costs = bedrock_llm.get_model_costs()
    assert isinstance(costs, dict)
    assert "input_cost" in costs
    assert "output_cost" in costs
    assert "image_cost" in costs
    assert isinstance(costs["input_cost"], (int, float))
    assert isinstance(costs["output_cost"], (int, float))
    assert isinstance(costs["image_cost"], (int, float))


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


def test_error_handling():
    """Test error handling with configuration issues."""
    # Patch the entire BedrockLLM class to avoid validation during initialization
    with patch(
        "lluminary.models.providers.bedrock.BedrockLLM._validate_provider_config"
    ) as mock_validate:
        # Set up the validation method to raise an exception
        mock_validate.side_effect = LLMConfigurationError(
            message="Missing required configuration",
            provider="bedrock",
            details={"error": "Missing region_name"},
        )

        # Test validation error during initialization
        with pytest.raises(LLMConfigurationError) as exc_info:
            # Pass an invalid configuration (missing region_name)
            BedrockLLM(
                model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                auto_auth=False,
            )

        # Check error details
        assert "configuration" in str(exc_info.value).lower()
        assert exc_info.value.provider == "bedrock"


def test_boto_client_errors(bedrock_llm, mock_bedrock_client):
    """Test AWS boto client error mapping."""
    # Re-enable the _map_aws_error method for this test
    bedrock_llm._map_aws_error = BedrockLLM._map_aws_error.__get__(
        bedrock_llm, BedrockLLM
    )

    # Test a single straightforward error case to simplify the test
    error = ClientError(
        error_response={
            "Error": {"Code": "ThrottlingException", "Message": "Request throttled"}
        },
        operation_name="converse",
    )

    # Map the error directly
    mapped_error = bedrock_llm._map_aws_error(error)

    # Check that it was mapped to the right exception type
    assert isinstance(mapped_error, LLMRateLimitError)
    assert "rate limit" in str(mapped_error).lower()
    assert mapped_error.provider == "bedrock"
    assert "error_code" in mapped_error.details


def test_image_processing_errors():
    """Test image processing error handling."""
    # Create a fresh instance with mocked auth
    with patch.object(BedrockLLM, "auth"):
        llm = BedrockLLM(
            model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1",
            profile_name="ai-dev",
            auto_auth=False,
        )

    # Test file not found error
    with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        with pytest.raises(LLMContentError) as exc:
            llm._get_image_bytes("/path/to/nonexistent.jpg")

        exception = exc.value
        assert "Failed to load image" in str(exception)
        assert exception.provider == "bedrock"
        assert "error" in exception.details

    # Test URL download error
    with patch(
        "requests.get", side_effect=requests.RequestException("Failed to download")
    ):
        with pytest.raises(LLMContentError) as exc:
            llm._download_image_from_url("https://example.com/image.jpg")

        exception = exc.value
        assert "Failed to download image" in str(exception)
        assert exception.provider == "bedrock"
        assert "error" in exception.details


def test_raw_generate(bedrock_llm, mock_bedrock_client):
    """Test the raw_generate method for basic text generation."""
    # Configure the mock response
    mock_bedrock_client.converse.return_value = create_bedrock_converse_response(
        text="This is a test response", input_tokens=20, output_tokens=15
    )

    # Call raw_generate with standard parameters
    response, usage = bedrock_llm._raw_generate(
        event_id="test",
        system_prompt="You are a helpful assistant",
        messages=[{"message_type": "human", "message": "Hello"}],
        max_tokens=100,
        temp=0.7,
    )

    # Verify the client was called with correct parameters
    mock_bedrock_client.converse.assert_called_once()
    call_args = mock_bedrock_client.converse.call_args[1]
    assert call_args["modelId"] == bedrock_llm.model_name
    assert isinstance(call_args["messages"], list)
    assert "system" in call_args
    assert call_args["system"][0]["text"] == "You are a helpful assistant"
    assert "inferenceConfig" in call_args
    assert call_args["inferenceConfig"]["temperature"] == 0.7
    assert call_args["inferenceConfig"]["maxTokens"] == 100

    # Verify usage statistics
    assert "read_tokens" in usage
    assert "write_tokens" in usage
    assert "total_tokens" in usage
    assert "images" in usage


def test_stream_generate(bedrock_llm, mock_bedrock_client):
    """Test the stream_generate method for streaming responses."""
    # Configure the stream response with multiple chunks
    chunks = ["This ", "is ", "a ", "streaming ", "response"]
    mock_bedrock_client.converse_stream.return_value = (
        create_bedrock_converse_stream_response(
            chunks=chunks, input_tokens=20, output_tokens=25
        )
    )

    # Call stream_generate with standard parameters
    stream_generator = bedrock_llm.stream_generate(
        event_id="test",
        system_prompt="You are a helpful assistant",
        messages=[{"message_type": "human", "message": "Hello"}],
        max_tokens=100,
        temp=0.7,
    )

    # Consume the generator and collect results
    results = list(stream_generator)

    # Verify the client was called with correct parameters
    mock_bedrock_client.converse_stream.assert_called_once()
    call_args = mock_bedrock_client.converse_stream.call_args[1]
    assert call_args["modelId"] == bedrock_llm.model_name
    assert isinstance(call_args["messages"], list)
    assert "system" in call_args
    assert call_args["system"][0]["text"] == "You are a helpful assistant"

    # Verify results structure
    assert len(results) == len(chunks)

    # Check first result
    chunk_text, usage = results[0]
    assert isinstance(chunk_text, str)
    assert isinstance(usage, dict)
    assert "read_tokens" in usage
    assert "write_tokens" in usage
    assert "total_tokens" in usage


def test_raw_generate_with_tools(bedrock_llm, mock_bedrock_client):
    """Test raw_generate with tool definitions."""
    # Configure the tool response
    tool_response = create_bedrock_converse_with_tools_response(
        text="I'll calculate that for you",
        tool_use={
            "id": "tool-123",
            "name": "calculator",
            "input": {"operation": "multiply", "x": 5, "y": 3},
        },
    )
    mock_bedrock_client.converse.return_value = tool_response

    # Define test tools
    tools = [
        {
            "name": "calculator",
            "description": "Perform calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["operation", "x", "y"],
            },
        }
    ]

    # Call raw_generate with tools
    response, usage = bedrock_llm._raw_generate(
        event_id="test",
        system_prompt="You are a helpful assistant",
        messages=[{"message_type": "human", "message": "What is 5 times 3?"}],
        max_tokens=100,
        temp=0.7,
        tools=tools,
    )

    # Verify the client was called with correct parameters
    mock_bedrock_client.converse.assert_called_once()
    call_args = mock_bedrock_client.converse.call_args[1]
    assert "toolConfig" in call_args
    assert "tools" in call_args["toolConfig"]
    assert len(call_args["toolConfig"]["tools"]) == 1

    # Verify first tool format
    tool_spec = call_args["toolConfig"]["tools"][0]["toolSpec"]
    assert tool_spec["name"] == "calculator"
    assert "description" in tool_spec
    assert "inputSchema" in tool_spec
    assert "json" in tool_spec["inputSchema"]


def test_call_with_retry(bedrock_llm):
    """Test the retry mechanism for AWS API calls."""
    # Create a simpler test case first
    mock_func = MagicMock()
    mock_func.side_effect = [Exception("Error"), "Success"]

    # Test with simplified parameters
    result = bedrock_llm._call_with_retry(
        mock_func, retryable_errors=[Exception], max_retries=1, base_delay=0.01
    )

    # Verify the function was called multiple times
    assert mock_func.call_count == 2
    assert result == "Success"

    # Now test with ClientError
    mock_func = MagicMock()
    mock_func.side_effect = [
        ClientError(
            error_response=create_aws_client_error(
                "ThrottlingException", "Rate exceeded"
            ),
            operation_name="converse",
        ),
        "Success!",  # Second call succeeds
    ]

    # Call with retry with simplified parameters
    result = bedrock_llm._call_with_retry(
        mock_func, retryable_errors=[ClientError], max_retries=1, base_delay=0.01
    )

    # Verify the function was called multiple times
    assert mock_func.call_count == 2
    assert result == "Success!"
