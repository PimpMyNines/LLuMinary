from unittest.mock import patch

from lluminary.exceptions import LLMAuthenticationError
from lluminary.models.providers.bedrock import BedrockLLM

# Test auth error handling by forcing an exception
with patch("boto3.session.Session") as mock_session:
    # Set up mock to raise exception
    mock_session.side_effect = Exception("AWS credentials not found")

    # Create a new LLM instance that will use our mocked boto3
    llm = BedrockLLM(
        model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-east-1",
    )

    # Call auth directly and verify it raises the right exception
    try:
        llm.auth()
        print("No exception was raised!")
    except LLMAuthenticationError as e:
        # Verify exception details
        print(f"Exception message: {e!s}")
        print(f"Exception provider: {e.provider}")
        print(f"Exception details: {e.details}")
    except Exception as e:
        print(f"Unexpected exception type: {type(e).__name__}")
        print(f"Exception message: {e!s}")
