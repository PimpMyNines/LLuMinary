from unittest.mock import MagicMock, patch

from lluminary.exceptions import LLMMistake
from lluminary.models.providers.bedrock import BedrockLLM

# Create a fresh instance with mocked dependencies
with patch.object(BedrockLLM, "auth"):
    llm = BedrockLLM(
        model_name="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-east-1",
    )

# Test file not found error
with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
    try:
        llm._get_image_bytes("/path/to/nonexistent.jpg")
    except LLMMistake as e:
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e!s}")
        print(f"Exception provider: {e.provider}")
        print(f"Exception error_type: {e.error_type}")
        print(f"Exception details: {e.details}")

# Test general image processing error
with patch("builtins.open") as mock_open, patch(
    "PIL.Image.open", side_effect=Exception("Invalid image format")
):
    # Mock the file open but fail on image processing
    mock_open.return_value.__enter__.return_value = MagicMock()

    try:
        llm._get_image_bytes("/path/to/corrupted.jpg")
    except Exception as e:
        print(f"\nException type: {type(e).__name__}")
        print(f"Exception message: {e!s}")
        if hasattr(e, "provider"):
            print(f"Exception provider: {e.provider}")
        if hasattr(e, "details"):
            print(f"Exception details: {e.details}")

# Test URL download error
with patch("requests.get", side_effect=Exception("Failed to download")):
    try:
        llm._download_image_from_url("https://example.com/image.jpg")
    except LLMMistake as e:
        print(f"\nException type: {type(e).__name__}")
        print(f"Exception message: {e!s}")
        print(f"Exception provider: {e.provider}")
        print(f"Exception error_type: {e.error_type}")
        print(f"Exception details: {e.details}")
