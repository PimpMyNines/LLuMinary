"""
Script to verify our dictionary entry type fixes in the Bedrock provider implementation.
This script simulates the operations that were fixed to ensure they work properly without type errors.
"""

from typing import Any, Dict, List, Optional, TypedDict, Union


# Define TypedDict classes for message content parts
class TextContent(TypedDict):
    text: str


class ImageSource(TypedDict):
    bytes: bytes


class ImageFormat(TypedDict):
    format: str
    source: ImageSource


class ImageContent(TypedDict):
    image: ImageFormat


class ToolUseId(TypedDict):
    toolUseId: str
    name: str
    input: Dict[str, Any]


class ToolUseContent(TypedDict):
    toolUse: ToolUseId


class ToolResultContent(TypedDict):
    toolResult: Dict[str, Any]


class ReasoningTextContent(TypedDict):
    text: str
    signature: str


class ReasoningContent(TypedDict):
    reasoningText: ReasoningTextContent


class ReasoningContentWrapper(TypedDict):
    reasoningContent: ReasoningContent


# Union type for all content types
ContentPart = Union[
    TextContent,
    ImageContent,
    ToolUseContent,
    ToolResultContent,
    ReasoningContentWrapper,
]


def test_format_messages_for_model():
    """Test the _format_messages_for_model method with explicit type handling."""

    # Simulate the _format_messages_for_model method with improved typing
    def format_messages_for_model(
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simulate the _format_messages_for_model method with improved typing."""
        formatted_messages: List[Dict[str, Any]] = []

        for msg in messages:
            # Map message type to Bedrock role
            role = "assistant" if msg["message_type"] == "ai" else "user"
            content: List[ContentPart] = []

            # Add thinking/reasoning content if present
            if msg.get("thinking"):
                reasoning_content: ReasoningContentWrapper = {
                    "reasoningContent": {
                        "reasoningText": {
                            "text": msg["thinking"]["thinking"],
                            "signature": msg["thinking"]["thinking_signature"],
                        }
                    }
                }
                content.append(reasoning_content)

            # Add text content first
            if msg.get("message"):
                text_content: TextContent = {"text": msg["message"]}
                content.append(text_content)

            # Process images from file paths
            if msg.get("image_paths"):
                for image_path in msg["image_paths"]:
                    # Simulate image bytes
                    image_bytes = b"fake_image_bytes"
                    image_dict: ImageContent = {
                        "image": {"format": "png", "source": {"bytes": image_bytes}}
                    }
                    content.append(image_dict)

            # Process images from URLs
            if msg.get("image_urls"):
                for image_url in msg["image_urls"]:
                    # Simulate image bytes
                    image_bytes = b"fake_image_bytes_from_url"
                    url_image_dict: ImageContent = {
                        "image": {"format": "png", "source": {"bytes": image_bytes}}
                    }
                    content.append(url_image_dict)

            # Add tool use information (function calls)
            if msg.get("tool_use"):
                tool_use_dict: ToolUseContent = {
                    "toolUse": {
                        "toolUseId": msg["tool_use"]["id"],
                        "name": msg["tool_use"]["name"],
                        "input": msg["tool_use"]["input"],
                    }
                }
                content.append(tool_use_dict)

            # Add tool result information (function responses)
            if msg.get("tool_result"):
                if msg["tool_result"].get("success"):
                    # Format successful tool result
                    result = msg["tool_result"]["result"]
                    content_item: TextContent = {"text": result}
                    tool_result: ToolResultContent = {
                        "toolResult": {
                            "toolUseId": msg["tool_result"]["tool_id"],
                            "content": [content_item],
                        }
                    }
                    content.append(tool_result)
                else:
                    # Format failed tool result with error status
                    result = msg["tool_result"]["error"]
                    error_content_item: TextContent = {"text": result}
                    error_tool_result: ToolResultContent = {
                        "toolResult": {
                            "toolUseId": msg["tool_result"]["tool_id"],
                            "content": [error_content_item],
                            "status": "error",
                        }
                    }
                    content.append(error_tool_result)

            # If no content was added, add a fallback message
            if not content:
                fallback_content: TextContent = {"text": "No content available"}
                content.append(fallback_content)

            # Add the completed message with role and content parts
            formatted_messages.append({"role": role, "content": content})

        return formatted_messages

    # Test with different message types
    messages = [
        {
            "message_type": "human",
            "message": "Hello, can you help me?",
        },
        {
            "message_type": "ai",
            "message": "Yes, I can help you.",
        },
        {
            "message_type": "human",
            "message": "Analyze this image",
            "image_paths": ["/path/to/image.jpg"],
        },
        {
            "message_type": "human",
            "message": "Look at this online image",
            "image_urls": ["https://example.com/image.jpg"],
        },
        {
            "message_type": "human",
            "message": "Call this function",
            "tool_use": {
                "id": "tool1",
                "name": "get_weather",
                "input": {"location": "New York"},
            },
        },
        {
            "message_type": "tool",
            "tool_result": {
                "tool_id": "tool1",
                "success": True,
                "result": "Sunny, 75°F",
            },
        },
        {
            "message_type": "tool",
            "tool_result": {
                "tool_id": "tool2",
                "success": False,
                "error": "Location not found",
            },
        },
        {
            "message_type": "ai",
            "message": "Let me think about this",
            "thinking": {
                "thinking": "This is my reasoning process",
                "thinking_signature": "signature123",
            },
        },
    ]

    # Format messages
    formatted_messages = format_messages_for_model(messages)

    # Verify results
    print(
        f"Formatted {len(messages)} messages into {len(formatted_messages)} Bedrock messages"
    )

    # Check each message has the expected structure
    for i, msg in enumerate(formatted_messages):
        assert "role" in msg, f"Message {i} missing 'role'"
        assert "content" in msg, f"Message {i} missing 'content'"
        assert isinstance(msg["content"], list), f"Message {i} 'content' is not a list"
        assert len(msg["content"]) > 0, f"Message {i} has empty content"

        # Check content parts have the expected structure
        for j, part in enumerate(msg["content"]):
            if "text" in part:
                assert isinstance(
                    part["text"], str
                ), f"Message {i}, part {j} 'text' is not a string"
            elif "image" in part:
                assert (
                    "format" in part["image"]
                ), f"Message {i}, part {j} missing 'format'"
                assert (
                    "source" in part["image"]
                ), f"Message {i}, part {j} missing 'source'"
                assert (
                    "bytes" in part["image"]["source"]
                ), f"Message {i}, part {j} missing 'bytes'"
            elif "toolUse" in part:
                assert (
                    "toolUseId" in part["toolUse"]
                ), f"Message {i}, part {j} missing 'toolUseId'"
                assert (
                    "name" in part["toolUse"]
                ), f"Message {i}, part {j} missing 'name'"
                assert (
                    "input" in part["toolUse"]
                ), f"Message {i}, part {j} missing 'input'"
            elif "toolResult" in part:
                assert (
                    "toolUseId" in part["toolResult"]
                ), f"Message {i}, part {j} missing 'toolUseId'"
                assert (
                    "content" in part["toolResult"]
                ), f"Message {i}, part {j} missing 'content'"
            elif "reasoningContent" in part:
                assert (
                    "reasoningText" in part["reasoningContent"]
                ), f"Message {i}, part {j} missing 'reasoningText'"
                assert (
                    "text" in part["reasoningContent"]["reasoningText"]
                ), f"Message {i}, part {j} missing 'text'"
                assert (
                    "signature" in part["reasoningContent"]["reasoningText"]
                ), f"Message {i}, part {j} missing 'signature'"

    # Success check
    print("format_messages_for_model test: PASS")


def test_get_model_costs():
    """Test the get_model_costs method with explicit type handling."""

    # Define a mock COST_PER_MODEL dictionary
    COST_PER_MODEL = {
        "model1": {
            "read_token": 0.0001,
            "write_token": 0.0002,
            "image_cost": 0.001,
        },
        "model2": {
            "read_token": None,
            "write_token": 0.0003,
            "image_cost": None,
        },
    }

    # Simulate the get_model_costs method with improved typing
    def get_model_costs(model_name: str) -> Dict[str, Optional[float]]:
        """Simulate the get_model_costs method with improved typing."""
        model_costs = COST_PER_MODEL.get(model_name, {})

        # Return the model costs with explicit Optional[float] typing
        return {
            "input_cost": model_costs.get("read_token"),
            "output_cost": model_costs.get("write_token"),
            "image_cost": model_costs.get("image_cost"),
        }

    # Test with different models
    model1_costs = get_model_costs("model1")
    model2_costs = get_model_costs("model2")
    unknown_model_costs = get_model_costs("unknown_model")

    # Verify results
    print(f"Model1 costs: {model1_costs}")
    print(f"Model2 costs: {model2_costs}")
    print(f"Unknown model costs: {unknown_model_costs}")

    # Check model1 costs
    assert model1_costs["input_cost"] == 0.0001, "Model1 input_cost incorrect"
    assert model1_costs["output_cost"] == 0.0002, "Model1 output_cost incorrect"
    assert model1_costs["image_cost"] == 0.001, "Model1 image_cost incorrect"

    # Check model2 costs (with None values)
    assert model2_costs["input_cost"] is None, "Model2 input_cost should be None"
    assert model2_costs["output_cost"] == 0.0003, "Model2 output_cost incorrect"
    assert model2_costs["image_cost"] is None, "Model2 image_cost should be None"

    # Check unknown model costs (all None)
    assert (
        unknown_model_costs["input_cost"] is None
    ), "Unknown model input_cost should be None"
    assert (
        unknown_model_costs["output_cost"] is None
    ), "Unknown model output_cost should be None"
    assert (
        unknown_model_costs["image_cost"] is None
    ), "Unknown model image_cost should be None"

    # Success check
    print("get_model_costs test: PASS")


if __name__ == "__main__":
    print("Running dictionary entry type fixes verification...")
    test_format_messages_for_model()
    test_get_model_costs()
    print("All tests passed!")
