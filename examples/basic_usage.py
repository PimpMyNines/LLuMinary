"""
Basic usage examples for LLMHandler package.
"""

from lluminary import get_llm_from_model


def basic_text_generation():
    """Example of basic text generation."""
    # Initialize any supported model
    llm = get_llm_from_model("gemini-2.0-flash-lite")  # or "o1" for OpenAI

    # Generate text
    response, usage, updated_messages = llm.generate(
        event_id="basic_example",
        system_prompt="You are a helpful AI assistant.",
        messages=[
            {
                "message_type": "human",
                "message": "Explain what an LLM is in one sentence.",
                "image_paths": [],
                "image_urls": [],
            }
        ],
        max_tokens=100,
    )

    print("Response:", response)
    print(f"Cost: ${usage['total_cost']:.6f}")
    print(f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write")

    return response, usage


def classification_example():
    """Example of using the classification feature."""
    # Initialize model
    llm = get_llm_from_model("claude-haiku-3.5")

    # Define categories
    categories = {
        "question": "A query seeking information",
        "command": "A directive to perform an action",
        "statement": "A declarative sentence",
    }

    # Optional examples for better accuracy
    examples = [
        {
            "user_input": "What is the weather like?",
            "doc_str": "This is a question seeking information about weather",
            "selection": "question",
        }
    ]

    # Message to classify
    message = {
        "message_type": "human",
        "message": "Tell me how to bake a cake.",
        "image_paths": [],
        "image_urls": [],
    }

    # Perform classification
    selection, usage = llm.classify(
        messages=[message], categories=categories, examples=examples
    )

    print("\nClassification result:", selection)
    print(f"Cost: ${usage['total_cost']:.6f}")

    return selection, usage


def function_calling_example():
    """Example of using function calling."""

    # Define a simple function
    def get_weather(location: str) -> str:
        """Get the current weather in a given location"""
        # In a real application, this would make an API call
        return f"72°F and sunny in {location}"

    # Initialize model
    llm = get_llm_from_model("gpt-4o-mini")

    # Create message that should trigger the function
    message = {
        "message_type": "human",
        "message": "What's the weather like in San Francisco?",
        "image_paths": [],
        "image_urls": [],
    }

    # First call - should trigger tool use
    print("\nSending query that should trigger function call...")
    response, usage, messages = llm.generate(
        event_id="function_example",
        system_prompt="You are a helpful assistant.",
        messages=[message],
        max_tokens=1000,
        functions=[get_weather],  # Pass functions that the LLM can use
    )

    print("Response:", response)

    # Check if a tool was used
    if usage.get("tool_use"):
        tool_id = usage["tool_use"].get("id") or usage["tool_use"].get("name")
        print(f"Tool used: {tool_id}")

        # In a real application, you would actually call the function here
        weather_result = get_weather("San Francisco")

        # Add function result to conversation
        messages.append(
            {
                "message_type": "tool_result",
                "tool_result": {
                    "tool_id": tool_id,
                    "success": True,
                    "result": weather_result,
                    "error": None,
                },
            }
        )

        # Get final response after tool use
        print("\nSending follow-up with tool result...")
        final_response, final_usage, _ = llm.generate(
            event_id="function_example_followup",
            system_prompt="You are a helpful assistant.",
            messages=messages,
            max_tokens=1000,
            functions=[get_weather],
        )

        print("Final response:", final_response)
        print(f"Cost: ${final_usage['total_cost']:.6f}")
    else:
        print("No tool was used.")

    return response, usage


if __name__ == "__main__":
    print("== Basic Text Generation Example ==")
    basic_text_generation()

    print("\n== Classification Example ==")
    classification_example()

    print("\n== Function Calling Example ==")
    function_calling_example()
