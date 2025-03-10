"""
Example test file demonstrating how to test with LLMHandler.
This file is meant as a simple template for users to understand testing patterns.
"""

from lluminary.exceptions import LLMMistake
from lluminary.models.router import get_llm_from_model


def test_simple_completion():
    """
    Basic example of a text completion test.
    """
    # Choose a fast and inexpensive model for testing
    model_name = "claude-haiku-3.5"

    # Initialize model
    llm = get_llm_from_model(model_name)

    # Generate response
    response, usage, updated_messages = llm.generate(
        event_id="example_test",
        system_prompt="You are a helpful AI assistant.",
        messages=[
            {
                "message_type": "human",
                "message": "Hello, how are you?",
                "image_paths": [],
                "image_urls": [],
            }
        ],
        max_tokens=50,
    )

    # Print the response and usage statistics
    print(f"Response: {response}")
    print(f"Read tokens: {usage['read_tokens']}")
    print(f"Write tokens: {usage['write_tokens']}")
    print(f"Total cost: ${usage['total_cost']:.6f}")


def test_json_response():
    """
    Example of getting a structured JSON response using processing function.
    """
    model_name = "gemini-2.0-flash-lite"

    # Define a JSON validation function
    def process_json(response: str):
        import json

        try:
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                json_str = response[response.find("{") : response.rfind("}") + 1]
                data = json.loads(json_str)
                return data
            else:
                raise LLMMistake("Response does not contain valid JSON.")
        except json.JSONDecodeError:
            raise LLMMistake("Failed to parse JSON from response.")

    # Initialize model
    llm = get_llm_from_model(model_name)

    # Generate response with JSON processing
    response, usage, _ = llm.generate(
        event_id="json_test",
        system_prompt="You are a helpful AI assistant.",
        messages=[
            {
                "message_type": "human",
                "message": "Return a JSON object with your name and version.",
                "image_paths": [],
                "image_urls": [],
            }
        ],
        max_tokens=200,
        result_processing_function=process_json,
        retry_limit=2,
    )

    # Print the structured response
    print(f"Structured response: {response}")
    print(f"Type: {type(response)}")
    print(f"Total cost: ${usage['total_cost']:.6f}")


if __name__ == "__main__":
    print("Running example test_simple_completion...")
    test_simple_completion()

    print("\nRunning example test_json_response...")
    test_json_response()
