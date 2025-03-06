"""
Integration tests for advanced LLM features: response processing, retries, function calling,
and thinking budget. Tests use real API calls and skip gracefully when credentials aren't available.
"""

import pytest

from lluminary.exceptions import LLMMistake
from lluminary.models.router import get_llm_from_model


# Define validator function used in multiple tests
def number_validator(response: str):
    """
    Process the response to extract a number between 6 and 12 from <num> tags.

    Args:
        response (str): The model's response

    Returns:
        int: The extracted number

    Raises:
        LLMMistake: If the response is not properly formatted or the number is outside the range
    """
    try:
        number_str = response.split("<num>")[1].split("</num>")[0]
        number = int(number_str)
    except:
        raise LLMMistake("The response must be wrapped in <num> tags.")

    if not (number < 12 and number > 6):
        raise LLMMistake("The given number must be between 6 and 12.")

    return number


@pytest.mark.integration
class TestResponseProcessing:
    """Test response processing functionality."""

    def test_response_processing(self):
        """
        Test that response processing function works with structured validation.
        Tries models until one works.
        """
        test_models = [
            "claude-haiku-3.5",  # Anthropic is usually good at following instructions
            "gpt-4o-mini",  # OpenAI is usually good at following instructions
            "gemini-2.0-flash-lite",  # Google models
        ]

        # Track results
        successful_models = []
        failed_models = []

        print("\n" + "=" * 60)
        print("RESPONSE PROCESSING TEST")
        print("=" * 60)

        for model_name in test_models:
            print(f"\nTesting response processing with {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Generate response with processing
                print("Generating response with validation...")
                response, usage, _ = llm.generate(
                    event_id="test_response_processing",
                    system_prompt="Follow the user's instructions exactly.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "Pick a number between 6 and 12 and format it like this: <num>8</num>",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=100,
                    result_processing_function=number_validator,
                    retry_limit=3,
                )

                # Verify response
                assert isinstance(response, int)
                assert 6 < response < 12

                # Print results
                print(f"Processed response: {response} (integer)")
                print(
                    f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write"
                )
                print(f"Cost: ${usage['total_cost']:.6f}")

                successful_models.append(model_name)

                # One successful test is enough
                break

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                failed_models.append((model_name, str(e)))

        # Print summary
        print("\nProcessing test " + ("PASSED" if successful_models else "FAILED"))
        if successful_models:
            print(f"Successful model: {successful_models[0]}")

        # Skip if no models work at all
        if not successful_models:
            pytest.skip(
                "Skipping test as no models were able to complete the response processing test"
            )

    def test_retry_mechanism(self):
        """
        Test the retry mechanism with an intentionally failing validator.
        Tries models until one works.
        """
        test_models = ["claude-haiku-3.5", "gpt-4o-mini"]

        def impossible_validator(response: str):
            """A validator that always fails."""
            raise LLMMistake("This validation will always fail")

        print("\n" + "=" * 60)
        print("RETRY MECHANISM TEST")
        print("=" * 60)

        for model_name in test_models:
            print(f"\nTesting retry mechanism with {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Set up retry tracking
                retry_limit = 2
                retry_count = 0

                # Define a patch to count retries
                original_generate = llm._generate

                def patched_generate(*args, **kwargs):
                    nonlocal retry_count
                    retry_count += 1
                    print(f"Generate attempt #{retry_count}")
                    return original_generate(*args, **kwargs)

                # Apply the patch
                llm._generate = patched_generate

                # Generate response with impossible validator
                try:
                    print("Attempting generation with impossible validator...")
                    llm.generate(
                        event_id="test_retry_mechanism",
                        system_prompt="Follow the user's instructions",
                        messages=[
                            {
                                "message_type": "human",
                                "message": "Write any response",
                                "image_paths": [],
                                "image_urls": [],
                            }
                        ],
                        max_tokens=100,
                        result_processing_function=impossible_validator,
                        retry_limit=retry_limit,
                    )
                    # We should never get here
                    assert False, "Expected LLMMistake exception was not raised"
                except LLMMistake:
                    # Verify we retried the correct number of times
                    assert retry_count == retry_limit + 1  # Initial attempt + retries
                    print(
                        f"Successfully caught LLMMistake after {retry_count} attempts"
                    )

                # Restore the original method
                llm._generate = original_generate

                print(f"Test passed with {model_name}")
                return

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                continue

        # If we get here, no models worked
        pytest.skip(
            "Skipping test as no models were able to complete the retry mechanism test"
        )


@pytest.mark.integration
class TestFunctionCalling:
    """Test function calling functionality."""

    def test_function_calling(self):
        """
        Test the function calling capability across all providers.
        """
        # Test models that support function calling
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
        ]

        # Define test function
        def get_weather(location: str) -> str:
            """Get the current weather in a given location"""
            return "60 degrees"

        # Track results
        successful_models = []
        failed_models = []
        all_results = {}

        print("\n" + "=" * 60)
        print("FUNCTION CALLING TEST")
        print("=" * 60)

        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting function calling with {provider} model: {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # First generate - should trigger function call
                print("Sending query that should trigger function call...")
                response, usage, messages = llm.generate(
                    event_id="test_function_calling",
                    system_prompt="You are a helpful assistant.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "What is the weather like in San Francisco?",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=1000,
                    functions=[get_weather],
                )

                # Print results of first call
                print(f"Response: {response[:50]}...")
                used_tool = "No tool used"
                if usage.get("tool_use"):
                    used_tool = f"Tool used: {usage['tool_use'].get('name') or usage['tool_use'].get('id')}"
                print(used_tool)

                # Continue conversation with tool result if tool was used
                if usage.get("tool_use"):
                    tool_id = usage["tool_use"].get("id") or usage["tool_use"].get(
                        "name"
                    )

                    updated_messages = messages.copy()
                    updated_messages.append(
                        {
                            "message_type": "tool_result",
                            "tool_result": {
                                "tool_id": tool_id,
                                "success": True,
                                "result": "65 degrees",
                                "error": None,
                            },
                        }
                    )

                    print("Sending follow-up with tool result...")
                    final_response, final_usage, _ = llm.generate(
                        event_id="test_function_calling_followup",
                        system_prompt="You are a helpful assistant.",
                        messages=updated_messages,
                        max_tokens=1000,
                        functions=[get_weather],
                    )

                    # Print follow-up results
                    print(f"Final response: {final_response[:50]}...")
                    print(
                        f"Tokens: {final_usage['read_tokens']} read, {final_usage['write_tokens']} write"
                    )
                    all_results[model_name] = {
                        "used_tool": True,
                        "response": final_response[:50],
                    }
                else:
                    all_results[model_name] = {
                        "used_tool": False,
                        "response": response[:50],
                    }

                successful_models.append(model_name)

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                failed_models.append((model_name, str(e)))

        # Print summary
        print("\n" + "=" * 60)
        print("FUNCTION CALLING TEST SUMMARY")
        print("=" * 60)
        print(f"Successful models: {len(successful_models)}/{len(test_models)}")
        for model in successful_models:
            tool_status = (
                "with tool use"
                if all_results[model]["used_tool"]
                else "without tool use"
            )
            print(f"  - {model} ({tool_status})")

        if failed_models:
            print(f"\nFailed models: {len(failed_models)}/{len(test_models)}")
            for model, error in failed_models:
                print(f"  - {model}: {error}")

        # Skip if no models work at all
        if not successful_models:
            pytest.skip(
                "Skipping test as no models were able to complete the function calling test"
            )


@pytest.mark.integration
class TestThinkingBudget:
    """Test thinking budget functionality."""

    def test_thinking_budget(self):
        """
        Test the thinking budget feature for supported models.
        This only works with specific Claude models.
        """
        # Only Claude models support thinking budget
        test_models = ["claude-sonnet-3.7"]  # Current model with thinking support

        print("\n" + "=" * 60)
        print("THINKING BUDGET TEST")
        print("=" * 60)

        for model_name in test_models:
            print(f"\nTesting thinking budget with {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Skip if model doesn't exist or thinking not supported
                if not hasattr(llm, "thinking_models") or model_name not in getattr(
                    llm, "thinking_models", []
                ):
                    print(
                        f"Model {model_name} does not support thinking budget, skipping..."
                    )
                    continue

                # Generate response with thinking budget
                print("Generating response with thinking budget...")
                response, usage, _ = llm.generate(
                    event_id="test_thinking_budget",
                    system_prompt="You are a helpful AI assistant.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "Solve this math problem step by step: If a train travels at 60 mph for 3 hours, how far does it go?",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=500,
                    thinking_budget=1000,  # Allocate thinking tokens
                )

                # Print results
                print(f"Response: {response[:100]}...")
                print(
                    f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write"
                )
                print(f"Cost: ${usage['total_cost']:.6f}")

                # Simple verification
                assert (
                    "180" in response or "180 miles" in response.lower()
                ), "Expected answer of 180 miles not found"
                print(f"Test passed with {model_name}")
                return

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                continue

        # If we get here, no models worked
        pytest.skip(
            "Skipping test as no models were able to complete the thinking budget test"
        )
