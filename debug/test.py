from lluminary.exceptions import LLMMistake
from lluminary.models.router import (
    MODEL_MAPPINGS,
    get_llm_from_model,
    list_available_models,
)


def get_weather(location: str) -> str:
    """Get the current weather in a given location

    Args:
        location: The city and state, e.g. San Francisco, CA

    Returns:
        The current weather in the given location
    """
    return "60 degrees"


def test_image_description(image_url: str, models_to_test=None):
    """
    Test image description capabilities across specified models.

    Args:
        image_url (str): URL of the image to test
        models_to_test (List[str], optional): List of model names to test. If None, all models will be tested.
    """
    if models_to_test is None:
        # Use all models if none specified
        models_to_test = list(MODEL_MAPPINGS.keys())

    # Track successful and failed models
    successful_models = []
    failed_models = []

    for model_name in models_to_test:
        provider_name = MODEL_MAPPINGS[model_name]["provider"].__name__

        print(f"\nTesting {provider_name} ({model_name})...")

        try:
            # Initialize the model
            llm = get_llm_from_model(model_name)

            # Check if the model supports images
            if not llm.supports_image_input():
                print(f"  - Model {model_name} does not support image input. Skipping.")
                continue

            # Generate response
            response, usage, updated_messages = llm.generate(
                event_id="image_test",
                system_prompt="You are a helpful AI assistant skilled at describing images.",
                messages=[
                    {
                        "message_type": "human",
                        "message": "Please describe the absolute opposite of this image. Be very creative and thoughtful.",
                        "image_paths": [],
                        "image_urls": [image_url],
                    }
                ],
                max_tokens=4000,
                thinking_budget=16000,
            )

            # Print results
            print("\nResponse:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            print("Usage statistics:")
            print(f"- Read tokens: {usage['read_tokens']}")
            print(f"- Write tokens: {usage['write_tokens']}")
            print(f"- Total cost: ${usage['total_cost']:.6f}")

            successful_models.append(model_name)

        except Exception as e:
            print(f"Error with {model_name}: {e!s}")
            failed_models.append((model_name, str(e)))

    # Print summary
    print("\n" + "=" * 70)
    print("IMAGE DESCRIPTION TEST SUMMARY")
    print("=" * 70)
    print(f"Successful models: {len(successful_models)}/{len(models_to_test)}")
    for model in successful_models:
        print(f"  - {model}")

    if failed_models:
        print(f"\nFailed models: {len(failed_models)}/{len(models_to_test)}")
        for model, error in failed_models:
            print(f"  - {model}: {error}")

    return successful_models, failed_models


def test_single_category_classification(models_to_test=None):
    """
    Test basic single category classification across specified models.

    Args:
        models_to_test (List[str], optional): List of model names to test. If None, all models will be tested.
    """
    if models_to_test is None:
        # Use all models if none specified
        models_to_test = list(MODEL_MAPPINGS.keys())

    # Test data
    categories = {
        "question": "A query seeking information",
        "command": "A directive to perform an action",
        "statement": "A declarative sentence",
    }

    examples = [
        {
            "user_input": "What is the weather like?",
            "doc_str": "This is a question seeking information about weather",
            "selection": "question",
        }
    ]

    messages = [
        {
            "message_type": "human",
            "message": "How do I open this file?",
            "image_paths": [],
            "image_urls": [],
        }
    ]

    # Track results
    successful_models = []
    failed_models = []
    all_results = {}

    print("\n" + "=" * 70)
    print("CLASSIFICATION TEST")
    print("=" * 70)

    for model_name in models_to_test:
        provider_name = MODEL_MAPPINGS[model_name]["provider"].__name__
        print(f"\nTesting classification with {provider_name} ({model_name})...")

        try:
            # Initialize the model
            llm = get_llm_from_model(model_name)

            # Perform classification
            selections, usage = llm.classify(messages, categories, examples)

            # Print results
            print(f"Classification result: {selections}")
            print("Usage statistics:")
            print(f"- Read tokens: {usage['read_tokens']}")
            print(f"- Write tokens: {usage['write_tokens']}")
            print(f"- Total cost: ${usage['total_cost']:.6f}")

            successful_models.append(model_name)
            all_results[model_name] = selections

        except Exception as e:
            print(f"Error with {model_name}: {e!s}")
            failed_models.append((model_name, str(e)))

    # Print summary
    print("\n" + "=" * 70)
    print("CLASSIFICATION TEST SUMMARY")
    print("=" * 70)
    print(f"Successful models: {len(successful_models)}/{len(models_to_test)}")
    for model in successful_models:
        print(f"  - {model}: {all_results[model]}")

    if failed_models:
        print(f"\nFailed models: {len(failed_models)}/{len(models_to_test)}")
        for model, error in failed_models:
            print(f"  - {model}: {error}")

    return all_results, successful_models, failed_models


def process_function(response: str):
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


def test_processing_function(models_to_test=None):
    """
    Test result processing function across specified models.

    Args:
        models_to_test (List[str], optional): List of model names to test. If None, all models will be tested.
    """
    if models_to_test is None:
        # Use all models if none specified
        models_to_test = list(MODEL_MAPPINGS.keys())

    # Track results
    successful_models = []
    failed_models = []
    all_results = {}

    print("\n" + "=" * 70)
    print("PROCESSING FUNCTION TEST")
    print("=" * 70)

    for model_name in models_to_test:
        provider_name = MODEL_MAPPINGS[model_name]["provider"].__name__
        print(f"\nTesting processing function with {provider_name} ({model_name})...")

        try:
            # Initialize the model
            llm = get_llm_from_model(model_name)

            # Test processing function
            response, usage, updated_messages = llm.generate(
                event_id="test_event",
                system_prompt="Follow the user's instructions",
                messages=[
                    {
                        "message_type": "human",
                        "message": "Pick a number between 6 and 12 and format it like this: <num>8</num>",
                        "image_paths": [],
                        "image_urls": [],
                    }
                ],
                max_tokens=100,
                result_processing_function=process_function,
                retry_limit=3,
                thinking_budget=16000,
            )

            # Print results
            print(f"Result: {response}")
            print(f"Type: {type(response)}")
            print("Usage statistics:")
            print(f"- Read tokens: {usage['read_tokens']}")
            print(f"- Write tokens: {usage['write_tokens']}")
            print(f"- Total cost: ${usage['total_cost']:.6f}")

            successful_models.append(model_name)
            all_results[model_name] = response

        except Exception as e:
            print(f"Error with {model_name}: {e!s}")
            failed_models.append((model_name, str(e)))

    # Print summary
    print("\n" + "=" * 70)
    print("PROCESSING FUNCTION TEST SUMMARY")
    print("=" * 70)
    print(f"Successful models: {len(successful_models)}/{len(models_to_test)}")
    for model in successful_models:
        print(f"  - {model}: {all_results[model]}")

    if failed_models:
        print(f"\nFailed models: {len(failed_models)}/{len(models_to_test)}")
        for model, error in failed_models:
            print(f"  - {model}: {error}")

    return all_results, successful_models, failed_models


def test_tool_use(models_to_test=None):
    """
    Test result processing function across specified models.

    Args:
        models_to_test (List[str], optional): List of model names to test. If None, all models will be tested.
    """
    if models_to_test is None:
        # Use all models if none specified
        models_to_test = list(MODEL_MAPPINGS.keys())

    # Track results
    successful_models = []
    failed_models = []
    all_results = {}

    print("\n" + "=" * 70)
    print("TOOL USE TEST")
    print("=" * 70)

    for model_name in models_to_test:
        provider_name = MODEL_MAPPINGS[model_name]["provider"].__name__
        print(f"\nTesting tool use with {provider_name} ({model_name})...")

        try:
            # Initialize the model
            llm = get_llm_from_model(model_name)

            messages = [
                {
                    "message_type": "human",
                    "message": "What is the weather like in San Francisco?",
                    "image_paths": [],
                    "image_urls": [],
                }
            ]

            # Test processing function
            response, usage, updated_messages = llm.generate(
                event_id="test_event",
                system_prompt="Follow the user's instructions",
                messages=messages,
                max_tokens=1000,
                thinking_budget=16000,
                functions=[get_weather],
            )

            if "tool_use" not in usage:
                raise Exception("No tool use found in usage statistics")

            tool_id = usage["tool_use"]["id"]
            if tool_id is None:
                tool_id = usage["tool_use"]["name"]

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

            response, usage, updated_messages = llm.generate(
                event_id="test_event",
                system_prompt="Follow the user's instructions",
                messages=updated_messages,
                max_tokens=1000,
                thinking_budget=16000,
                functions=[get_weather],
            )

            # Print results
            print(f"Result: {response}")
            print("Usage statistics:")
            print(f"- Read tokens: {usage['read_tokens']}")
            print(f"- Write tokens: {usage['write_tokens']}")
            print(f"- Total cost: ${usage['total_cost']:.6f}")

            successful_models.append(model_name)
            all_results[model_name] = response

        except Exception as e:
            print(f"Error with {model_name}: {e!s}")
            failed_models.append((model_name, str(e)))

    # Print summary
    print("\n" + "=" * 70)
    print("TOOL USE TEST SUMMARY")
    print("=" * 70)
    print(f"Successful models: {len(successful_models)}/{len(models_to_test)}")
    for model in successful_models:
        print(f"  - {model}: {all_results[model]}")

    if failed_models:
        print(f"\nFailed models: {len(failed_models)}/{len(models_to_test)}")
        for model, error in failed_models:
            print(f"  - {model}: {error}")

    return all_results, successful_models, failed_models


def print_available_models():
    """Print all available models grouped by provider."""
    models_by_provider = list_available_models()

    print("\nAVAILABLE MODELS")
    print("=" * 70)

    for provider, models in models_by_provider.items():
        print(f"\n{provider} ({len(models)} models):")
        for model in models:
            print(f"  - {model}")


def main():
    """Main function to run all tests with all available models."""
    # Print available models
    print_available_models()

    # Test with a simple, publicly available image
    image_url = "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png"

    # Run all tests
    # Uncomment the tests you want to run

    models_to_test = [
        # "claude-sonnet-3.7",
        # "bedrock-claude-sonnet-3.7",
        # "gpt-4o",
        "gemini-2.0-flash",
    ]

    # Test image description - this requires models that support images
    # test_image_description(image_url, models_to_test=models_to_test)

    # Test classification
    # test_single_category_classification(models_to_test=models_to_test)

    # Test processing function
    # test_processing_function(models_to_test=models_to_test)

    # Test tool use
    test_tool_use(models_to_test=models_to_test)

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
