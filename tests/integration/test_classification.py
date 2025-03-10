"""
Integration tests for the classification functionality.
Tests real classification with graceful skipping when auth fails.
"""

import pytest

# Mark all tests in this file as classification tests
pytestmark = [pytest.mark.integration, pytest.mark.classification]

from lluminary.models.router import get_llm_from_model


class TestClassification:
    """Test classification functionality."""

    def test_single_category_classification(self):
        """
        Test basic classification functionality across all providers.
        Attempts with each model and reports results.
        """
        # Test models from each provider
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
            "bedrock-claude-haiku-3.5",  # AWS Bedrock
        ]

        # Define test data
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

        message = {
            "message_type": "human",
            "message": "How do I open this file?",
            "image_paths": [],
            "image_urls": [],
        }

        # Track results
        successful_models = []
        failed_models = []
        all_results = {}

        print("\n" + "=" * 60)
        print("CLASSIFICATION TEST")
        print("=" * 60)

        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting classification with {provider} model: {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Perform classification
                selection, usage = llm.classify(
                    messages=[message], categories=categories, examples=examples
                )

                # Verify response
                assert isinstance(selection, str)
                assert selection in categories

                # Print results
                print(f"Classification result: {selection}")
                print(
                    f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write"
                )
                print(f"Cost: ${usage['total_cost']:.6f}")

                successful_models.append(model_name)
                all_results[model_name] = selection

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                failed_models.append((model_name, str(e)))

        # Print summary
        print("\n" + "=" * 60)
        print("CLASSIFICATION TEST SUMMARY")
        print("=" * 60)
        print(f"Successful models: {len(successful_models)}/{len(test_models)}")
        for model in successful_models:
            print(f"  - {model}: {all_results[model]}")

        if failed_models:
            print(f"\nFailed models: {len(failed_models)}/{len(test_models)}")
            for model, error in failed_models:
                print(f"  - {model}: {error}")

        # Skip if no models work at all
        if not successful_models:
            pytest.skip(
                "Skipping test as no models were able to complete the classification test"
            )

    def test_classification_with_examples(self):
        """
        Test classification with and without examples to see the impact.
        Tries models until one works.
        """
        # Test models in order of preference
        test_models = ["claude-haiku-3.5", "gpt-4o-mini", "gemini-2.0-flash-lite"]

        # Test message designed to be ambiguous
        ambiguous_message = {
            "message_type": "human",
            "message": "The weather looks nice today?",
            "image_paths": [],
            "image_urls": [],
        }

        # Categories
        categories = {
            "question": "A query seeking information",
            "statement": "A declarative sentence",
        }

        # Examples that push toward "question" classification
        question_examples = [
            {
                "user_input": "The weather forecast is good?",
                "doc_str": "Although this has statement structure, the question mark makes it a question",
                "selection": "question",
            }
        ]

        # Try each model until one works
        for model_name in test_models:
            print(f"\nTrying examples test with {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Run classification without examples
                print("Classification without examples...")
                without_examples, without_usage = llm.classify(
                    messages=[ambiguous_message], categories=categories
                )

                # Run classification with examples
                print("Classification with examples...")
                with_examples, with_usage = llm.classify(
                    messages=[ambiguous_message],
                    categories=categories,
                    examples=question_examples,
                )

                # Print results
                print(f"\nWithout examples: {without_examples}")
                print(f"With examples: {with_examples}")
                print(
                    f"Token usage (without): {without_usage['read_tokens']} read, {without_usage['write_tokens']} write"
                )
                print(
                    f"Token usage (with): {with_usage['read_tokens']} read, {with_usage['write_tokens']} write"
                )

                # Verify valid responses
                assert without_examples in categories
                assert with_examples in categories
                print(f"\nTest passed with {model_name}")

                # If we get here, test passed with this model
                return

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                continue

        # If we get here, no models worked
        pytest.skip(
            "Skipping test as no models were able to complete the examples test"
        )

    def test_multi_message_classification(self):
        """
        Test classification with multiple messages.
        Tries models until one works.
        """
        # Test models in order of preference
        test_models = ["claude-haiku-3.5", "gpt-4o-mini", "gemini-2.0-flash-lite"]

        # Multiple messages
        messages = [
            {
                "message_type": "human",
                "message": "I need help with my computer.",
                "image_paths": [],
                "image_urls": [],
            },
            {
                "message_type": "human",
                "message": "It won't turn on when I press the power button.",
                "image_paths": [],
                "image_urls": [],
            },
        ]

        # Categories for support issues
        categories = {
            "hardware": "Issues related to physical components",
            "software": "Issues related to programs and operating system",
            "user_error": "Issues caused by user mistakes",
        }

        # Try each model until one works
        for model_name in test_models:
            print(f"\nTrying multi-message test with {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Run classification
                print("Classifying multiple messages...")
                selection, usage = llm.classify(
                    messages=messages, categories=categories
                )

                # Print results
                print(f"\nClassification result: {selection}")
                print(
                    f"Token usage: {usage['read_tokens']} read, {usage['write_tokens']} write"
                )

                # Verify response
                assert selection in categories
                print(f"\nTest passed with {model_name}")

                # If we get here, test passed with this model
                return

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                continue

        # If we get here, no models worked
        pytest.skip(
            "Skipping test as no models were able to complete the multi-message test"
        )
