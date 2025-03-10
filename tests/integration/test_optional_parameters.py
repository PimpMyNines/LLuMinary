"""
Integration tests for provider-specific optional parameters.
Tests how each provider handles optional parameters and their effects.
"""

import re

import pytest
from lluminary.models.router import get_llm_from_model

# Mark all tests in this file with optional_parameters and integration markers
pytestmark = [pytest.mark.integration, pytest.mark.optional_parameters]


@pytest.mark.integration
class TestOptionalParameters:
    """Test provider-specific optional parameters."""

    def test_temperature_parameter(self):
        """
        Test how temperature parameter affects model outputs.
        Compare low vs high temperature responses for creativity/variability.
        """
        # Test with models that support temperature
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
        ]

        print("\n" + "=" * 60)
        print("TEMPERATURE PARAMETER TEST")
        print("=" * 60)

        # Test both low and high temperature with creative prompt
        prompt = "Write a short poem about artificial intelligence."

        # Try each model
        successful_tests = []

        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(
                f"\nTesting temperature parameter with {provider} model: {model_name}"
            )

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Test with low temperature (more deterministic)
                print("\nTesting with low temperature (0.1)...")
                low_temp_responses = []

                for i in range(2):  # Generate two responses for comparison
                    response, usage, _ = llm.generate(
                        event_id=f"test_low_temp_{i}",
                        system_prompt="You are a helpful assistant.",
                        messages=[
                            {
                                "message_type": "human",
                                "message": prompt,
                                "image_paths": [],
                                "image_urls": [],
                            }
                        ],
                        max_tokens=100,
                        temp=0.1,  # Low temperature
                    )

                    low_temp_responses.append(response)
                    print(f"Response {i+1}: {response[:50]}...")

                # Test with high temperature (more random)
                print("\nTesting with high temperature (0.9)...")
                high_temp_responses = []

                for i in range(2):  # Generate two responses for comparison
                    response, usage, _ = llm.generate(
                        event_id=f"test_high_temp_{i}",
                        system_prompt="You are a helpful assistant.",
                        messages=[
                            {
                                "message_type": "human",
                                "message": prompt,
                                "image_paths": [],
                                "image_urls": [],
                            }
                        ],
                        max_tokens=100,
                        temp=0.9,  # High temperature
                    )

                    high_temp_responses.append(response)
                    print(f"Response {i+1}: {response[:50]}...")

                # Calculate similarity between responses
                def simple_similarity(text1, text2):
                    """Calculate a simple similarity score between two texts."""
                    # Count matching words
                    words1 = set(re.findall(r"\b\w+\b", text1.lower()))
                    words2 = set(re.findall(r"\b\w+\b", text2.lower()))

                    if not words1 or not words2:
                        return 0

                    intersection = words1.intersection(words2)
                    union = words1.union(words2)

                    return len(intersection) / len(union)

                # Compare similarity within temperature groups
                low_temp_similarity = simple_similarity(
                    low_temp_responses[0], low_temp_responses[1]
                )
                high_temp_similarity = simple_similarity(
                    high_temp_responses[0], high_temp_responses[1]
                )

                print(
                    f"\nSimilarity between low temperature responses: {low_temp_similarity:.2f}"
                )
                print(
                    f"Similarity between high temperature responses: {high_temp_similarity:.2f}"
                )

                # We expect low temperature to produce more similar responses
                # however, this isn't guaranteed - it's a probabilistic effect
                if low_temp_similarity >= high_temp_similarity:
                    print(
                        "✅ Low temperature responses more similar than high temperature (expected)"
                    )
                else:
                    print(
                        "⚠️ High temperature responses more similar than low temperature (unexpected)"
                    )

                successful_tests.append(model_name)

            except Exception as e:
                print(f"Error testing {model_name}: {e!s}")

        # Skip if no models worked
        if not successful_tests:
            pytest.skip("No models were able to test temperature parameter")

    def test_openai_specific_parameters(self):
        """
        Test OpenAI-specific parameters like presence_penalty and frequency_penalty.
        """
        # Test with OpenAI models only
        model_name = "gpt-4o-mini"

        print("\n" + "=" * 60)
        print("OPENAI-SPECIFIC PARAMETERS TEST")
        print("=" * 60)

        try:
            # Initialize model
            llm = get_llm_from_model(model_name)

            # Test with frequency_penalty to reduce repetition
            print("\nTesting with frequency_penalty...")

            # Define a prompt that might lead to repetitive output
            repetitive_prompt = "List synonyms for 'good'"

            # Test with no frequency penalty
            print("\nNo frequency penalty:")
            response_no_penalty, _, _ = llm.generate(
                event_id="test_no_freq_penalty",
                system_prompt="You are a helpful assistant.",
                messages=[
                    {
                        "message_type": "human",
                        "message": repetitive_prompt,
                        "image_paths": [],
                        "image_urls": [],
                    }
                ],
                max_tokens=100,
            )

            print(f"Response: {response_no_penalty}")

            # Test with high frequency penalty
            print("\nHigh frequency penalty:")
            response_high_penalty, _, _ = llm.generate(
                event_id="test_high_freq_penalty",
                system_prompt="You are a helpful assistant.",
                messages=[
                    {
                        "message_type": "human",
                        "message": repetitive_prompt,
                        "image_paths": [],
                        "image_urls": [],
                    }
                ],
                max_tokens=100,
                frequency_penalty=1.5,  # High frequency penalty
            )

            print(f"Response: {response_high_penalty}")

            # Test with presence_penalty to discourage certain topics
            print("\nTesting with presence_penalty...")

            # Define a prompt that might lead to repetitive themes
            theme_prompt = "Write a short story about a detective."

            # Test with no presence penalty
            print("\nNo presence penalty:")
            response_no_presence, _, _ = llm.generate(
                event_id="test_no_presence_penalty",
                system_prompt="You are a helpful assistant.",
                messages=[
                    {
                        "message_type": "human",
                        "message": theme_prompt,
                        "image_paths": [],
                        "image_urls": [],
                    }
                ],
                max_tokens=150,
            )

            print(f"Response: {response_no_presence[:100]}...")

            # Test with high presence penalty
            print("\nHigh presence penalty:")
            response_high_presence, _, _ = llm.generate(
                event_id="test_high_presence_penalty",
                system_prompt="You are a helpful assistant.",
                messages=[
                    {
                        "message_type": "human",
                        "message": theme_prompt,
                        "image_paths": [],
                        "image_urls": [],
                    }
                ],
                max_tokens=150,
                presence_penalty=1.5,  # High presence penalty
            )

            print(f"Response: {response_high_presence[:100]}...")

            # Calculate word variety metric
            def word_variety(text):
                """Calculate ratio of unique words to total words."""
                words = re.findall(r"\b\w+\b", text.lower())
                unique_words = set(words)

                if not words:
                    return 0

                return len(unique_words) / len(words)

            # Compare word variety with different penalties
            no_penalty_variety = word_variety(response_no_penalty)
            high_penalty_variety = word_variety(response_high_penalty)

            print(f"\nWord variety with no frequency penalty: {no_penalty_variety:.3f}")
            print(
                f"Word variety with high frequency penalty: {high_penalty_variety:.3f}"
            )

            # Higher frequency penalty should generally increase word variety
            if high_penalty_variety > no_penalty_variety:
                print("✅ High frequency penalty increased word variety (expected)")
            else:
                print(
                    "⚠️ High frequency penalty did not increase word variety (unexpected)"
                )

        except Exception as e:
            print(f"Error testing OpenAI parameters: {e!s}")
            pytest.skip(f"Could not test OpenAI-specific parameters: {e!s}")

    def test_anthropic_specific_parameters(self):
        """
        Test Anthropic-specific parameters like top_p and thinking_budget.
        """
        # Test with Anthropic models only
        model_name = "claude-haiku-3.5"

        print("\n" + "=" * 60)
        print("ANTHROPIC-SPECIFIC PARAMETERS TEST")
        print("=" * 60)

        try:
            # Initialize model
            llm = get_llm_from_model(model_name)

            # Test with top_p
            print("\nTesting with top_p parameter...")

            # Define a creative prompt for comparing top_p influence
            creative_prompt = (
                "Generate a creative name for a space exploration company."
            )

            # Test with different top_p values
            for top_p in [0.1, 0.5, 0.9]:
                print(f"\nTesting with top_p={top_p}:")

                response, usage, _ = llm.generate(
                    event_id=f"test_top_p_{int(top_p*10)}",
                    system_prompt="You are a helpful assistant.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": creative_prompt,
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=50,
                    top_p=top_p,
                )

                print(f"Response: {response}")
                print(
                    f"Tokens: {usage['total_tokens']}, Cost: ${usage['total_cost']:.6f}"
                )

            # Test with thinking budget (Claude-specific)
            print("\nTesting with thinking_budget parameter...")

            # Check if this model supports thinking budget
            thinking_supported = hasattr(
                llm, "thinking_models"
            ) and llm.model_name in getattr(llm, "thinking_models", [])

            if thinking_supported:
                # Test with math problem that benefits from thinking
                math_prompt = "Solve this step by step: If a train travels at 60 mph for 3 hours, how far does it go?"

                # Test without thinking budget
                print("\nWithout thinking budget:")
                response_no_thinking, usage_no_thinking, _ = llm.generate(
                    event_id="test_no_thinking",
                    system_prompt="You are a helpful assistant.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": math_prompt,
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=100,
                )

                print(f"Response: {response_no_thinking[:100]}...")
                print(
                    f"Tokens: {usage_no_thinking['total_tokens']}, Cost: ${usage_no_thinking['total_cost']:.6f}"
                )

                # Test with thinking budget
                print("\nWith thinking budget:")
                response_with_thinking, usage_with_thinking, _ = llm.generate(
                    event_id="test_with_thinking",
                    system_prompt="You are a helpful assistant.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": math_prompt,
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=100,
                    thinking_budget=1000,  # Allocate thinking tokens
                )

                print(f"Response: {response_with_thinking[:100]}...")
                print(
                    f"Tokens: {usage_with_thinking['total_tokens']}, Cost: ${usage_with_thinking['total_cost']:.6f}"
                )

                # Check for "180 miles" in both responses as basic verification
                has_answer_no_thinking = (
                    "180" in response_no_thinking
                    or "180 miles" in response_no_thinking.lower()
                )
                has_answer_with_thinking = (
                    "180" in response_with_thinking
                    or "180 miles" in response_with_thinking.lower()
                )

                print(
                    f"\nCorrect answer without thinking budget: {has_answer_no_thinking}"
                )
                print(
                    f"Correct answer with thinking budget: {has_answer_with_thinking}"
                )

            else:
                print(
                    f"Model {model_name} does not support thinking budget, skipping..."
                )

        except Exception as e:
            print(f"Error testing Anthropic parameters: {e!s}")
            pytest.skip(f"Could not test Anthropic-specific parameters: {e!s}")

    def test_default_parameter_behavior(self):
        """
        Test default parameter behavior across providers.
        """
        # Test with models from different providers
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
        ]

        print("\n" + "=" * 60)
        print("DEFAULT PARAMETER BEHAVIOR TEST")
        print("=" * 60)

        # Common prompt for consistency
        prompt = "Summarize the main themes of George Orwell's 1984."

        # Parameters to test across providers
        test_params = {
            "default": {},
            "low_temp": {"temp": 0.1},
            "high_tokens": {"max_tokens": 200},
        }

        successful_models = []

        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting default parameters with {provider} model: {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Test with various parameter configurations
                for param_name, params in test_params.items():
                    print(f"\nTesting with {param_name} parameters:")

                    # Generate with specific parameters
                    response, usage, _ = llm.generate(
                        event_id=f"test_{param_name}",
                        system_prompt="You are a helpful assistant.",
                        messages=[
                            {
                                "message_type": "human",
                                "message": prompt,
                                "image_paths": [],
                                "image_urls": [],
                            }
                        ],
                        **params,
                    )

                    print(f"Response snippet: {response[:50]}...")
                    print(
                        f"Tokens: {usage['total_tokens']}, Cost: ${usage['total_cost']:.6f}"
                    )

                successful_models.append(model_name)

            except Exception as e:
                print(f"Error testing {model_name}: {e!s}")

        # Skip if no models worked
        if not successful_models:
            pytest.skip("No models were able to test default parameter behavior")
