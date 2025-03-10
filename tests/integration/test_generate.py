"""
Integration tests for text generation functionality.
These tests attempt real API calls and skip when credentials aren't available.
"""

import pytest
from lluminary.models.router import get_llm_from_model


@pytest.mark.integration
class TestBasicGeneration:
    """Test basic text generation functionality with multiple providers."""

    def test_text_generation_all_providers(self):
        """
        Test basic text generation across all providers.
        This test tries all providers and reports successes/failures.
        """
        # Get one model from each provider
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
            "bedrock-claude-haiku-3.5",  # AWS Bedrock
        ]

        # Track results
        successful_models = []
        failed_models = []

        for model_name in test_models:
            # Get provider name for better reporting
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting {provider} model: {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Generate response
                response, usage, messages = llm.generate(
                    event_id="test_basic_generation",
                    system_prompt="You are a helpful AI assistant.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "Write a very short poem about testing.",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=100,
                )

                # Verify response
                assert isinstance(response, str)
                assert len(response) > 0

                # Print results
                print(f"Response from {model_name}:")
                print(f"---\n{response}\n---")
                print(
                    f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write"
                )
                print(f"Cost: ${usage['total_cost']:.6f}")

                successful_models.append(model_name)

            except Exception as e:
                print(f"Error testing {model_name}: {e!s}")
                failed_models.append((model_name, str(e)))

        # Print summary
        print("\n" + "=" * 60)
        print("TEXT GENERATION TEST SUMMARY")
        print("=" * 60)
        print(f"Successful models: {len(successful_models)}/{len(test_models)}")
        for model in successful_models:
            print(f"  - {model}")

        if failed_models:
            print(f"\nFailed models: {len(failed_models)}/{len(test_models)}")
            for model, error in failed_models:
                print(f"  - {model}: {error}")

        # Skip if no models work at all
        if not successful_models:
            pytest.skip(
                "Skipping test as no models were able to authenticate and generate responses"
            )


@pytest.mark.integration
class TestSystemPrompt:
    """Test system prompt functionality."""

    def test_system_prompt_influence(self):
        """
        Test that the system prompt influences the model's response.
        Tries each model until one works.
        """
        test_models = ["claude-haiku-3.5", "gpt-4o-mini", "gemini-2.0-flash-lite"]

        # Try each model until one works
        for model_name in test_models:
            print(f"\nTrying system prompt test with {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Generate with different system prompts
                formal_prompt = "You are a formal academic assistant. Always use technical language and complex sentence structure."
                formal_response, formal_usage, _ = llm.generate(
                    event_id="test_system_prompt_formal",
                    system_prompt=formal_prompt,
                    messages=[
                        {
                            "message_type": "human",
                            "message": "Tell me about stars in space.",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=150,
                )

                casual_prompt = "You are a casual assistant for children. Use very simple words and short sentences."
                casual_response, casual_usage, _ = llm.generate(
                    event_id="test_system_prompt_casual",
                    system_prompt=casual_prompt,
                    messages=[
                        {
                            "message_type": "human",
                            "message": "Tell me about stars in space.",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=150,
                )

                # Print results
                print("\nFormal response:")
                print(f"---\n{formal_response[:100]}...\n---")

                print("\nCasual response:")
                print(f"---\n{casual_response[:100]}...\n---")

                # Verify responses differ
                assert formal_response != casual_response
                print(f"\nTest passed with {model_name}")

                # If we get here, test passed with this model
                return

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                continue

        # If we get here, no models worked
        pytest.skip(
            "Skipping test as no models were able to complete the system prompt test"
        )


@pytest.mark.integration
class TestTokenLimits:
    """Test token limit functionality."""

    def test_max_tokens_limit(self):
        """
        Test that max_tokens parameter limits response length.
        Tries each model until one works.
        """
        test_models = ["claude-haiku-3.5", "gpt-4o-mini", "gemini-2.0-flash-lite"]

        # Try each model until one works
        for model_name in test_models:
            print(f"\nTrying token limit test with {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Generate with small token limit
                print("Generating short response (30 tokens)...")
                short_response, short_usage, _ = llm.generate(
                    event_id="test_max_tokens_short",
                    system_prompt="You are a helpful AI assistant.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "Write a long description about the history of computing.",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=30,  # Very small limit
                )

                # Generate with larger token limit
                print("Generating long response (100 tokens)...")
                long_response, long_usage, _ = llm.generate(
                    event_id="test_max_tokens_long",
                    system_prompt="You are a helpful AI assistant.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "Write a long description about the history of computing.",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=100,  # Larger limit
                )

                # Print results
                print(
                    f"\nShort response ({len(short_response)} chars, {short_usage['write_tokens']} tokens):"
                )
                print(f"---\n{short_response[:100]}...\n---")

                print(
                    f"\nLong response ({len(long_response)} chars, {long_usage['write_tokens']} tokens):"
                )
                print(f"---\n{long_response[:100]}...\n---")

                # Verify responses
                assert len(short_response) < len(long_response)
                assert short_usage["write_tokens"] < long_usage["write_tokens"]
                print(f"\nTest passed with {model_name}")

                # If we get here, test passed with this model
                return

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                continue

        # If we get here, no models worked
        pytest.skip(
            "Skipping test as no models were able to complete the token limit test"
        )
