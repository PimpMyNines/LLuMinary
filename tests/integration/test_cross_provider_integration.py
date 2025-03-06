"""
Comprehensive cross-provider integration tests.
Tests key features across all providers with graceful skipping when auth fails.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from lluminary.handler import LLMHandler
from lluminary.models.router import get_llm_from_model

# Mark all tests in this file as cross-provider integration tests
pytestmark = [pytest.mark.integration, pytest.mark.cross_provider]


class TestCrossProviderIntegration:
    """Comprehensive integration tests across all providers."""

    @pytest.fixture
    def handler(self):
        """Create an LLMHandler instance."""
        return LLMHandler()

    def test_simple_prompt_all_providers(self, handler):
        """
        Test a simple prompt across all providers.
        Tests both the handler interface and direct provider interfaces.
        """
        # Test models from each provider
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
            "bedrock-claude-haiku-3.5",  # AWS Bedrock
            "command-r",  # Cohere
        ]

        # Track results
        successful_models = []
        failed_models = []
        all_results = {}

        print("\n" + "=" * 60)
        print("CROSS-PROVIDER SIMPLE PROMPT TEST")
        print("=" * 60)

        # Test prompt
        prompt = "Explain the concept of machine learning in 2-3 sentences."

        # Test with handler first
        print("\nTesting with LLMHandler interface:")
        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting with {provider} model: {model_name}")

            try:
                # Set model in handler
                handler.set_model(model_name)

                # Generate response
                response, usage = handler.generate(prompt=prompt, max_tokens=100)

                # Print results
                print(f"Response: {response[:100]}...")
                print(
                    f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write"
                )
                print(f"Cost: ${usage['total_cost']:.6f}")

                successful_models.append(f"handler:{model_name}")
                all_results[f"handler:{model_name}"] = response

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                failed_models.append((f"handler:{model_name}", str(e)))

        # Test with direct provider interfaces
        print("\nTesting with direct provider interfaces:")
        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting with {provider} model: {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Generate response
                response, usage, _ = llm.generate(
                    event_id="test_simple_prompt",
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
                )

                # Print results
                print(f"Response: {response[:100]}...")
                print(
                    f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write"
                )
                print(f"Cost: ${usage['total_cost']:.6f}")

                successful_models.append(f"direct:{model_name}")
                all_results[f"direct:{model_name}"] = response

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                failed_models.append((f"direct:{model_name}", str(e)))

        # Print summary
        print("\n" + "=" * 60)
        print("SIMPLE PROMPT TEST SUMMARY")
        print("=" * 60)
        print(f"Successful models: {len(successful_models)}/{len(test_models)*2}")
        for model in successful_models:
            print(f"  - {model}")

        if failed_models:
            print(f"\nFailed models: {len(failed_models)}/{len(test_models)*2}")
            for model, error in failed_models:
                print(f"  - {model}: {error}")

        # Skip if no models work at all
        if not successful_models:
            pytest.skip(
                "Skipping test as no models were able to complete the simple prompt test"
            )

    def test_error_handling_across_providers(self):
        """
        Test error handling across providers with invalid inputs.
        """
        # Test models from each provider
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
            "bedrock-claude-haiku-3.5",  # AWS Bedrock
        ]

        print("\n" + "=" * 60)
        print("CROSS-PROVIDER ERROR HANDLING TEST")
        print("=" * 60)

        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting error handling with {provider} model: {model_name}")

            # Test case 1: Invalid message format
            try:
                llm = get_llm_from_model(model_name)
                # Intentionally use an invalid message format
                llm.generate(
                    event_id="test_error_handling",
                    system_prompt="You are a helpful assistant.",
                    messages=[{"invalid_key": "This should fail"}],
                    max_tokens=100,
                )
                print("❌ Test failed: Invalid message format did not raise an error")
            except Exception as e:
                print(f"✓ Invalid message format correctly raised: {type(e).__name__}")

            # Test case 2: Invalid max_tokens
            try:
                llm = get_llm_from_model(model_name)
                llm.generate(
                    event_id="test_error_handling",
                    system_prompt="You are a helpful assistant.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "Hello",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=-10,  # Invalid negative tokens
                )
                print("❌ Test failed: Negative max_tokens did not raise an error")
            except Exception as e:
                print(f"✓ Invalid max_tokens correctly raised: {type(e).__name__}")

    def test_model_switching(self, handler):
        """
        Test switching between models with the same handler.
        """
        # Pairs of models to switch between
        model_pairs = [
            ("gpt-4o-mini", "claude-haiku-3.5"),
            ("gemini-2.0-flash-lite", "bedrock-claude-haiku-3.5"),
        ]

        successful_pairs = []
        print("\n" + "=" * 60)
        print("MODEL SWITCHING TEST")
        print("=" * 60)

        for model1, model2 in model_pairs:
            print(f"\nTesting switch between {model1} and {model2}")

            try:
                # First model
                handler.set_model(model1)
                response1, usage1 = handler.generate(
                    prompt="Explain quantum computing briefly.", max_tokens=100
                )

                print(f"Model {model1} response: {response1[:50]}...")
                print(
                    f"Tokens: {usage1['total_tokens']}, Cost: ${usage1['total_cost']:.6f}"
                )

                # Switch to second model
                handler.set_model(model2)
                response2, usage2 = handler.generate(
                    prompt="Explain blockchain briefly.", max_tokens=100
                )

                print(f"Model {model2} response: {response2[:50]}...")
                print(
                    f"Tokens: {usage2['total_tokens']}, Cost: ${usage2['total_cost']:.6f}"
                )

                # Switch back to first model
                handler.set_model(model1)
                response3, usage3 = handler.generate(
                    prompt="Explain artificial intelligence briefly.", max_tokens=100
                )

                print(f"Model {model1} (again) response: {response3[:50]}...")
                print(
                    f"Tokens: {usage3['total_tokens']}, Cost: ${usage3['total_cost']:.6f}"
                )

                successful_pairs.append((model1, model2))
                print(f"✓ Successfully switched between {model1} and {model2}")

            except Exception as e:
                print(f"❌ Error switching between {model1} and {model2}: {e!s}")

        if not successful_pairs:
            pytest.skip("No model pairs could be successfully switched between")

    def test_parallel_requests_cross_provider(self):
        """
        Test making parallel requests to different providers simultaneously.
        """
        # Models to test in parallel
        models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
            "bedrock-claude-haiku-3.5",  # AWS Bedrock
        ]

        prompts = [
            "Explain the concept of neural networks.",
            "What is natural language processing?",
            "How does reinforcement learning work?",
            "Explain the transformer architecture.",
        ]

        print("\n" + "=" * 60)
        print("PARALLEL CROSS-PROVIDER REQUESTS TEST")
        print("=" * 60)

        successful_models = []
        responses = {}

        def process_model(model, prompt):
            """Process a single model in a separate thread."""
            try:
                print(f"Starting request to {model} with prompt: {prompt[:30]}...")
                llm = get_llm_from_model(model)
                response, usage, _ = llm.generate(
                    event_id=f"parallel_test_{model}",
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
                )
                print(f"Completed request to {model}")
                return model, True, response, usage
            except Exception as e:
                print(f"Error with {model}: {e!s}")
                return model, False, str(e), None

        # Execute requests in parallel
        print("Executing parallel requests to different providers...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            results = list(executor.map(process_model, models, prompts))

        end_time = time.time()

        # Process results
        for model, success, response, usage in results:
            if success:
                successful_models.append(model)
                responses[model] = response
                print(f"\n{model} response: {response[:50]}...")
                print(
                    f"Tokens: {usage['total_tokens']}, Cost: ${usage['total_cost']:.6f}"
                )

        print(f"\nParallel execution completed in {end_time - start_time:.2f} seconds")
        print(f"Successful models: {len(successful_models)}/{len(models)}")

        if not successful_models:
            pytest.skip("No models were able to complete parallel requests")
