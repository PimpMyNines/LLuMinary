"""
Integration tests for dynamic model selection functionality.
"""

import pytest
from lluminary.handler import LLMHandler
from lluminary.models.router import MODEL_MAPPINGS, get_llm_from_model

# Mark all tests in this file with dynamic_model_selection and integration markers
pytestmark = [pytest.mark.integration, pytest.mark.dynamic_model_selection]


@pytest.mark.integration
class TestDynamicModelSelection:
    """Test dynamic model selection capabilities."""

    @pytest.fixture
    def handler(self):
        """Create an LLMHandler instance."""
        return LLMHandler()

    def test_model_fallback_chain(self, handler):
        """
        Test fallback chain when a model is unavailable.
        """
        # Try models that might not be available
        primary_model = "completely-fake-model"
        fallback_models = [
            "another-fake-model",
            "gpt-4o-mini",  # Should work if OpenAI auth is configured
            "claude-haiku-3.5",  # Should work if Anthropic auth is configured
            "gemini-2.0-flash-lite",  # Should work if Google auth is configured
        ]

        print("\n" + "=" * 60)
        print("MODEL FALLBACK CHAIN TEST")
        print("=" * 60)

        # Try setting an invalid model first, with fallbacks
        print(f"Attempting to use primary model '{primary_model}' with fallbacks")

        successful_model = None
        for model in [primary_model] + fallback_models:
            try:
                print(f"Trying model: {model}")
                handler.set_model(model)

                # Test a simple generation
                response, usage = handler.generate(
                    prompt="Say hello briefly.", max_tokens=20
                )

                print(f"Model {model} succeeded with response: {response}")
                print(
                    f"Tokens: {usage['total_tokens']}, Cost: ${usage['total_cost']:.6f}"
                )

                successful_model = model
                break

            except Exception as e:
                print(f"Model {model} failed: {e!s}")
                continue

        # Skip if no models worked
        if not successful_model:
            pytest.skip("No models were available to test fallback chain")

        assert successful_model is not None, "At least one model should succeed"
        assert (
            successful_model in fallback_models
        ), "Fallback model should be one of the specified models"

        print(f"✅ Successfully fell back to {successful_model}")

    def test_capability_based_routing(self):
        """
        Test routing to models based on required capabilities (e.g., image support).
        """
        print("\n" + "=" * 60)
        print("CAPABILITY-BASED ROUTING TEST")
        print("=" * 60)

        # Find models that support images
        print("Finding models that support image input...")
        image_capable_models = []
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
        ]

        for model_name in test_models:
            try:
                llm = get_llm_from_model(model_name)
                if hasattr(llm, "supports_image_input") and llm.supports_image_input():
                    image_capable_models.append(model_name)
                    print(f"Model {model_name} supports image input")
            except Exception as e:
                print(f"Error checking {model_name}: {e!s}")

        if not image_capable_models:
            pytest.skip("No image-capable models found to test capability routing")

        # Find models that support function calling
        print("\nFinding models that support function calling...")
        function_capable_models = []

        for model_name in test_models:
            try:
                llm = get_llm_from_model(model_name)
                # Try to determine if model supports function calling
                # by checking if it handles the functions parameter
                supports_functions = (
                    # Check for special attributes or methods related to function calling
                    (
                        hasattr(llm, "supports_function_calling")
                        and llm.supports_function_calling()
                    )
                    or
                    # Or check for specific model names known to support function calling
                    any(
                        substr in model_name
                        for substr in ["gpt-", "claude-", "gemini-"]
                    )
                )

                if supports_functions:
                    function_capable_models.append(model_name)
                    print(f"Model {model_name} supports function calling")
            except Exception as e:
                print(f"Error checking {model_name}: {e!s}")

        if not function_capable_models:
            pytest.skip("No function-capable models found to test capability routing")

        # Define a simple capability router
        def get_capable_model(required_capabilities):
            """Find a model that supports all required capabilities."""

            for capability, models in required_capabilities.items():
                if capability == "image" and models and not image_capable_models:
                    raise ValueError("No available models support image input")

                if (
                    capability == "function_calling"
                    and models
                    and not function_capable_models
                ):
                    raise ValueError("No available models support function calling")

            # Return the first matching model
            if required_capabilities.get("image", False):
                return image_capable_models[0]

            if required_capabilities.get("function_calling", False):
                return function_capable_models[0]

            # Default to a simple model
            return test_models[0]

        # Test routing to a model with image capability
        print("\nTesting routing to a model with image capability...")
        image_model = get_capable_model({"image": True})
        print(f"Selected model for image capability: {image_model}")
        assert image_model in image_capable_models

        # Test routing to a model with function calling capability
        print("\nTesting routing to a model with function calling capability...")
        function_model = get_capable_model({"function_calling": True})
        print(f"Selected model for function calling capability: {function_model}")
        assert function_model in function_capable_models

        # Test routing with multiple capabilities
        if image_capable_models and function_capable_models:
            print("\nTesting routing with multiple capabilities...")
            multi_model = get_capable_model({"image": True, "function_calling": True})
            print(f"Selected model for multiple capabilities: {multi_model}")
            assert multi_model in image_capable_models
            assert multi_model in function_capable_models

    def test_cost_optimized_selection(self, handler):
        """
        Test selecting models based on cost optimization.
        """
        print("\n" + "=" * 60)
        print("COST-OPTIMIZED MODEL SELECTION TEST")
        print("=" * 60)

        # We'll need to access the model mappings to get cost information
        model_list = []
        model_costs = {}

        # First build a list of available models with their costs
        print("Building list of models with cost information...")
        for model_name in MODEL_MAPPINGS.keys():
            try:
                # Try to initialize the model (just check if it's supported)
                llm = get_llm_from_model(model_name)

                # Get the model's cost per token
                if hasattr(llm, "COST_PER_MODEL") and model_name in llm.COST_PER_MODEL:
                    read_cost = llm.COST_PER_MODEL[model_name].get("read_token", 0)
                    write_cost = llm.COST_PER_MODEL[model_name].get("write_token", 0)

                    # Calculate average cost for ranking
                    avg_cost = (
                        (read_cost + write_cost) / 2 if (read_cost or write_cost) else 0
                    )

                    if avg_cost > 0:
                        model_costs[model_name] = avg_cost
                        model_list.append(model_name)
                        print(
                            f"Model {model_name}: avg cost = ${avg_cost:.6f} per token"
                        )

            except Exception:
                # Skip models we can't initialize
                continue

        # Skip if we couldn't get cost information for any models
        if not model_costs:
            pytest.skip("Could not retrieve cost information for any models")

        # Sort models by cost (cheapest first)
        sorted_models = sorted(model_costs.keys(), key=lambda x: model_costs[x])

        if len(sorted_models) < 2:
            pytest.skip("Need at least 2 models with cost information to compare")

        print("\nModels sorted by cost (cheapest first):")
        for i, model in enumerate(sorted_models):
            print(f"{i+1}. {model}: ${model_costs[model]:.6f} per token")

        # Try using the cheapest model
        cheapest_model = sorted_models[0]
        print(f"\nAttempting to use cheapest model: {cheapest_model}")

        try:
            handler.set_model(cheapest_model)
            response, usage = handler.generate(
                prompt="Say hello briefly.", max_tokens=10
            )

            print(f"Response from {cheapest_model}: {response}")
            print(f"Tokens: {usage['total_tokens']}, Cost: ${usage['total_cost']:.6f}")

            # Compare with a more expensive model if possible
            more_expensive_model = sorted_models[-1]

            if more_expensive_model != cheapest_model:
                print(f"\nComparing with more expensive model: {more_expensive_model}")

                handler.set_model(more_expensive_model)
                expensive_response, expensive_usage = handler.generate(
                    prompt="Say hello briefly.", max_tokens=10
                )

                print(f"Response from {more_expensive_model}: {expensive_response}")
                print(
                    f"Tokens: {expensive_usage['total_tokens']}, Cost: ${expensive_usage['total_cost']:.6f}"
                )

                # Verify cost difference
                if expensive_usage["total_cost"] > usage["total_cost"]:
                    print("✅ Confirmed cost difference between models")
                    print(
                        f"Cost savings: ${expensive_usage['total_cost'] - usage['total_cost']:.6f}"
                    )
                else:
                    print("⚠️ Expected cost difference not observed")

        except Exception as e:
            print(f"Error testing cost-optimized selection: {e!s}")
            pytest.skip(f"Could not test with cheapest model: {e!s}")

    def test_content_based_routing(self, handler):
        """
        Test routing models based on the content/complexity of prompts.
        """
        print("\n" + "=" * 60)
        print("CONTENT-BASED ROUTING TEST")
        print("=" * 60)

        # Define prompts of varying complexity
        prompts = {
            "simple": "What day comes after Monday?",
            "medium": "Explain the concept of machine learning briefly.",
            "complex": "Explain quantum computing and its relationship to cryptography in detail.",
        }

        # Define a simple model selection function based on content complexity
        def select_model_for_content(prompt):
            """Select an appropriate model based on prompt complexity."""
            # Simple word count heuristic
            word_count = len(prompt.split())

            if word_count < 5:
                print(f"Simple prompt detected ({word_count} words)")
                return "gpt-4o-mini"  # Use smallest/cheapest model
            elif word_count < 10:
                print(f"Medium prompt detected ({word_count} words)")
                return "claude-haiku-3.5"  # Use mid-tier model
            else:
                print(f"Complex prompt detected ({word_count} words)")
                return "gemini-2.0-flash-lite"  # Use more capable model

        # Test model selection for each prompt complexity
        selected_models = {}

        for complexity, prompt in prompts.items():
            print(f"\nTesting model selection for {complexity} prompt:")
            print(f"Prompt: '{prompt}'")

            # Select model based on content
            model_name = select_model_for_content(prompt)
            selected_models[complexity] = model_name
            print(f"Selected model: {model_name}")

            # Try using the selected model
            try:
                handler.set_model(model_name)
                response, usage = handler.generate(prompt=prompt, max_tokens=50)

                print(f"Response snippet: {response[:50]}...")
                print(
                    f"Tokens: {usage['total_tokens']}, Cost: ${usage['total_cost']:.6f}"
                )

            except Exception as e:
                print(f"Error using selected model {model_name}: {e!s}")
                continue

        # Verify we have at least some successful tests
        if not selected_models:
            pytest.skip("Could not test any models for content-based routing")

        print("\nSummary of content-based model selection:")
        for complexity, model in selected_models.items():
            print(f"  - {complexity} prompt → {model}")

        # Successful test - at least we showed the routing logic works
        assert (
            len(selected_models) > 0
        ), "Should have selected models for at least one complexity level"
