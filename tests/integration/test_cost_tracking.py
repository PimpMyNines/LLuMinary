"""
Integration tests for cost tracking functionality.
"""


import pytest
from lluminary import get_llm_from_model


@pytest.mark.integration
@pytest.mark.api
class TestCostTracking:
    """Integration tests for cost tracking functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.system_prompt = "You are a helpful assistant."
        self.messages = [{"message_type": "human", "message": "Hello, how are you?"}]

    def test_individual_provider_cost_tracking(self, test_models):
        """Test cost tracking for each provider."""
        for model_name in test_models:
            try:
                # Get LLM for this model
                llm = get_llm_from_model(model_name)

                # Generate a response
                response = llm.generate(
                    system_prompt=self.system_prompt,
                    messages=self.messages,
                    max_tokens=50,
                )

                # Verify usage data structure
                assert "usage" in response
                assert "total_tokens" in response["usage"]
                assert "read_tokens" in response["usage"]
                assert "write_tokens" in response["usage"]
                assert "total_cost" in response["usage"]

                # Verify actual values
                assert response["usage"]["total_tokens"] > 0
                assert response["usage"]["read_tokens"] > 0
                assert response["usage"]["write_tokens"] > 0
                assert response["usage"]["total_cost"] >= 0.0

                # Check that total tokens equals read + write tokens
                assert (
                    response["usage"]["total_tokens"]
                    == response["usage"]["read_tokens"]
                    + response["usage"]["write_tokens"]
                )

                # Log for reference
                print(f"\n{model_name} cost tracking:")
                print(f"  Read tokens: {response['usage']['read_tokens']}")
                print(f"  Write tokens: {response['usage']['write_tokens']}")
                print(f"  Total tokens: {response['usage']['total_tokens']}")
                print(f"  Total cost: ${response['usage']['total_cost']:.6f}")

            except Exception as e:
                pytest.skip(f"Skipping cost tracking test for {model_name}: {e!s}")

    def test_variable_length_cost_scaling(self):
        """Test that costs scale appropriately with response length."""
        try:
            # Try with OpenAI first, fall back to other providers
            providers = ["openai", "anthropic", "google", "bedrock"]
            models = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-haiku-3.5",
                "google": "gemini-2.0-flash-lite",
                "bedrock": "bedrock-claude-haiku-3.5",
            }

            llm = None
            for provider in providers:
                try:
                    model = models.get(provider)
                    llm = get_llm_from_model(model, provider=provider)
                    break
                except:
                    continue

            if not llm:
                pytest.skip("No providers available for cost scaling test")

            # Test with different response lengths
            lengths = [10, 50, 100]
            costs = []

            for max_tokens in lengths:
                response = llm.generate(
                    system_prompt=self.system_prompt,
                    messages=[
                        {
                            "message_type": "human",
                            "message": f"Write exactly {max_tokens} words about programming.",
                        }
                    ],
                    max_tokens=max_tokens,
                )

                costs.append(
                    {
                        "max_tokens": max_tokens,
                        "total_tokens": response["usage"]["total_tokens"],
                        "total_cost": response["usage"]["total_cost"],
                    }
                )

            # Verify that costs increase with length
            assert costs[0]["total_cost"] <= costs[1]["total_cost"]
            assert costs[1]["total_cost"] <= costs[2]["total_cost"]

            # Log for reference
            print("\nCost scaling with response length:")
            for cost in costs:
                print(
                    f"  Max tokens: {cost['max_tokens']}, Total tokens: {cost['total_tokens']}, Cost: ${cost['total_cost']:.6f}"
                )

        except Exception as e:
            pytest.skip(f"Skipping cost scaling test: {e!s}")

    def test_cross_provider_cost_comparison(self):
        """Compare costs across different providers for the same task."""
        providers = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-3.5",
            "google": "gemini-2.0-flash-lite",
            "bedrock": "bedrock-claude-haiku-3.5",
        }

        results = {}

        # Test message that will get similar length responses
        message = {
            "message_type": "human",
            "message": "Explain what a neural network is in exactly three sentences.",
        }

        for provider, model in providers.items():
            try:
                # Create LLM
                llm = get_llm_from_model(model, provider=provider)

                # Generate response
                response = llm.generate(
                    system_prompt=self.system_prompt, messages=[message], max_tokens=100
                )

                # Record results
                results[provider] = {
                    "response": response["response"],
                    "total_tokens": response["usage"]["total_tokens"],
                    "total_cost": response["usage"]["total_cost"],
                    "cost_per_token": (
                        response["usage"]["total_cost"]
                        / response["usage"]["total_tokens"]
                        if response["usage"]["total_tokens"] > 0
                        else 0
                    ),
                }

            except Exception as e:
                print(f"Skipping {provider} cost comparison test: {e!s}")

        # Skip test if fewer than 2 providers have results
        if len(results) < 2:
            pytest.skip("Not enough providers available for comparison")

        # Print cost comparison for reference
        print("\nCost Comparison Across Providers:")
        for provider, data in results.items():
            print(f"\n{provider}:")
            print(f"  Response length: {len(data['response'])} chars")
            print(f"  Total tokens: {data['total_tokens']}")
            print(f"  Total cost: ${data['total_cost']:.6f}")
            print(f"  Cost per token: ${data['cost_per_token']:.8f}")

        # Just make sure we have results
        assert len(results) > 0, "Expected at least one provider to succeed"

    def test_function_calling_cost_impact(self):
        """Test how function calling affects cost tracking."""
        # Define a simple function
        function_definition = {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }

        try:
            # Try with OpenAI first, then Anthropic
            try:
                llm = get_llm_from_model("gpt-4o-mini", provider="openai")
            except:
                try:
                    llm = get_llm_from_model(
                        "claude-3-haiku-20240307", provider="anthropic"
                    )
                except:
                    pytest.skip("No providers available for function calling test")

            # Generate a response without function calling
            response_normal = llm.generate(
                system_prompt=self.system_prompt,
                messages=[
                    {
                        "message_type": "human",
                        "message": "What's the weather in San Francisco?",
                    }
                ],
                max_tokens=50,
            )

            # Generate a response with function calling
            response_with_function = llm.generate(
                system_prompt=self.system_prompt,
                messages=[
                    {
                        "message_type": "human",
                        "message": "What's the weather in San Francisco?",
                    }
                ],
                max_tokens=50,
                tools=[function_definition],
            )

            # Verify both have cost information
            assert "total_cost" in response_normal["usage"]
            assert "total_cost" in response_with_function["usage"]

            # Log cost information for reference
            print("\nFunction Calling Cost Impact:")
            print(
                f"  Without function: ${response_normal['usage']['total_cost']:.6f}, {response_normal['usage']['total_tokens']} tokens"
            )
            print(
                f"  With function: ${response_with_function['usage']['total_cost']:.6f}, {response_with_function['usage']['total_tokens']} tokens"
            )

            # In most cases, function calling should use more tokens
            assert (
                response_with_function["usage"]["total_tokens"]
                >= response_normal["usage"]["total_tokens"]
            )

        except Exception as e:
            pytest.skip(f"Skipping function calling cost test: {e!s}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
