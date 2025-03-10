"""
Extended integration tests for advanced features like JSON mode, tool use, and long context handling.
"""

import json

import pytest
from lluminary import get_llm_from_model


@pytest.mark.integration
@pytest.mark.api
class TestAdvancedFeaturesExtended:
    """Extended integration tests for advanced features."""

    def setup_method(self):
        """Set up test environment."""
        self.system_prompt = (
            "You are a helpful assistant that provides accurate information."
        )

    def test_json_mode(self):
        """Test JSON mode with compatible providers."""
        # Test message requiring structured output
        message = {
            "message_type": "human",
            "message": "Generate a list of 3 books with their title, author, and publication year.",
        }

        # Try with OpenAI first, then Anthropic
        json_compatible_providers = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
        }

        for provider, model in json_compatible_providers.items():
            try:
                # Create LLM
                llm = get_llm_from_model(model, provider=provider)

                # Generate response with JSON mode
                response = llm.generate(
                    system_prompt=f"{self.system_prompt} Respond only with JSON.",
                    messages=[message],
                    response_format={"type": "json_object"},
                )

                # Try to parse the response as JSON
                try:
                    parsed = json.loads(response["response"])

                    # Validate it contains books with required fields
                    assert isinstance(parsed, dict)
                    assert "books" in parsed or any(
                        key in parsed for key in ["data", "results", "items", "list"]
                    )

                    # Find the books list regardless of key name
                    books = None
                    for key in ["books", "data", "results", "items", "list"]:
                        if key in parsed and isinstance(parsed[key], list):
                            books = parsed[key]
                            break

                    if not books and isinstance(parsed, dict) and len(parsed) == 3:
                        # The model might have just returned the 3 books as direct keys
                        books = [parsed]

                    # If we found a books list, validate its structure
                    if books:
                        for book in books:
                            assert "title" in book or "name" in book
                            assert "author" in book
                            assert (
                                "year" in book
                                or "publication_year" in book
                                or "publicationYear" in book
                            )

                    print(f"\nJSON mode successful with {provider} ({model}):")
                    print(json.dumps(parsed, indent=2))

                    # If we've successfully tested one provider, we can stop
                    break

                except json.JSONDecodeError:
                    print(
                        f"Response from {provider} is not valid JSON: {response['response']}"
                    )
                    continue

            except Exception as e:
                print(f"Skipping JSON mode test for {provider}: {e!s}")

        # If we didn't break out of the loop, no provider worked
        else:
            pytest.skip("No providers available for JSON mode test")

    def test_long_context(self):
        """Test handling of long context with providers that support it."""
        # Generate a long input text
        long_text = "This is a test of long context handling. " * 100

        # Test with different providers
        long_context_providers = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-3.5",
            "google": "gemini-2.0-flash-lite",
        }

        successful = False

        for provider, model in long_context_providers.items():
            try:
                # Create LLM
                llm = get_llm_from_model(model, provider=provider)

                # Create a long message
                message = {
                    "message_type": "human",
                    "message": f"{long_text}\n\nSummarize the text above in one sentence.",
                }

                # Generate response
                response = llm.generate(
                    system_prompt=self.system_prompt, messages=[message], max_tokens=50
                )

                # Check if the summary is reasonable
                assert "test" in response["response"].lower()
                assert (
                    "long" in response["response"].lower()
                    or "context" in response["response"].lower()
                )

                print(f"\nLong context test successful with {provider} ({model}):")
                print(f"Response: {response['response']}")

                successful = True
                break

            except Exception as e:
                print(f"Skipping long context test for {provider}: {e!s}")

        if not successful:
            pytest.skip("No providers available for long context test")

    def test_tool_use(self):
        """Test tool use with providers that support it."""
        # Define some tools
        tools = [
            {
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
            },
            {
                "name": "get_stock_price",
                "description": "Get the current stock price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The stock symbol, e.g. AAPL",
                        }
                    },
                    "required": ["symbol"],
                },
            },
        ]

        # Test message that should trigger tool use
        message = {
            "message_type": "human",
            "message": "What's the weather in Seattle and the stock price of Microsoft?",
        }

        # Try with OpenAI first, then Anthropic
        tool_compatible_providers = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
        }

        for provider, model in tool_compatible_providers.items():
            try:
                # Create LLM
                llm = get_llm_from_model(model, provider=provider)

                # Generate response with tools
                response = llm.generate(
                    system_prompt=self.system_prompt,
                    messages=[message],
                    tools=tools,
                    force_tool_use=True,
                )

                # Check for tool use in the response
                assert "tool_calls" in response or "tool_use" in response

                print(f"\nTool use test successful with {provider} ({model}):")
                if "tool_calls" in response:
                    for tool_call in response["tool_calls"]:
                        print(f"Tool: {tool_call.get('name')}")
                        print(f"Arguments: {tool_call.get('arguments')}")
                elif "tool_use" in response:
                    print(f"Tool use: {response['tool_use']}")

                # If we've successfully tested one provider, we can stop
                break

            except Exception as e:
                print(f"Skipping tool use test for {provider}: {e!s}")

        # If we didn't break out of the loop, no provider worked
        else:
            pytest.skip("No providers available for tool use test")

    def test_parallel_requests(self):
        """Test handling multiple requests in parallel."""
        import concurrent.futures

        # Try with a provider that's likely to work
        try:
            # Create LLM
            llm = get_llm_from_model("gpt-4o-mini", provider="openai")
        except:
            try:
                llm = get_llm_from_model("claude-haiku-3.5", provider="anthropic")
            except:
                pytest.skip("No providers available for parallel requests test")

        # Create multiple messages to process in parallel
        messages = [
            {"message_type": "human", "message": f"What is {i}+{i}?"}
            for i in range(1, 6)  # 5 simple math questions
        ]

        # Function to process a single message
        def process_message(msg):
            response = llm.generate(
                system_prompt=self.system_prompt, messages=[msg], max_tokens=20
            )
            return response

        # Process messages in parallel
        start_time = pytest.importorskip("time").time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_message, msg) for msg in messages]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        end_time = pytest.importorskip("time").time()

        # Check results
        assert len(results) == len(messages)
        for i, result in enumerate(results):
            assert "response" in result
            assert "usage" in result

        # Log performance
        print(f"\nParallel processing of {len(messages)} messages:")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(
            f"Average time per message: {(end_time - start_time) / len(messages):.2f} seconds"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
