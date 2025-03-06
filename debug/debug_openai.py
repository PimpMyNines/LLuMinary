"""
Debug OpenAI tests to identify issues
"""

from unittest.mock import MagicMock, patch

from lluminary.models.providers.openai import OpenAILLM


def debug_raw_generation():
    """Debug OpenAI raw generation test"""
    print("\nDebugging raw_generation...")

    try:
        # Create the OpenAI LLM instance with mocked auth
        with patch.object(OpenAILLM, "auth") as mock_auth:
            mock_auth.return_value = None
            openai_llm = OpenAILLM("gpt-4o", api_key="test-key")

            print(f"Created OpenAILLM with model: {openai_llm.model_name}")

            # Explicitly create a client attribute
            openai_llm.client = MagicMock()

            # Now patch the client
            with patch.object(openai_llm, "client") as mock_client:
                # Create a more detailed mock response
                message_mock = MagicMock()
                message_mock.content = "test response"

                choice_mock = MagicMock()
                choice_mock.message = message_mock

                usage_mock = MagicMock()
                usage_mock.prompt_tokens = 10
                usage_mock.completion_tokens = 5
                usage_mock.total_tokens = 15

                response_mock = MagicMock()
                response_mock.choices = [choice_mock]
                response_mock.usage = usage_mock

                mock_client.chat.completions.create.return_value = response_mock

                print("Mock client and response setup complete")

                # Call the raw generate method
                print("Calling _raw_generate...")
                try:
                    response, usage = openai_llm._raw_generate(
                        event_id="test",
                        system_prompt="You are a helpful assistant",
                        messages=[{"message_type": "human", "message": "test"}],
                        max_tokens=100,
                        temp=0,
                    )

                    print(f"Response: {response}")
                    print(f"Usage: {usage}")
                except Exception as e:
                    print(f"ERROR in _raw_generate: {e}")
                    import traceback

                    traceback.print_exc()

                # Check call arguments
                if mock_client.chat.completions.create.called:
                    print("\nAPI call arguments:")
                    (
                        call_args,
                        call_kwargs,
                    ) = mock_client.chat.completions.create.call_args
                    for key, value in call_kwargs.items():
                        print(f"  {key}: {value}")
                else:
                    print("\nAPI was NOT called")

    except Exception as e:
        print(f"ERROR: {e}")


def debug_classification():
    """Debug OpenAI classification test"""
    print("\nDebugging classification...")

    try:
        # Create the OpenAI LLM instance with mocked auth
        with patch.object(OpenAILLM, "auth") as mock_auth:
            mock_auth.return_value = None
            openai_llm = OpenAILLM("gpt-4o", api_key="test-key")

            # Explicitly create a client attribute
            openai_llm.client = MagicMock()

            # Mock the classify method
            categories = {
                "category1": "First category description",
                "category2": "Second category description",
            }
            messages = [{"message_type": "human", "message": "test message"}]

            # Set up the mock response
            message_mock = MagicMock()
            message_mock.content = "<choice>1</choice>"

            choice_mock = MagicMock()
            choice_mock.message = message_mock

            usage_mock = MagicMock()
            usage_mock.prompt_tokens = 10
            usage_mock.completion_tokens = 5
            usage_mock.total_tokens = 15

            response_mock = MagicMock()
            response_mock.choices = [choice_mock]
            response_mock.usage = usage_mock

            openai_llm.client.chat.completions.create.return_value = response_mock

            # Call the classify method
            print("Calling classify...")
            try:
                result, usage = openai_llm.classify(
                    messages=messages, categories=categories
                )

                print(f"Result: {result}")
                print(f"Usage: {usage}")
            except Exception as e:
                print(f"ERROR in classify: {e}")
                import traceback

                traceback.print_exc()

    except Exception as e:
        print(f"ERROR: {e}")


def debug_cost_tracking():
    """Debug OpenAI cost tracking test"""
    print("\nDebugging cost_tracking...")

    try:
        # Create the OpenAI LLM instance with mocked auth
        with patch.object(OpenAILLM, "auth") as mock_auth:
            mock_auth.return_value = None
            openai_llm = OpenAILLM("gpt-4o", api_key="test-key")

            # Explicitly create a client attribute
            openai_llm.client = MagicMock()

            # Now patch the client
            with patch.object(openai_llm, "client") as mock_client:
                message_mock = MagicMock()
                message_mock.content = "test response"

                choice_mock = MagicMock()
                choice_mock.message = message_mock

                usage_mock = MagicMock()
                usage_mock.prompt_tokens = 10
                usage_mock.completion_tokens = 5
                usage_mock.total_tokens = 15

                response_mock = MagicMock()
                response_mock.choices = [choice_mock]
                response_mock.usage = usage_mock

                mock_client.chat.completions.create.return_value = response_mock

                # Call the generate method
                print("Calling generate...")
                try:
                    response, usage, _ = openai_llm.generate(
                        event_id="test",
                        system_prompt="You are a helpful assistant",
                        messages=[{"message_type": "human", "message": "test"}],
                        max_tokens=100,
                    )

                    print(f"Response: {response}")
                    print(f"Usage: {usage}")
                except Exception as e:
                    print(f"ERROR in generate: {e}")
                    import traceback

                    traceback.print_exc()

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    debug_raw_generation()
    debug_cost_tracking()
    debug_classification()
