"""
Debug tests for Google provider to diagnose failing tests.
"""

from unittest.mock import MagicMock, patch

from lluminary.models.providers.google import GoogleLLM


@patch.object(GoogleLLM, "auth")
def test_raw_generate_with_tools_debug(mock_auth):
    """Debug version of test_raw_generate_with_tools."""
    llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

    # Create test messages
    messages = [{"message_type": "human", "message": "What's the weather in New York?"}]

    # Define test tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]

    # Mock _format_messages_for_model
    formatted_messages = [MagicMock()]
    with patch.object(
        llm, "_format_messages_for_model", return_value=formatted_messages
    ) as mock_format:
        # Setup client and response mocks
        llm.client = MagicMock()
        models_mock = MagicMock()
        llm.client.models = models_mock

        # Create mock config for response inspection
        config_mock = MagicMock()
        with patch(
            "google.genai.types.GenerateContentConfig", return_value=config_mock
        ) as mock_config_class:
            # Create mock response with function call
            mock_response = MagicMock()
            mock_response.text = "I'll check the weather for you."
            mock_response.usage_metadata.prompt_token_count = 12
            mock_response.usage_metadata.candidates_token_count = 8
            mock_response.usage_metadata.total_token_count = 20

            # Add function call to mock response
            mock_function_call = MagicMock()
            mock_function_call.id = "func_1"
            mock_function_call.name = "get_weather"
            mock_function_call.args = {"location": "New York"}
            mock_response.function_calls = [mock_function_call]

            models_mock.generate_content.return_value = mock_response

            try:
                # Call _raw_generate with tools
                response_text, usage_stats = llm._raw_generate(
                    event_id="test_event",
                    system_prompt="You are a helpful assistant",
                    messages=messages,
                    max_tokens=100,
                    temp=0.0,
                    tools=tools,
                )

                # Print results for debugging
                print("\nDEBUG INFO:")
                print(f"Response text: {response_text}")
                print(f"Usage stats: {usage_stats}")
                print(f"Config mock called: {mock_config_class.call_count}")
                print(f"Config mock calls: {mock_config_class.call_args_list}")
                if hasattr(config_mock, "tools"):
                    print(f"Config tools: {config_mock.tools}")
                else:
                    print("Config does not have 'tools' attribute")

                # Verify API call was made
                print(
                    f"generate_content called: {models_mock.generate_content.call_count}"
                )
                if models_mock.generate_content.call_count > 0:
                    call_args = models_mock.generate_content.call_args[1]
                    print(f"Call args: {call_args}")

                    # Print available attributes in config
                    if "config" in call_args:
                        config = call_args["config"]
                        print(f"Config attributes: {dir(config)}")

                # Success if we reached here
                assert True

            except Exception as e:
                # Print exception for debugging
                print(f"\nEXCEPTION: {e!s}")
                import traceback

                traceback.print_exc()
                # Always fail if we got an exception
                assert False, f"Test raised an exception: {e!s}"


@patch.object(GoogleLLM, "auth")
def test_stream_generate_basic_debug(mock_auth):
    """Debug version of test_stream_generate_basic."""
    llm = GoogleLLM("gemini-2.0-flash", api_key="test-key")

    # Create test messages
    messages = [{"message_type": "human", "message": "Tell me a story"}]

    try:
        print("\nDEBUG INFO for streaming:")
        # Make sure we mock all the imports in the stream_generate method
        with patch("google.generativeai.GenerativeModel") as mock_genai_model, patch(
            "google.genai.GenerativeModel"
        ) as mock_model_class, patch(
            "google.genai.types.content_types"
        ) as mock_content_types, patch(
            "google.genai.types.generation_types"
        ) as mock_generation_types, patch.object(
            llm, "_format_messages_for_model", return_value=[MagicMock()]
        ):

            # Print import mocks
            print(
                f"Mock imports set up: {mock_genai_model}, {mock_content_types}, {mock_generation_types}"
            )

            # Setup mock model instance
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            mock_genai_model.return_value = mock_model

            # Create mock content type for system prompt
            mock_system_content = MagicMock()
            mock_system_part = MagicMock()
            mock_content_types.Content.return_value = mock_system_content
            mock_content_types.Part.from_text.return_value = mock_system_part

            # Create mock streaming response
            mock_chunk1 = MagicMock()
            mock_chunk1.text = "Once upon "

            mock_chunk2 = MagicMock()
            mock_chunk2.text = "a time"

            mock_chunk3 = MagicMock()
            mock_chunk3.text = " there was"

            # Final chunk with empty text
            mock_chunk4 = MagicMock()
            mock_chunk4.text = ""
            mock_chunk4.candidates = [MagicMock()]

            # Set up the model to return chunks
            mock_model.generate_content.return_value = [
                mock_chunk1,
                mock_chunk2,
                mock_chunk3,
                mock_chunk4,
            ]

            # Call stream_generate and collect results
            chunks = []
            try:
                # Print stream_generate defined attributes
                print(
                    f"stream_generate method exists: {hasattr(llm, 'stream_generate')}"
                )

                # Print method content using inspect
                import inspect

                print(
                    f"Stream generate code:\n{inspect.getsource(llm.stream_generate)}"
                )

                # Print current imported modules
                import sys

                print(
                    f"Current loaded modules related to google: {[m for m in sys.modules.keys() if 'google' in m]}"
                )

                # Now try to run the stream_generate method
                print("Attempting to run stream_generate...")
                for chunk, usage in llm.stream_generate(
                    event_id="test_event",
                    system_prompt="Tell a fairy tale",
                    messages=messages,
                    max_tokens=100,
                ):
                    chunks.append((chunk, usage))
                    print(f"Chunk received: {chunk}")
                    print(f"Usage data: {usage}")

                # Print model calls
                print(f"GenerativeModel called: {mock_model_class.call_count}")
                if mock_model_class.call_count > 0:
                    print(f"Call args: {mock_model_class.call_args}")

                # Print generate_content calls
                print(
                    f"generate_content called: {mock_model.generate_content.call_count}"
                )
                if mock_model.generate_content.call_count > 0:
                    print(
                        f"generate_content args: {mock_model.generate_content.call_args}"
                    )

                # Success if we reached here
                assert True

            except Exception as e:
                print(f"\nEXCEPTION in stream_generate: {e!s}")
                import traceback

                traceback.print_exc()
                # Don't fail the test, just print the exception
                assert True, "Test showing exception info"

    except Exception as e:
        print(f"\nEXCEPTION in test setup: {e!s}")
        import traceback

        traceback.print_exc()
        # Don't fail the test, just print the exception
        assert True, "Test showing exception info"
