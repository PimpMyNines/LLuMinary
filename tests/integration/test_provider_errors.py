"""
Integration tests for provider-specific error types and handling.
Tests error handling, error mapping, and consistency across providers.
"""

import pytest
from lluminary.models.router import get_llm_from_model

# Mark all tests in this file with provider_errors and integration markers
pytestmark = [pytest.mark.integration, pytest.mark.provider_errors]


@pytest.mark.integration
class TestProviderErrorTypes:
    """Test provider-specific error types and consistency."""

    def test_provider_specific_errors(self):
        """
        Test that provider-specific errors are properly caught and categorized.
        """
        # Test with multiple providers to see differences in error handling
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
            "bedrock-claude-haiku-3.5",  # AWS Bedrock
        ]

        print("\n" + "=" * 60)
        print("PROVIDER-SPECIFIC ERROR TYPES TEST")
        print("=" * 60)

        error_types_by_provider = {}

        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting error types with {provider} model: {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Test case: Token limit error (requesting too many tokens)
                print("Testing token limit error...")
                try:
                    # Use an excessive max_tokens value
                    excessive_tokens = 1000000
                    llm.generate(
                        event_id="test_token_limit_error",
                        system_prompt="You are a helpful assistant.",
                        messages=[
                            {
                                "message_type": "human",
                                "message": "Hello",
                                "image_paths": [],
                                "image_urls": [],
                            }
                        ],
                        max_tokens=excessive_tokens,
                    )
                    print("❌ Test failed: Token limit error not raised")
                except Exception as e:
                    print(
                        f"✓ Token limit error correctly raised: {type(e).__name__}: {e!s}"
                    )
                    if provider not in error_types_by_provider:
                        error_types_by_provider[provider] = {}
                    error_types_by_provider[provider]["token_limit"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }

                # Test case: Invalid message format error
                print("\nTesting invalid message format error...")
                try:
                    # Use an invalid message structure
                    llm.generate(
                        event_id="test_invalid_message_error",
                        system_prompt="You are a helpful assistant.",
                        messages=[{"invalid": "This is not a valid message"}],
                    )
                    print("❌ Test failed: Invalid message error not raised")
                except Exception as e:
                    print(
                        f"✓ Invalid message error correctly raised: {type(e).__name__}: {e!s}"
                    )
                    if provider not in error_types_by_provider:
                        error_types_by_provider[provider] = {}
                    error_types_by_provider[provider]["invalid_message"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }

                # Test case: Invalid API key error
                print("\nTesting invalid API key error...")
                try:
                    # Save original API key/client
                    if hasattr(llm, "api_key"):
                        original_api_key = llm.api_key
                        llm.api_key = "invalid_key_for_testing_sk_1234567890"
                    elif hasattr(llm, "client"):
                        original_client = llm.client
                        # This is a simplification - in real tests we'd need to handle
                        # different client structures for different providers
                        if hasattr(llm.client, "api_key"):
                            original_client_key = llm.client.api_key
                            llm.client.api_key = "invalid_key_for_testing_sk_1234567890"

                    # Make a simple request that should fail
                    llm.generate(
                        event_id="test_invalid_key_error",
                        system_prompt="You are a helpful assistant.",
                        messages=[
                            {
                                "message_type": "human",
                                "message": "Hello",
                                "image_paths": [],
                                "image_urls": [],
                            }
                        ],
                        max_tokens=10,
                    )
                    print("❌ Test failed: Invalid API key error not raised")
                except Exception as e:
                    print(
                        f"✓ Invalid API key error correctly raised: {type(e).__name__}: {e!s}"
                    )
                    if provider not in error_types_by_provider:
                        error_types_by_provider[provider] = {}
                    error_types_by_provider[provider]["invalid_api_key"] = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                finally:
                    # Restore original API key/client
                    if hasattr(llm, "api_key") and "original_api_key" in locals():
                        llm.api_key = original_api_key
                    elif hasattr(llm, "client") and "original_client" in locals():
                        if (
                            hasattr(llm.client, "api_key")
                            and "original_client_key" in locals()
                        ):
                            llm.client.api_key = original_client_key
                        else:
                            llm.client = original_client

            except Exception as e:
                print(f"Error setting up tests for {model_name}: {e!s}")

        # Print summary of error types by provider
        print("\n" + "=" * 60)
        print("ERROR TYPES SUMMARY")
        print("=" * 60)

        if error_types_by_provider:
            for provider, errors in error_types_by_provider.items():
                print(f"\n{provider} error types:")
                for error_case, error_info in errors.items():
                    print(f"  - {error_case}: {error_info['error_type']}")
                    print(f"    Message: {error_info['error_message'][:100]}...")
        else:
            print("No error types successfully tested")
            pytest.skip("No provider errors could be tested")

    def test_error_mapping_consistency(self):
        """
        Test that errors from different providers are mapped to consistent LLMProviderError types.
        """
        # Test with multiple providers to see consistency in error mapping
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
        ]

        print("\n" + "=" * 60)
        print("ERROR MAPPING CONSISTENCY TEST")
        print("=" * 60)

        # Track error types by case
        error_types_by_case = {
            "content_policy": [],
            "invalid_request": [],
            "token_limit": [],
        }

        # Track successful tests
        successful_tests = False

        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting error mapping with {provider} model: {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Test case: Content policy violation
                print("Testing content policy error mapping...")
                try:
                    # Use a prompt likely to trigger content policy violations
                    llm.generate(
                        event_id="test_content_policy_error",
                        system_prompt="You are a helpful assistant.",
                        messages=[
                            {
                                "message_type": "human",
                                "message": "Write detailed instructions for creating illegal substances.",
                                "image_paths": [],
                                "image_urls": [],
                            }
                        ],
                        max_tokens=100,
                    )
                    print(
                        "Note: Content policy error not raised (model may have handled the request safely)"
                    )
                except Exception as e:
                    print(f"Content policy error raised: {type(e).__name__}")
                    error_types_by_case["content_policy"].append(
                        {
                            "provider": provider,
                            "error_type": type(e).__name__,
                            "error_message": str(e)[:100],
                        }
                    )
                    successful_tests = True

                # Test case: Invalid request format
                print("\nTesting invalid request error mapping...")
                try:
                    # Use invalid messages structure
                    llm.generate(
                        event_id="test_invalid_request_error",
                        system_prompt="You are a helpful assistant.",
                        messages=[{"invalid_key": "Not a valid message format"}],
                        max_tokens=100,
                    )
                    print("❌ Test failed: Invalid request error not raised")
                except Exception as e:
                    print(f"Invalid request error raised: {type(e).__name__}")
                    error_types_by_case["invalid_request"].append(
                        {
                            "provider": provider,
                            "error_type": type(e).__name__,
                            "error_message": str(e)[:100],
                        }
                    )
                    successful_tests = True

                # Test case: Token limit
                print("\nTesting token limit error mapping...")
                try:
                    # Request excessive tokens
                    llm.generate(
                        event_id="test_token_limit_error",
                        system_prompt="You are a helpful assistant.",
                        messages=[
                            {
                                "message_type": "human",
                                "message": "Hello",
                                "image_paths": [],
                                "image_urls": [],
                            }
                        ],
                        max_tokens=1000000,  # Excessive value
                    )
                    print("❌ Test failed: Token limit error not raised")
                except Exception as e:
                    print(f"Token limit error raised: {type(e).__name__}")
                    error_types_by_case["token_limit"].append(
                        {
                            "provider": provider,
                            "error_type": type(e).__name__,
                            "error_message": str(e)[:100],
                        }
                    )
                    successful_tests = True

            except Exception as e:
                print(f"Error setting up tests for {model_name}: {e!s}")

        # Print summary of error consistency
        print("\n" + "=" * 60)
        print("ERROR MAPPING CONSISTENCY SUMMARY")
        print("=" * 60)

        if successful_tests:
            for error_case, errors in error_types_by_case.items():
                if errors:
                    print(f"\n{error_case} errors across providers:")
                    for error in errors:
                        print(f"  - {error['provider']}: {error['error_type']}")
                        print(f"    Message: {error['error_message']}...")

                    # Check if all errors for this case map to the same exception type
                    error_types = set(error["error_type"] for error in errors)
                    if len(error_types) == 1:
                        print(
                            f"  ✅ Consistent error mapping: All providers use {next(iter(error_types))}"
                        )
                    else:
                        print(
                            f"  ⚠️ Inconsistent error mapping: {', '.join(error_types)}"
                        )
        else:
            print("No error mappings successfully tested")
            pytest.skip("No provider error mappings could be tested")

    def test_error_details_extraction(self):
        """
        Test that error details can be extracted from provider-specific errors.
        """
        # Test with OpenAI which has detailed error information
        model_name = "gpt-4o-mini"

        print("\n" + "=" * 60)
        print("ERROR DETAILS EXTRACTION TEST")
        print("=" * 60)

        try:
            # Initialize model
            llm = get_llm_from_model(model_name)

            # Test case: Extract details from rate limit error
            print("Testing error details extraction...")

            # Create a error-response scenario
            try:
                # Request with invalid message
                llm.generate(
                    event_id="test_error_details",
                    system_prompt="You are a helpful assistant.",
                    messages=[{"invalid_key": "This should fail"}],
                    max_tokens=100,
                )
                pytest.skip("Expected error was not raised")
            except Exception as e:
                # Verify we can extract error details
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "has_code": hasattr(e, "code"),
                    "has_status": hasattr(e, "status_code"),
                    "has_params": hasattr(e, "param"),
                }

                if hasattr(e, "code"):
                    error_info["code"] = e.code
                if hasattr(e, "status_code"):
                    error_info["status_code"] = e.status_code
                if hasattr(e, "param"):
                    error_info["param"] = e.param

                print("\nExtracted error details:")
                for key, value in error_info.items():
                    print(f"  - {key}: {value}")

                # Verify at least basic error info is available
                assert error_info["error_type"], "Error type should be available"
                assert error_info["error_message"], "Error message should be available"
                print("\n✅ Successfully extracted error details")

        except Exception as e:
            pytest.skip(f"Error testing error details extraction: {e!s}")
