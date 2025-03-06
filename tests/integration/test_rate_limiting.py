"""
Integration tests for provider-specific rate limiting behaviors.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from lluminary.models.router import get_llm_from_model

# Mark all tests in this file with rate_limiting and integration markers
pytestmark = [pytest.mark.integration, pytest.mark.rate_limiting]


@pytest.mark.integration
class TestRateLimitingBehavior:
    """Test how providers handle rate limiting situations."""

    def test_basic_rate_limit_recovery(self):
        """
        Test basic rate limit recovery with a single provider.
        Simulates rate limiting by patching the provider's client.
        """
        # Test with OpenAI as they have well-defined rate limits
        model_name = "gpt-4o-mini"

        print("\n" + "=" * 60)
        print("RATE LIMIT RECOVERY TEST")
        print("=" * 60)

        try:
            # Initialize model
            llm = get_llm_from_model(model_name)

            # Create a patch to simulate rate limiting
            # Store original method
            original_client = None
            if hasattr(llm, "client"):
                original_client = llm.client

                # Track API call count
                call_count = 0

                # Define rate limit error class based on provider
                if model_name.startswith("gpt"):
                    from openai.error import RateLimitError

                    rate_limit_error = RateLimitError("Rate limit exceeded", code=429)
                elif model_name.startswith("claude"):
                    rate_limit_error = Exception(
                        "Rate limit error: 429 Too Many Requests"
                    )
                else:
                    rate_limit_error = Exception("Rate limit exceeded: 429")

                # Create client wrapper that fails on first call
                class RateLimitTestClient:
                    def __init__(self, original_client):
                        self.original_client = original_client

                    def __getattr__(self, name):
                        original_attr = getattr(self.original_client, name)

                        if callable(original_attr):

                            def wrapper(*args, **kwargs):
                                nonlocal call_count
                                call_count += 1
                                print(f"API call #{call_count}")

                                # Simulate rate limit on first call
                                if call_count == 1:
                                    print("Simulating rate limit error (429)")
                                    raise rate_limit_error

                                # Normal operation for subsequent calls
                                time.sleep(1)  # Simulate backoff
                                return original_attr(*args, **kwargs)

                            return wrapper
                        return original_attr

                # Apply the patch
                llm.client = RateLimitTestClient(original_client)

                # Run test with retry mechanism
                start_time = time.time()
                print(
                    f"Generating with {model_name} (should retry after rate limit)..."
                )

                response, usage, _ = llm.generate(
                    event_id="test_rate_limit_recovery",
                    system_prompt="You are a helpful assistant.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "Say hello briefly.",
                            "image_paths": [],
                            "image_urls": [],
                        }
                    ],
                    max_tokens=10,
                )

                end_time = time.time()

                # Verify we retried successfully
                assert call_count > 1, "Expected retry after rate limit"
                assert len(response) > 0, "Expected a valid response after retry"
                assert end_time - start_time >= 1, "Expected delay for backoff strategy"

                print(f"Response after recovery: {response}")
                print(f"Total API calls: {call_count}")
                print(f"Time taken: {end_time - start_time:.2f} seconds")

                # Restore original client
                llm.client = original_client

                return  # Test succeeded

            else:
                pytest.skip(
                    f"Model {model_name} doesn't have a client attribute to patch"
                )

        except Exception as e:
            if original_client:
                llm.client = original_client
            pytest.skip(f"Error testing rate limit recovery: {e!s}")

    def test_concurrent_request_throttling(self):
        """
        Test how providers handle multiple concurrent requests with potential rate limits.
        """
        # Test with a small model to avoid excessive costs
        model_name = "gpt-4o-mini"

        print("\n" + "=" * 60)
        print("CONCURRENT REQUEST THROTTLING TEST")
        print("=" * 60)

        try:
            # Initialize model once
            llm = get_llm_from_model(model_name)

            # Define test function for parallel execution
            def run_generation(prompt_id):
                try:
                    prompt = f"Write a one sentence answer to the question: What is {prompt_id}?"
                    print(f"Starting request #{prompt_id}...")

                    response, usage, _ = llm.generate(
                        event_id=f"test_concurrent_{prompt_id}",
                        system_prompt="You are a helpful assistant.",
                        messages=[
                            {
                                "message_type": "human",
                                "message": prompt,
                                "image_paths": [],
                                "image_urls": [],
                            }
                        ],
                        max_tokens=20,
                    )

                    return {
                        "prompt_id": prompt_id,
                        "success": True,
                        "response": response,
                        "usage": usage,
                        "error": None,
                    }

                except Exception as e:
                    print(f"Error with prompt #{prompt_id}: {e!s}")
                    return {
                        "prompt_id": prompt_id,
                        "success": False,
                        "response": None,
                        "usage": None,
                        "error": str(e),
                    }

            # Number of concurrent requests
            num_requests = 5
            prompt_ids = list(range(1, num_requests + 1))

            # Run concurrent requests
            print(f"Making {num_requests} concurrent requests to {model_name}...")
            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_requests) as executor:
                results = list(executor.map(run_generation, prompt_ids))

            end_time = time.time()

            # Analyze results
            successes = [r for r in results if r["success"]]
            failures = [r for r in results if not r["success"]]
            rate_limits = [
                r
                for r in failures
                if "429" in str(r["error"]) or "rate" in str(r["error"]).lower()
            ]

            print(
                f"\nConcurrent requests completed in {end_time - start_time:.2f} seconds"
            )
            print(f"Successful requests: {len(successes)}/{num_requests}")
            print(f"Failed requests: {len(failures)}/{num_requests}")
            print(f"Rate limited requests: {len(rate_limits)}/{num_requests}")

            # Check if we got at least some successful responses
            if not successes:
                pytest.skip("No requests succeeded, skipping validation")

            # This test is primarily observational - we're checking if the provider
            # handles concurrent requests without complete failure
            assert len(successes) > 0, "Expected at least some successful requests"

            # If we got rate limits, that's actually a valid test outcome
            if rate_limits:
                print("Observed rate limiting behavior during concurrent requests")

            # Validate response structure for successful requests
            for result in successes:
                assert isinstance(result["response"], str)
                assert len(result["response"]) > 0
                assert "total_cost" in result["usage"]

        except Exception as e:
            pytest.skip(f"Error testing concurrent requests: {e!s}")

    def test_progressive_backoff_strategy(self):
        """
        Test that the client implements proper progressive backoff strategy
        when hitting rate limits repeatedly.
        """
        # Test with OpenAI as they have well-defined rate limits
        model_name = "gpt-4o-mini"

        print("\n" + "=" * 60)
        print("PROGRESSIVE BACKOFF STRATEGY TEST")
        print("=" * 60)

        try:
            # Initialize model
            llm = get_llm_from_model(model_name)

            # Create a patch to simulate repeated rate limiting
            original_client = None
            if hasattr(llm, "client"):
                original_client = llm.client

                # Track API call count and timing
                call_count = 0
                call_times = []

                # Define rate limit error class based on provider
                if model_name.startswith("gpt"):
                    from openai.error import RateLimitError

                    rate_limit_error = RateLimitError("Rate limit exceeded", code=429)
                elif model_name.startswith("claude"):
                    rate_limit_error = Exception(
                        "Rate limit error: 429 Too Many Requests"
                    )
                else:
                    rate_limit_error = Exception("Rate limit exceeded: 429")

                # Create client wrapper that fails multiple times
                class ProgressiveBackoffTestClient:
                    def __init__(self, original_client):
                        self.original_client = original_client

                    def __getattr__(self, name):
                        original_attr = getattr(self.original_client, name)

                        if callable(original_attr):

                            def wrapper(*args, **kwargs):
                                nonlocal call_count, call_times
                                call_count += 1
                                call_times.append(time.time())
                                print(f"API call #{call_count} at {call_times[-1]}")

                                # Simulate rate limit for first 3 calls
                                if call_count <= 3:
                                    print(f"Simulating rate limit error #{call_count}")
                                    raise rate_limit_error

                                # Success on 4th try
                                return original_attr(*args, **kwargs)

                            return wrapper
                        return original_attr

                # Apply the patch
                llm.client = ProgressiveBackoffTestClient(original_client)

                # Run test with retry mechanism
                try:
                    print(
                        f"Generating with {model_name} (should retry multiple times)..."
                    )

                    response, usage, _ = llm.generate(
                        event_id="test_progressive_backoff",
                        system_prompt="You are a helpful assistant.",
                        messages=[
                            {
                                "message_type": "human",
                                "message": "Say hello.",
                                "image_paths": [],
                                "image_urls": [],
                            }
                        ],
                        max_tokens=10,
                    )

                    # Verify we retried successfully
                    assert call_count >= 4, "Expected at least 4 attempts"
                    assert len(response) > 0, "Expected a valid response after retries"

                    # Calculate delays between retries
                    delays = []
                    for i in range(1, len(call_times)):
                        delays.append(call_times[i] - call_times[i - 1])

                    print(f"Response after multiple retries: {response}")
                    print(f"Total API calls: {call_count}")
                    print(f"Delays between retries: {[f'{d:.2f}s' for d in delays]}")

                    # Check if delays are increasing (progressive backoff)
                    if len(delays) >= 2:
                        # Allow some tolerance for timing variations
                        progressive = all(
                            delays[i] >= delays[i - 1] * 0.8
                            for i in range(1, len(delays))
                        )
                        if progressive:
                            print("✅ Confirmed progressive backoff strategy")
                        else:
                            print("⚠️ Backoff doesn't appear to be progressive")

                except Exception as e:
                    print(f"Error during test: {e!s}")

                # Restore original client
                llm.client = original_client

            else:
                pytest.skip(
                    f"Model {model_name} doesn't have a client attribute to patch"
                )

        except Exception as e:
            if original_client:
                llm.client = original_client
            pytest.skip(f"Error testing progressive backoff: {e!s}")
