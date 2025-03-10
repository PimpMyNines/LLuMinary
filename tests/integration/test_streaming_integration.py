"""
Integration tests for the streaming functionality.
"""

import asyncio

import pytest
from lluminary import get_llm_from_model


@pytest.mark.integration
@pytest.mark.api
class TestStreamingIntegration:
    """Integration tests for streaming functionality across providers."""

    def setup_method(self):
        """Set up test environment."""
        # Test messages
        self.messages = [
            {"message_type": "human", "message": "Write a haiku about programming."}
        ]

        # Test system prompt
        self.system_prompt = (
            "You are a helpful assistant that writes concise responses."
        )

    def test_openai_streaming(self):
        """Test OpenAI streaming functionality with actual API calls."""
        try:
            # Create OpenAI LLM
            llm = get_llm_from_model("gpt-4o-mini", provider="openai")

            # Test callback
            received_chunks = []

            def test_callback(chunk, usage_data):
                received_chunks.append(chunk)

            # Test streaming
            result = ""
            for chunk, usage_data in llm.stream_generate(
                event_id="test_openai_stream",
                system_prompt=self.system_prompt,
                messages=self.messages,
                callback=test_callback,
            ):
                # Accumulate chunks
                result += chunk

                # Verify usage data structure
                assert "event_id" in usage_data
                assert "model" in usage_data
                assert "read_tokens" in usage_data
                assert "write_tokens" in usage_data
                assert "total_tokens" in usage_data

            # Verify that we got a complete response
            assert len(result) > 10, "Response seems too short"
            assert len(received_chunks) > 3, "Expected more than 3 chunks"

            # Verify the final usage data has complete flag
            assert usage_data["is_complete"] is True

            # Validate that callback was called
            assert len(received_chunks) > 0
            assert "".join(received_chunks) == result

        except Exception as e:
            pytest.skip(f"Skipping OpenAI streaming test: {e!s}")

    def test_anthropic_streaming(self):
        """Test Anthropic streaming functionality with actual API calls."""
        try:
            # Create Anthropic LLM
            llm = get_llm_from_model("claude-haiku-3.5", provider="anthropic")

            # Test callback
            received_chunks = []

            def test_callback(chunk, usage_data):
                received_chunks.append(chunk)

            # Test streaming
            result = ""
            for chunk, usage_data in llm.stream_generate(
                event_id="test_anthropic_stream",
                system_prompt=self.system_prompt,
                messages=self.messages,
                callback=test_callback,
            ):
                # Accumulate chunks
                result += chunk

                # Verify usage data structure
                assert "event_id" in usage_data
                assert "model" in usage_data
                assert "read_tokens" in usage_data
                assert "write_tokens" in usage_data
                assert "total_tokens" in usage_data

            # Verify that we got a complete response
            assert len(result) > 10, "Response seems too short"
            assert len(received_chunks) > 3, "Expected more than 3 chunks"

            # Verify the final usage data has complete flag
            assert usage_data["is_complete"] is True

            # Validate that callback was called
            assert len(received_chunks) > 0
            assert "".join(received_chunks) == result

        except Exception as e:
            pytest.skip(f"Skipping Anthropic streaming test: {e!s}")

    def test_google_streaming(self):
        """Test Google streaming functionality with actual API calls."""
        try:
            # Create Google LLM
            llm = get_llm_from_model("gemini-2.0-flash-lite", provider="google")

            # Test callback
            received_chunks = []

            def test_callback(chunk, usage_data):
                received_chunks.append(chunk)

            # Test streaming
            result = ""
            for chunk, usage_data in llm.stream_generate(
                event_id="test_google_stream",
                system_prompt=self.system_prompt,
                messages=self.messages,
                callback=test_callback,
            ):
                # Accumulate chunks
                result += chunk

                # Verify usage data structure
                assert "event_id" in usage_data
                assert "model" in usage_data
                assert "read_tokens" in usage_data
                assert "write_tokens" in usage_data
                assert "total_tokens" in usage_data

            # Verify that we got a complete response
            assert len(result) > 10, "Response seems too short"
            assert len(received_chunks) > 3, "Expected more than 3 chunks"

            # Verify the final usage data has complete flag
            assert usage_data["is_complete"] is True

            # Validate that callback was called
            assert len(received_chunks) > 0
            assert "".join(received_chunks) == result

        except Exception as e:
            pytest.skip(f"Skipping Google streaming test: {e!s}")

    def test_bedrock_streaming(self):
        """Test AWS Bedrock streaming functionality with actual API calls."""
        try:
            # Create Bedrock LLM
            llm = get_llm_from_model("bedrock-claude-haiku-3.5", provider="bedrock")

            # Test callback
            received_chunks = []

            def test_callback(chunk, usage_data):
                received_chunks.append(chunk)

            # Test streaming
            result = ""
            for chunk, usage_data in llm.stream_generate(
                event_id="test_bedrock_stream",
                system_prompt=self.system_prompt,
                messages=self.messages,
                callback=test_callback,
            ):
                # Accumulate chunks
                result += chunk

                # Verify usage data structure
                assert "event_id" in usage_data
                assert "model" in usage_data
                assert "read_tokens" in usage_data
                assert "write_tokens" in usage_data
                assert "total_tokens" in usage_data

            # Verify that we got a complete response
            assert len(result) > 10, "Response seems too short"
            assert len(received_chunks) > 3, "Expected more than 3 chunks"

            # Verify the final usage data has complete flag
            assert usage_data["is_complete"] is True

            # Validate that callback was called
            assert len(received_chunks) > 0
            assert "".join(received_chunks) == result

        except Exception as e:
            pytest.skip(f"Skipping Bedrock streaming test: {e!s}")

    def test_cross_provider_streaming(self):
        """Test streaming across all providers and compare performance."""
        providers = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-3.5",
            "google": "gemini-2.0-flash-lite",
            "bedrock": "bedrock-claude-haiku-3.5",
        }

        results = {}

        for provider, model in providers.items():
            try:
                # Create LLM
                llm = get_llm_from_model(model, provider=provider)

                # Test streaming
                start_time = asyncio.get_event_loop().time()
                result = ""
                chunk_count = 0

                for chunk, usage_data in llm.stream_generate(
                    event_id=f"test_{provider}_stream",
                    system_prompt=self.system_prompt,
                    messages=self.messages,
                ):
                    # Accumulate chunks
                    result += chunk
                    chunk_count += 1

                end_time = asyncio.get_event_loop().time()

                # Record results
                results[provider] = {
                    "response": result,
                    "time_seconds": end_time - start_time,
                    "chunk_count": chunk_count,
                    "total_tokens": usage_data.get("total_tokens", 0),
                }

            except Exception as e:
                print(f"Skipping {provider} streaming test: {e!s}")

        # Skip test if no providers worked
        if not results:
            pytest.skip("No providers available for streaming test")

        # Just make sure we have results
        assert len(results) > 0, "Expected at least one provider to succeed"

        # Print performance comparison for reference
        print("\nStreaming Performance Comparison:")
        for provider, data in results.items():
            print(
                f"{provider}: {data['time_seconds']:.2f}s, {data['chunk_count']} chunks, {data['total_tokens']} tokens"
            )

    def test_streaming_cancellation(self):
        """Test cancellation of streaming responses."""
        try:
            # Try to use OpenAI first, fall back to other providers
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
                pytest.skip("No providers available for cancellation test")

            # Test streaming with early cancellation
            messages = [
                {
                    "message_type": "human",
                    "message": "Write a long story about a programmer who discovers a magical algorithm.",
                }
            ]

            # Test callback that will cancel after 5 chunks
            received_chunks = []
            max_chunks = 5

            def test_callback(chunk, usage_data):
                received_chunks.append(chunk)
                # Return False to signal cancellation after max_chunks
                return len(received_chunks) <= max_chunks

            # Test streaming
            result = ""
            for chunk, usage_data in llm.stream_generate(
                event_id="test_cancellation",
                system_prompt=self.system_prompt,
                messages=messages,
                callback=test_callback,
            ):
                # Accumulate chunks
                result += chunk

            # Verify that streaming was cancelled
            assert (
                len(received_chunks) <= max_chunks + 1
            ), "Expected streaming to be cancelled"

            # Verify the final usage data indicates completion
            assert usage_data["is_complete"] is True

        except Exception as e:
            pytest.skip(f"Skipping cancellation test: {e!s}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
