#!/usr/bin/env python3
"""
Example demonstrating the streaming functionality with different providers.
"""
import os
import sys
import time
from pathlib import Path

# Add package to path for local development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lluminary import get_llm_from_model


def main():
    """Run the streaming example with different providers."""

    # Make sure you have API keys set in your environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
    else:
        # Test with OpenAI
        print("\n=== Testing streaming with OpenAI (GPT-4o-mini) ===\n")
        test_streaming_with_provider("gpt-4o-mini", openai_api_key)

    if not anthropic_api_key:
        print("Warning: ANTHROPIC_API_KEY environment variable not set.")
    else:
        # Test with Anthropic
        print("\n=== Testing streaming with Anthropic (Claude 3 Haiku) ===\n")
        test_streaming_with_provider("claude-haiku-3.5", anthropic_api_key)

    if not google_api_key:
        print("Warning: GOOGLE_API_KEY environment variable not set.")
    else:
        # Test with Google
        print("\n=== Testing streaming with Google (Gemini 2.0 Flash) ===\n")
        test_streaming_with_provider("gemini-2.0-flash", google_api_key)


def test_streaming_with_provider(model_name, api_key):
    """Test streaming with the specified provider and model."""

    # Set up a test conversation
    messages = [
        {
            "message_type": "human",
            "message": "Can you explain briefly what quantum computing is and how it differs from classical computing?",
        }
    ]

    system_prompt = "You are a helpful and concise assistant. Keep your responses informative but brief."

    try:
        # Initialize an LLM with the specified model
        llm = get_llm_from_model(model_name, api_key=api_key)
        print(f"Using model: {model_name}")

        # Create a streaming request with a callback
        print("Response is streaming in real-time:\n")

        # Define a callback function for each chunk
        def process_chunk(chunk, usage_data):
            # Print the chunk without newline
            if chunk:
                print(chunk, end="", flush=True)
            else:
                # Empty chunk signals completion
                print("\n\nStream completed.")
                print(f"Tokens used: {usage_data.get('total_tokens', 'unknown')}")
                print(f"Final cost: ${usage_data.get('total_cost', 0):.6f}")

        # Start the streaming request
        start_time = time.time()

        # Stream the response
        for _, _ in llm.stream_generate(
            event_id="stream_example",
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=300,
            temp=0.7,
            callback=process_chunk,
        ):
            # The callback handles the output, we don't need to do anything here
            pass

        # Calculate and display elapsed time
        elapsed_time = time.time() - start_time
        print(f"Total time: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Error: {e!s}")


if __name__ == "__main__":
    main()
