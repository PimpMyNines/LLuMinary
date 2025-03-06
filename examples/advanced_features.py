"""
Advanced features example for LLMHandler package.
Demonstrates embeddings, streaming, and custom provider registration.
"""

import os
import time

from lluminary import get_llm_from_model
from lluminary.models.router import register_model


def embedding_example():
    """Example of using embeddings functionality."""
    # Initialize OpenAI for embeddings
    llm = get_llm_from_model("gpt-4o-mini")  # Any OpenAI model will work

    print("\n=== EMBEDDING EXAMPLE ===")

    # List of texts to embed
    texts = [
        "The cat sat on the mat.",
        "The dog played in the yard.",
        "Artificial intelligence transforms businesses.",
        "Machine learning algorithms require data.",
    ]

    print(f"Getting embeddings for {len(texts)} texts...")

    # Get embeddings
    embeddings, usage = llm.embed(texts=texts)

    print(f"Received {len(embeddings)} embeddings.")
    print(f"First embedding vector has {len(embeddings[0])} dimensions.")
    print(f"Tokens used: {usage['tokens']}")
    print(f"Cost: ${usage['cost']:.6f}")

    # Calculate similarity between first two embeddings (simple cosine)
    def cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (magnitude1 * magnitude2)

    # Compare pet sentences vs AI sentences
    similarity1 = cosine_similarity(embeddings[0], embeddings[1])  # cat vs dog
    similarity2 = cosine_similarity(embeddings[2], embeddings[3])  # AI vs ML
    similarity3 = cosine_similarity(embeddings[0], embeddings[2])  # cat vs AI

    print("\nSimilarity scores:")
    print(f"Cat vs Dog: {similarity1:.4f}")
    print(f"AI vs ML: {similarity2:.4f}")
    print(f"Cat vs AI: {similarity3:.4f}")

    return embeddings, usage


def streaming_example():
    """Example of using streaming responses."""
    # Initialize a model that supports streaming
    llm = get_llm_from_model("gpt-4o-mini")

    print("\n=== STREAMING EXAMPLE ===")
    print("Generating a long response with streaming...\n")

    # Callback to process chunks as they arrive
    def chunk_callback(chunk, usage_data):
        if chunk:  # Empty chunk signals completion
            print(chunk, end="", flush=True)
        else:
            print("\n\nStream completed.")
            print(f"Total tokens: {usage_data['total_tokens']}")
            print(f"Cost: ${usage_data.get('total_cost', 0):.6f}")

    # Create message
    messages = [
        {
            "message_type": "human",
            "message": "Write a short story about a robot learning to paint. Make it exactly 5 paragraphs long.",
            "image_paths": [],
            "image_urls": [],
        }
    ]

    # Start streaming
    print("Response: ", end="", flush=True)
    start_time = time.time()

    # Get streaming response
    for chunk, usage in llm.stream_generate(
        event_id="streaming_example",
        system_prompt="You are a creative writing assistant.",
        messages=messages,
        max_tokens=1000,
        temp=0.7,
        callback=chunk_callback,
    ):
        # We're using the callback, but you could also process chunks here
        pass

    elapsed = time.time() - start_time
    print(f"\nTime elapsed: {elapsed:.2f} seconds")


def custom_provider_registration_example():
    """Example of registering a custom model."""
    # First, check if Cohere provider is available (depends on implementation)
    print("\n=== CUSTOM PROVIDER EXAMPLE ===")

    try:
        # Register a new model alias for Cohere
        register_model(
            friendly_name="cohere-command-custom",
            provider_name="cohere",
            model_id="command",
        )

        # Try to get the model
        llm = get_llm_from_model("cohere-command-custom")

        print("Successfully registered and initialized custom Cohere model.")

        # Skip actual API calls if no Cohere key is available
        if os.environ.get("COHERE_API_KEY"):
            # Test a simple generation
            response, usage, _ = llm.generate(
                event_id="cohere_test",
                system_prompt="You are a helpful assistant.",
                messages=[
                    {
                        "message_type": "human",
                        "message": "What's your name and who created you?",
                        "image_paths": [],
                        "image_urls": [],
                    }
                ],
                max_tokens=100,
            )

            print(f"Response from Cohere: {response}")
            print(f"Tokens used: {usage['total_tokens']}")
            print(f"Cost: ${usage['total_cost']:.6f}")
        else:
            print("Skipping Cohere API call (no API key available).")

    except (ImportError, ValueError) as e:
        print(f"Could not use Cohere provider: {e!s}")
        print("Cohere provider might not be implemented or registered.")


if __name__ == "__main__":
    # Run each example
    try:
        embedding_example()
    except Exception as e:
        print(f"Embedding example failed: {e!s}")

    try:
        streaming_example()
    except Exception as e:
        print(f"Streaming example failed: {e!s}")

    try:
        custom_provider_registration_example()
    except Exception as e:
        print(f"Custom provider example failed: {e!s}")
