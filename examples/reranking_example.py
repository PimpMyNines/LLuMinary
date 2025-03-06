#!/usr/bin/env python3
"""
Example demonstrating the document reranking functionality with different providers.
"""
import os
import sys
from pathlib import Path
from pprint import pprint

# Add package to path for local development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lluminary import get_llm_from_model


# We need to manually set API keys since set_provider_config is not available
def set_api_key_for_provider(provider, model_name, api_key):
    """Set API key for a provider by creating an LLM instance with the key."""
    if api_key:
        return get_llm_from_model(model_name, api_key=api_key)
    return None


def main():
    """Run the reranking example with different providers."""

    # Make sure you have API keys set in your environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    cohere_api_key = os.environ.get("COHERE_API_KEY")

    if not openai_api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
    else:
        # Test with OpenAI
        print("\n=== Testing reranking with OpenAI ===\n")
        openai_llm = set_api_key_for_provider(
            "openai", "text-embedding-3-small", openai_api_key
        )
        if openai_llm:
            test_reranking_with_llm(openai_llm)

    if not cohere_api_key:
        print("Warning: COHERE_API_KEY environment variable not set.")
    else:
        # Test with Cohere
        print("\n=== Testing reranking with Cohere ===\n")
        cohere_llm = set_api_key_for_provider(
            "cohere", "rerank-english-v3.0", cohere_api_key
        )
        if cohere_llm:
            test_reranking_with_llm(cohere_llm)


def test_reranking_with_llm(llm):
    """Test document reranking with the specified LLM instance."""

    # Sample documents about various topics
    documents = [
        "Python is a popular programming language known for its simplicity and readability.",
        "The Eiffel Tower is a famous landmark in Paris, France.",
        "Neural networks are a class of machine learning models inspired by the human brain.",
        "Cats are common household pets known for their independence and agility.",
        "The Python programming language was created by Guido van Rossum in the late 1980s.",
        "The Great Wall of China is over 13,000 miles long and was built over many centuries.",
        "Deep learning is a subset of machine learning using multiple layers of neural networks.",
        "Dogs are known as man's best friend and come in hundreds of different breeds.",
        "Python supports multiple programming paradigms, including procedural and object-oriented.",
        "The Louvre Museum in Paris is home to the Mona Lisa painting.",
    ]

    # Query related to programming
    programming_query = "What programming language is easy to learn?"

    # Query related to tourism
    tourism_query = "What are some famous places to visit in France?"

    try:
        print(f"Using model: {llm.model_name} from provider: {llm.__class__.__name__}")

        # Check if the model supports reranking
        if not llm.supports_reranking():
            print(f"Model {llm.model_name} does not support reranking")
            return

        # Rerank documents based on the programming query
        print(f"\nRanking documents for query: '{programming_query}'")
        programming_results = llm.rerank(
            query=programming_query,
            documents=documents,
            top_n=5,  # Return top 5 results
        )

        # Print ranked documents for programming query
        print("\nTop 5 documents (programming):")
        for i, (doc, score) in enumerate(
            zip(programming_results["ranked_documents"], programming_results["scores"])
        ):
            print(f"{i+1}. [{score:.4f}] {doc}")

        # Print usage info
        print("\nUsage info:")
        pprint(programming_results["usage"])

        # Rerank documents based on the tourism query
        print(f"\nRanking documents for query: '{tourism_query}'")
        tourism_results = llm.rerank(
            query=tourism_query, documents=documents, top_n=5  # Return top 5 results
        )

        # Print ranked documents for tourism query
        print("\nTop 5 documents (tourism):")
        for i, (doc, score) in enumerate(
            zip(tourism_results["ranked_documents"], tourism_results["scores"])
        ):
            print(f"{i+1}. [{score:.4f}] {doc}")

        # Print usage info
        print("\nUsage info:")
        pprint(tourism_results["usage"])

    except Exception as e:
        print(f"Error: {e!s}")


if __name__ == "__main__":
    main()
