"""
Document reranking example for LLMHandler package.
This example demonstrates how to use the document reranking functionality
with different providers.
"""

import os
import time

import numpy as np
from lluminary import get_llm_from_model


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)


def simple_embedding_search(query_embedding, document_embeddings, top_n=None):
    """Perform a simple embedding-based search."""
    similarities = [
        cosine_similarity(query_embedding, doc_embedding)
        for doc_embedding in document_embeddings
    ]

    # Get indices sorted by similarity (highest first)
    ranked_indices = np.argsort(similarities)[::-1]

    if top_n:
        ranked_indices = ranked_indices[:top_n]

    return ranked_indices, [similarities[i] for i in ranked_indices]


def openai_reranking_example():
    """Example of using OpenAI for document reranking."""
    print("\n=== OPENAI RERANKING EXAMPLE ===")

    # Initialize OpenAI for embeddings and reranking
    llm = get_llm_from_model("text-embedding-3-small")

    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require substantial training data.",
        "Natural language processing has advanced significantly in recent years.",
        "Deep learning is a subset of machine learning based on neural networks.",
        "Python is a popular programming language for data science and AI.",
        "Transformers have revolutionized natural language processing tasks.",
        "Convolutional neural networks are commonly used for image recognition.",
        "Reinforcement learning enables agents to learn through trial and error.",
        "Data preprocessing is a crucial step in any machine learning pipeline.",
        "Transfer learning leverages knowledge from one task to improve another.",
    ]

    # Query to search for
    query = "advances in natural language processing"
    print(f"Query: '{query}'\n")

    # STEP 1: First, let's do a simple embedding-based search
    print("STEP 1: Embedding-based search")
    print("------------------------------")

    # Get embeddings for query and documents
    start_time = time.time()
    query_embedding, _ = llm.embed(texts=[query])
    document_embeddings, embedding_usage = llm.embed(texts=documents)

    # Perform simple embedding search
    ranked_indices, scores = simple_embedding_search(
        query_embedding[0], document_embeddings, top_n=5
    )

    # Display results
    print("Top 5 documents by embedding similarity:")
    for i, (idx, score) in enumerate(zip(ranked_indices, scores)):
        print(f"{i+1}. [{score:.4f}] {documents[idx]}")

    embedding_time = time.time() - start_time
    print(f"\nEmbedding search time: {embedding_time:.2f} seconds")
    print(f"Tokens used: {embedding_usage['tokens']}")
    print(f"Cost: ${embedding_usage['cost']:.6f}")

    # STEP 2: Now, let's use the reranking functionality
    print("\nSTEP 2: LLM-based reranking")
    print("-------------------------")

    # Rerank all documents
    start_time = time.time()
    results = llm.rerank(
        query=query,
        documents=documents,
        top_n=5,  # Return only top 5 results
        return_scores=True,
    )

    # Display results
    print("Top 5 documents after reranking:")
    for i, (doc, score) in enumerate(
        zip(results["ranked_documents"], results["scores"])
    ):
        print(f"{i+1}. [{score:.4f}] {doc}")

    reranking_time = time.time() - start_time
    print(f"\nReranking time: {reranking_time:.2f} seconds")
    print(f"Tokens used: {results['usage']['total_tokens']}")
    print(f"Cost: ${results['usage']['total_cost']:.6f}")

    # STEP 3: Compare the results
    print("\nSTEP 3: Comparison")
    print("----------------")

    # Check for differences in the top results
    embedding_top = [documents[idx] for idx in ranked_indices[:3]]
    reranking_top = results["ranked_documents"][:3]

    print("Top 3 by embedding similarity:")
    for i, doc in enumerate(embedding_top):
        print(f"{i+1}. {doc}")

    print("\nTop 3 after reranking:")
    for i, doc in enumerate(reranking_top):
        print(f"{i+1}. {doc}")

    # Calculate overlap
    overlap = len(set(embedding_top) & set(reranking_top))
    print(f"\nOverlap between methods: {overlap}/3 documents")

    print("\nConclusion:")
    if overlap == 3:
        print("Both methods returned identical top results.")
    elif overlap > 0:
        print(f"The methods partially agree, with {overlap} documents in common.")
    else:
        print("The methods returned completely different top results.")

    print(
        "Reranking provides a more semantically nuanced ordering based on full document understanding."
    )

    return results


def cohere_reranking_example():
    """Example of using Cohere for document reranking."""
    # Skip if no Cohere API key
    if not os.environ.get("COHERE_API_KEY"):
        print("\n=== COHERE RERANKING EXAMPLE ===")
        print("Skipping Cohere example (no API key available).")
        print("Please set the COHERE_API_KEY environment variable to run this example.")
        return None

    print("\n=== COHERE RERANKING EXAMPLE ===")

    # Initialize Cohere
    llm = get_llm_from_model("cohere-command")

    # Use the same documents and query as in the OpenAI example
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require substantial training data.",
        "Natural language processing has advanced significantly in recent years.",
        "Deep learning is a subset of machine learning based on neural networks.",
        "Python is a popular programming language for data science and AI.",
        "Transformers have revolutionized natural language processing tasks.",
        "Convolutional neural networks are commonly used for image recognition.",
        "Reinforcement learning enables agents to learn through trial and error.",
        "Data preprocessing is a crucial step in any machine learning pipeline.",
        "Transfer learning leverages knowledge from one task to improve another.",
    ]

    query = "advances in natural language processing"
    print(f"Query: '{query}'\n")

    # Use Cohere's rerank functionality
    start_time = time.time()
    results = llm.rerank(
        query=query,
        documents=documents,
        top_n=5,
        return_scores=True,
        model="rerank-english-v2.0",  # Explicitly specify reranking model
    )

    # Display results
    print("Top 5 documents after Cohere reranking:")
    for i, (doc, score) in enumerate(
        zip(results["ranked_documents"], results["scores"])
    ):
        print(f"{i+1}. [{score:.4f}] {doc}")

    reranking_time = time.time() - start_time
    print(f"\nReranking time: {reranking_time:.2f} seconds")
    print(f"Tokens used: {results['usage']['total_tokens']}")
    print(f"Cost: ${results['usage']['total_cost']:.6f}")

    return results


def compare_providers(openai_results, cohere_results):
    """Compare reranking results from different providers."""
    if not cohere_results:
        print("\n=== PROVIDER COMPARISON ===")
        print("Cannot compare providers (Cohere results not available).")
        return

    print("\n=== PROVIDER COMPARISON ===")

    # Get top 3 documents from each provider
    openai_top = openai_results["ranked_documents"][:3]
    cohere_top = cohere_results["ranked_documents"][:3]

    print("Top 3 from OpenAI:")
    for i, doc in enumerate(openai_top):
        print(f"{i+1}. {doc}")

    print("\nTop 3 from Cohere:")
    for i, doc in enumerate(cohere_top):
        print(f"{i+1}. {doc}")

    # Calculate overlap
    overlap = len(set(openai_top) & set(cohere_top))
    print(f"\nOverlap between providers: {overlap}/3 documents")

    if overlap == 3:
        print("Providers returned identical top results.")
    elif overlap > 0:
        print(f"Providers partially agree, with {overlap} documents in common.")
    else:
        print("Providers returned completely different top results.")

    print(
        "\nThis demonstrates how different providers might prioritize different aspects of relevance."
    )


def real_world_example():
    """A more realistic document reranking example with longer texts."""
    print("\n=== REAL-WORLD RERANKING EXAMPLE ===")

    # Initialize LLM
    llm = get_llm_from_model("text-embedding-3-small")

    # More realistic documents (e.g., article snippets or research abstracts)
    documents = [
        """Transformer models have significantly advanced the field of natural language
        processing. Introduced in the 2017 paper 'Attention Is All You Need', transformers
        use a self-attention mechanism that allows them to process sequential data in parallel,
        making them much faster to train than previous recurrent neural network models.""",
        """Data preprocessing is a critical step in any machine learning pipeline. It involves
        cleaning, normalizing, and transforming raw data into a format suitable for model training.
        Effective preprocessing can significantly improve model performance and training efficiency.""",
        """The recent advances in large language models (LLMs) have revolutionized natural
        language processing. Models like GPT-4, Claude, and Gemini can understand and generate
        human-like text, translate languages, write different kinds of creative content, and
        answer questions in an informative way.""",
        """Retrieval-Augmented Generation (RAG) combines information retrieval with text
        generation to enhance the accuracy and reliability of language models. By retrieving
        relevant documents from a knowledge base before generating a response, RAG helps ground
        model outputs in factual information and reduce hallucinations.""",
        """Computer vision technologies have made substantial progress in recent years,
        particularly with the development of deep convolutional neural networks. These
        systems can now recognize objects, detect faces, and even understand complex scenes
        with human-level accuracy in many domains.""",
    ]

    # More specific query
    query = "How have transformer models improved NLP applications?"
    print(f"Query: '{query}'\n")

    # Perform reranking
    results = llm.rerank(query=query, documents=documents, return_scores=True)

    # Display results
    print("Documents ranked by relevance:")
    for i, (doc, score) in enumerate(
        zip(results["ranked_documents"], results["scores"])
    ):
        # Print truncated version for readability
        truncated_doc = doc[:100] + "..." if len(doc) > 100 else doc
        print(f"{i+1}. [{score:.4f}] {truncated_doc}")

    print("\nMost relevant document (full text):")
    print(results["ranked_documents"][0])

    print(f"\nToken usage: {results['usage']['total_tokens']}")
    print(f"Cost: ${results['usage']['total_cost']:.6f}")


if __name__ == "__main__":
    # Run the examples
    try:
        openai_results = openai_reranking_example()
    except Exception as e:
        print(f"OpenAI reranking example failed: {e!s}")
        openai_results = None

    try:
        cohere_results = cohere_reranking_example()
    except Exception as e:
        print(f"Cohere reranking example failed: {e!s}")
        cohere_results = None

    try:
        if openai_results and cohere_results:
            compare_providers(openai_results, cohere_results)
    except Exception as e:
        print(f"Provider comparison failed: {e!s}")

    try:
        real_world_example()
    except Exception as e:
        print(f"Real-world example failed: {e!s}")
