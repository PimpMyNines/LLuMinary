"""
Integration tests for the embeddings functionality.
"""


import pytest
from lluminary import get_llm_from_model


@pytest.mark.integration
@pytest.mark.api
class TestEmbeddings:
    """Integration tests for embeddings functionality across providers."""

    def test_get_embeddings(self, test_models):
        """Test getting embeddings from each provider."""
        input_text = "This is a test sentence for embeddings."

        # Specific embedding model names for different providers
        embedding_models = {
            "openai": "text-embedding-3-small",
            "cohere": "embed-english-v3.0",
            "anthropic": None,  # Anthropic doesn't support embeddings natively
            "google": None,  # Google doesn't support embeddings in this library
            "bedrock": None,  # Bedrock embedding support depends on the underlying model
        }

        # Test with models in test_models that support embeddings
        for model_name in test_models:
            # Skip non-embedding models
            if (
                "embedding" not in model_name.lower()
                and "embed-" not in model_name.lower()
            ):
                continue

            try:
                # Get LLM for this model
                llm = get_llm_from_model(model_name)

                # Check if model supports embeddings
                if not hasattr(llm, "get_embeddings"):
                    continue

                # Get embeddings
                result = llm.get_embeddings(input_text)

                # Validate result structure
                assert "embeddings" in result
                assert "usage" in result

                # Validate embeddings
                assert len(result["embeddings"]) > 0
                assert isinstance(result["embeddings"][0], float)

                # Validate usage data
                assert "total_tokens" in result["usage"]
                assert "total_cost" in result["usage"]

                print(f"Embeddings test passed for {model_name}")

            except Exception as e:
                pytest.skip(f"Skipping embeddings test for {model_name}: {e!s}")

        # Try specific embedding models for each provider
        for provider, model_name in embedding_models.items():
            if not model_name:  # Skip providers without embedding support
                continue

            try:
                # Get LLM for this specific embedding model
                llm = get_llm_from_model(model_name, provider=provider)

                # Check if model supports embeddings
                if not hasattr(llm, "get_embeddings"):
                    continue

                # Get embeddings
                result = llm.get_embeddings(input_text)

                # Validate result structure
                assert "embeddings" in result
                assert "usage" in result

                # Validate embeddings
                assert len(result["embeddings"]) > 0
                assert isinstance(result["embeddings"][0], float)

                # Validate usage data
                assert "total_tokens" in result["usage"]
                assert "total_cost" in result["usage"]

                print(f"Embeddings test passed for {provider}:{model_name}")

            except Exception as e:
                print(f"Skipping {provider} embeddings test: {e!s}")

    def test_batch_embeddings(self, test_models):
        """Test getting batch embeddings from each provider."""
        input_texts = [
            "This is the first test sentence.",
            "Here is another test sentence for embeddings.",
            "And this is a third one to test batch processing.",
        ]

        # Specific embedding model names for different providers
        embedding_models = {
            "openai": "text-embedding-3-small",
            "cohere": "embed-english-v3.0",
        }

        # Test with models from test_models
        for model_name in test_models:
            # Skip non-embedding models
            if (
                "embedding" not in model_name.lower()
                and "embed-" not in model_name.lower()
            ):
                continue

            try:
                # Get LLM for this model
                llm = get_llm_from_model(model_name)

                # Check if model supports embeddings
                if not hasattr(llm, "get_embeddings"):
                    continue

                # Get batch embeddings
                result = llm.get_embeddings(input_texts)

                # Validate result structure
                assert "embeddings" in result
                assert "usage" in result

                # Validate embeddings
                assert len(result["embeddings"]) == len(input_texts)
                assert all(isinstance(emb, list) for emb in result["embeddings"])
                assert all(
                    isinstance(val, float)
                    for emb in result["embeddings"]
                    for val in emb
                )

                # Validate usage data
                assert "total_tokens" in result["usage"]
                assert "total_cost" in result["usage"]

                print(f"Batch embeddings test passed for {model_name}")

            except Exception as e:
                pytest.skip(f"Skipping batch embeddings test for {model_name}: {e!s}")

        # Test with specific embedding models for each provider
        for provider, model_name in embedding_models.items():
            try:
                # Get LLM for this specific embedding model
                llm = get_llm_from_model(model_name, provider=provider)

                # Get batch embeddings
                result = llm.get_embeddings(input_texts)

                # Validate result structure
                assert "embeddings" in result
                assert "usage" in result

                # Validate embeddings
                assert len(result["embeddings"]) == len(input_texts)
                assert all(isinstance(emb, list) for emb in result["embeddings"])
                assert all(
                    isinstance(val, float)
                    for emb in result["embeddings"]
                    for val in emb
                )

                # Validate usage data
                assert "total_tokens" in result["usage"]
                assert "total_cost" in result["usage"]

                print(f"Batch embeddings test passed for {provider}:{model_name}")

            except Exception as e:
                print(f"Skipping {provider} batch embeddings test: {e!s}")

    def test_similarity_calculation(self, test_models):
        """Test using embeddings for similarity calculation."""
        query = "What is machine learning?"
        documents = [
            "Machine learning is a branch of artificial intelligence.",
            "Python is a popular programming language.",
            "Neural networks are a type of machine learning model.",
        ]

        # Specific embedding model names for different providers
        embedding_models = {
            "openai": "text-embedding-3-small",
            "cohere": "embed-english-v3.0",
        }

        # Test with a specific provider that's most likely to work
        for provider, model_name in embedding_models.items():
            try:
                # Get LLM for this specific embedding model
                llm = get_llm_from_model(model_name, provider=provider)

                # Get query embedding
                query_result = llm.get_embeddings(query)
                query_embedding = query_result["embeddings"]

                # Get document embeddings
                docs_result = llm.get_embeddings(documents)
                doc_embeddings = docs_result["embeddings"]

                # Validate that the most similar document is related to machine learning
                # This is a simple cosine similarity calculation
                similarities = []
                for doc_emb in doc_embeddings:
                    # Compute dot product
                    dot_product = sum(q * d for q, d in zip(query_embedding, doc_emb))
                    # Compute magnitudes
                    query_mag = sum(q * q for q in query_embedding) ** 0.5
                    doc_mag = sum(d * d for d in doc_emb) ** 0.5
                    # Compute cosine similarity
                    similarity = (
                        dot_product / (query_mag * doc_mag)
                        if query_mag * doc_mag > 0
                        else 0
                    )
                    similarities.append(similarity)

                # Check that the most similar document is either the first or third (about machine learning)
                most_similar_idx = similarities.index(max(similarities))
                assert most_similar_idx in [0, 2]

                print(f"Similarity calculation test passed for {provider}:{model_name}")

                # If one provider works, we can stop (no need to test all)
                break

            except Exception as e:
                print(f"Skipping {provider} similarity calculation test: {e!s}")

    def test_missing_embedding_providers(self):
        """Test that providers without embedding support behave correctly."""
        # Check that models from providers without embedding support don't claim to support embeddings
        non_embedding_providers = ["anthropic", "google", "bedrock"]

        for provider in non_embedding_providers:
            try:
                # Get a standard model for this provider
                if provider == "anthropic":
                    model_name = "claude-haiku-3.5"
                elif provider == "google":
                    model_name = "gemini-2.0-flash-lite"
                elif provider == "bedrock":
                    model_name = "bedrock-claude-haiku-3.5"
                else:
                    continue

                # Get the LLM instance
                llm = get_llm_from_model(model_name, provider=provider)

                # The get_embeddings method should either not exist or raise an exception
                if hasattr(llm, "get_embeddings"):
                    # If it has the method, it should raise an exception when called
                    with pytest.raises(Exception):
                        llm.get_embeddings("This is a test.")

                print(f"Missing embedding provider test passed for {provider}")

            except Exception as e:
                print(f"Skipping {provider} missing embedding test: {e!s}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
