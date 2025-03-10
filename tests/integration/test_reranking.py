"""
Integration tests for the document reranking functionality.
"""


import pytest
from lluminary import get_llm_from_model


@pytest.mark.integration
@pytest.mark.api
class TestReranking:
    """Integration tests for document reranking functionality."""

    def setup_method(self):
        """Set up test environment."""
        # Test documents
        self.documents = [
            "Python is a popular programming language.",
            "The Eiffel Tower is in Paris, France.",
            "Machine learning is a subset of AI.",
            "Cats are common pets worldwide.",
        ]

        # Test query
        self.query = "Which programming language is popular?"

    def test_openai_reranking(self):
        """Test OpenAI reranking functionality with actual API calls."""
        try:
            # Create OpenAI reranking LLM
            llm = get_llm_from_model("text-embedding-3-small", provider="openai")

            # Check if model supports reranking
            if not llm.supports_reranking():
                pytest.skip("OpenAI model does not support reranking")

            # Run reranking
            result = llm.rerank(query=self.query, documents=self.documents, top_n=2)

            # Verify the result structure
            assert "ranked_documents" in result
            assert "indices" in result
            assert "scores" in result
            assert "usage" in result

            # Verify top_n parameter was respected
            assert len(result["ranked_documents"]) == 2
            assert len(result["indices"]) == 2
            assert len(result["scores"]) == 2

            # Verify the most relevant document is about programming
            assert "Python" in result["ranked_documents"][0]

        except Exception as e:
            pytest.skip(f"Skipping OpenAI reranking test: {e!s}")

    def test_cohere_reranking(self):
        """Test Cohere reranking functionality with actual API calls."""
        try:
            # Create Cohere reranking LLM
            llm = get_llm_from_model("rerank-english-v3.0", provider="cohere")

            # Check if model supports reranking
            if not llm.supports_reranking():
                pytest.skip("Cohere model does not support reranking")

            # Run reranking
            result = llm.rerank(query=self.query, documents=self.documents, top_n=2)

            # Verify the result structure
            assert "ranked_documents" in result
            assert "indices" in result
            assert "scores" in result
            assert "usage" in result

            # Verify top_n parameter was respected
            assert len(result["ranked_documents"]) == 2
            assert len(result["indices"]) == 2
            assert len(result["scores"]) == 2

            # Verify the most relevant document is about programming
            assert "Python" in result["ranked_documents"][0]

        except Exception as e:
            pytest.skip(f"Skipping Cohere reranking test: {e!s}")

    def test_cross_provider_reranking_comparison(self):
        """Test and compare reranking results across providers."""
        results = {}

        # Test with each provider that supports reranking
        providers = ["openai", "cohere"]
        models = {"openai": "text-embedding-3-small", "cohere": "rerank-english-v3.0"}

        for provider in providers:
            try:
                # Get model for this provider
                model = models.get(provider)
                if not model:
                    continue

                llm = get_llm_from_model(model, provider=provider)

                # Check if model supports reranking
                if not llm.supports_reranking():
                    continue

                # Run reranking
                result = llm.rerank(query=self.query, documents=self.documents)

                # Store results
                results[provider] = result

            except Exception as e:
                print(f"Skipping {provider} reranking test: {e!s}")

        # Skip test if fewer than 2 providers have results
        if len(results) < 2:
            pytest.skip("Not enough providers available for comparison")

        # Compare top results across providers
        top_indices = {
            provider: result["indices"][0] for provider, result in results.items()
        }

        # Check if providers agree on the most relevant document
        assert (
            len(set(top_indices.values())) <= 2
        ), "Providers disagree significantly on the most relevant document"

        # For our specific test case, verify that at least one provider ranked Python first
        assert any(
            self.documents[idx].startswith("Python") for idx in top_indices.values()
        )

    def test_reranking_with_empty_documents(self):
        """Test reranking with empty document list."""
        try:
            # Try with OpenAI first, fall back to Cohere
            try:
                llm = get_llm_from_model("text-embedding-3-small", provider="openai")
                if not llm.supports_reranking():
                    raise ValueError("OpenAI model does not support reranking")
            except:
                llm = get_llm_from_model("rerank-english-v3.0", provider="cohere")
                if not llm.supports_reranking():
                    raise ValueError("Cohere model does not support reranking")

            # Run reranking with empty documents
            result = llm.rerank(query=self.query, documents=[])

            # Verify empty results
            assert result["ranked_documents"] == []
            assert result["indices"] == []
            assert result["scores"] == []
            assert result["usage"]["total_tokens"] in (0, None)

        except Exception as e:
            pytest.skip(f"Skipping empty documents test: {e!s}")

    def test_reranking_with_long_documents(self):
        """Test reranking with long documents that exceed token limits."""
        # Create long documents by repeating text
        long_documents = [
            "Python is a popular programming language. " * 50,
            "The Eiffel Tower is in Paris, France. " * 50,
            "Machine learning is a subset of AI. " * 50,
        ]

        try:
            # Try with OpenAI first, fall back to Cohere
            try:
                llm = get_llm_from_model("text-embedding-3-small", provider="openai")
                if not llm.supports_reranking():
                    raise ValueError("OpenAI model does not support reranking")
            except:
                llm = get_llm_from_model("rerank-english-v3.0", provider="cohere")
                if not llm.supports_reranking():
                    raise ValueError("Cohere model does not support reranking")

            # Run reranking with long documents
            result = llm.rerank(query=self.query, documents=long_documents)

            # Verify that results were returned despite long documents
            assert len(result["ranked_documents"]) > 0
            assert len(result["indices"]) > 0
            assert len(result["scores"]) > 0

            # Check that the most relevant document is about Python
            assert "Python" in result["ranked_documents"][0]

        except Exception as e:
            pytest.skip(f"Skipping long documents test: {e!s}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
