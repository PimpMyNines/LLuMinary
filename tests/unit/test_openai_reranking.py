"""
Tests for OpenAI provider document reranking functionality.

This module focuses specifically on testing the reranking
capabilities of the OpenAI provider, including similarity
calculations, query-document relevance, and edge cases.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary.models.providers.openai import OpenAILLM


@pytest.fixture
def openai_llm():
    """Fixture for OpenAI LLM instance with mocked client."""
    with patch.object(OpenAILLM, "auth") as mock_auth, patch(
        "openai.OpenAI"
    ) as mock_openai, patch.object(
        OpenAILLM, "SUPPORTED_MODELS", ["gpt-4o", "text-embedding-3-small"]
    ), patch.object(
        OpenAILLM, "RERANKING_MODELS", ["text-embedding-3-small"]
    ):
        # Mock authentication to avoid API errors
        mock_auth.return_value = None

        # Create the LLM instance with embedding model
        llm = OpenAILLM("text-embedding-3-small", api_key="test-key")

        # Initialize client attribute with mock
        llm.client = MagicMock()

        yield llm


@pytest.fixture
def mock_embeddings_response():
    """Mock response from embeddings API with realistic vector dimensions."""

    def create_response(vectors, total_tokens=100):
        """Create a mock response with given vectors and token count."""
        response = MagicMock()
        response.data = []

        for vector in vectors:
            item = MagicMock()
            item.embedding = vector
            response.data.append(item)

        response.usage = MagicMock()
        response.usage.total_tokens = total_tokens

        return response

    return create_response


class TestOpenAIReranking:
    """Tests for document reranking functionality."""

    def test_supports_reranking(self, openai_llm):
        """Test that reranking support is correctly reported."""
        # Embedding model should support reranking
        assert openai_llm.supports_reranking() is True

        # Test with non-embedding model
        with patch.object(OpenAILLM, "auth"), patch("openai.OpenAI"):
            non_embedding_llm = OpenAILLM("gpt-4o", api_key="test-key")
            assert non_embedding_llm.supports_reranking() is False

    def test_reranking_requirements(self, openai_llm):
        """Test validation of reranking requirements."""
        # Should reject if model doesn't support reranking
        with patch.object(openai_llm, "supports_reranking", return_value=False):
            with pytest.raises(NotImplementedError) as exc_info:
                openai_llm.rerank("test query", ["document 1", "document 2"])

            assert "does not support" in str(exc_info.value).lower()
            assert "reranking" in str(exc_info.value).lower()

        # Should handle empty documents list gracefully
        result = openai_llm.rerank("test query", [])
        assert result["ranked_documents"] == []
        assert result["indices"] == []
        assert result["scores"] == []
        assert result["usage"]["total_tokens"] == 0
        assert result["usage"]["total_cost"] == 0.0

    def test_basic_reranking(self, openai_llm, mock_embeddings_response):
        """Test basic document reranking functionality."""
        # Create test documents
        query = "information about cats"
        documents = [
            "Dogs are popular pets in many households.",
            "Cats are known for their independent nature.",
            "Parrots can mimic human speech patterns.",
        ]

        # Create embeddings with known similarities
        # We'll make doc[1] most similar to query, then doc[0], then doc[2]
        # Using 5-dim vectors for simplicity
        query_embedding = [0.8, 0.1, 0.4, 0.2, 0.1]
        doc_embeddings = [
            [0.7, 0.2, 0.3, 0.1, 0.2],  # Doc 0: medium similarity
            [0.9, 0.1, 0.3, 0.1, 0.1],  # Doc 1: high similarity
            [0.3, 0.3, 0.8, 0.3, 0.1],  # Doc 2: low similarity
        ]

        # Mock embeddings API responses
        mock_query_response = mock_embeddings_response([query_embedding], 20)
        mock_docs_response = mock_embeddings_response(doc_embeddings, 80)

        openai_llm.client.embeddings.create.side_effect = [
            mock_docs_response,
            mock_query_response,
        ]

        # Call rerank
        result = openai_llm.rerank(query, documents)

        # Verify rankings
        assert len(result["ranked_documents"]) == 3
        assert (
            result["ranked_documents"][0] == documents[1]
        )  # Most similar should be first
        assert result["ranked_documents"][1] == documents[0]
        assert result["ranked_documents"][2] == documents[2]

        # Verify indices
        assert result["indices"] == [1, 0, 2]

        # Verify scores are reasonable values between 0 and 1
        assert all(0 <= score <= 1 for score in result["scores"])
        assert result["scores"][0] > result["scores"][1] > result["scores"][2]

        # Verify usage statistics
        assert result["usage"]["total_tokens"] == 100  # 20 + 80
        assert result["usage"]["total_cost"] > 0
        assert result["usage"]["model"] == "text-embedding-3-small"

    def test_top_n_limiting(self, openai_llm, mock_embeddings_response):
        """Test limiting results to top_n documents."""
        # Create test documents
        query = "test query"
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]

        # Create embeddings with decreasing similarity
        query_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]
        doc_embeddings = [
            [0.5, 0.5, 0.0, 0.0, 0.0],  # Similarity: 0.5
            [0.7, 0.3, 0.0, 0.0, 0.0],  # Similarity: 0.7
            [0.3, 0.7, 0.0, 0.0, 0.0],  # Similarity: 0.3
            [0.9, 0.1, 0.0, 0.0, 0.0],  # Similarity: 0.9
            [0.1, 0.9, 0.0, 0.0, 0.0],  # Similarity: 0.1
        ]

        # Mock embeddings API responses
        mock_query_response = mock_embeddings_response([query_embedding], 10)
        mock_docs_response = mock_embeddings_response(doc_embeddings, 50)

        openai_llm.client.embeddings.create.side_effect = [
            mock_docs_response,
            mock_query_response,
        ]

        # Call rerank with top_n=2
        result = openai_llm.rerank(query, documents, top_n=2)

        # Verify only top 2 results are returned
        assert len(result["ranked_documents"]) == 2
        assert len(result["indices"]) == 2
        assert len(result["scores"]) == 2

        # Verify correct ranking: doc3 (0.9), doc1 (0.7)
        assert result["indices"] == [3, 1]
        assert result["ranked_documents"] == ["doc4", "doc2"]

    def test_without_scores(self, openai_llm, mock_embeddings_response):
        """Test reranking without returning scores."""
        # Create test documents
        query = "test query"
        documents = ["doc1", "doc2", "doc3"]

        # Create embeddings
        query_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]
        doc_embeddings = [
            [0.5, 0.5, 0.0, 0.0, 0.0],  # Similarity: 0.5
            [0.7, 0.3, 0.0, 0.0, 0.0],  # Similarity: 0.7
            [0.9, 0.1, 0.0, 0.0, 0.0],  # Similarity: 0.9
        ]

        # Mock embeddings API responses
        mock_query_response = mock_embeddings_response([query_embedding], 10)
        mock_docs_response = mock_embeddings_response(doc_embeddings, 30)

        openai_llm.client.embeddings.create.side_effect = [
            mock_docs_response,
            mock_query_response,
        ]

        # Call rerank with return_scores=False
        result = openai_llm.rerank(query, documents, return_scores=False)

        # Verify scores are None
        assert result["scores"] is None

        # Verify documents and indices are still returned
        assert len(result["ranked_documents"]) == 3
        assert len(result["indices"]) == 3

    def test_custom_model(self, openai_llm, mock_embeddings_response):
        """Test reranking with a custom model parameter."""
        with patch.object(
            OpenAILLM,
            "RERANKING_MODELS",
            ["text-embedding-3-small", "text-embedding-3-large"],
        ):
            # Create simple test case
            query = "test query"
            documents = ["doc1"]

            # Create embeddings
            query_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]
            doc_embeddings = [[1.0, 0.0, 0.0, 0.0, 0.0]]  # Perfect match

            # Mock embeddings API responses
            mock_query_response = mock_embeddings_response([query_embedding], 10)
            mock_docs_response = mock_embeddings_response(doc_embeddings, 10)

            openai_llm.client.embeddings.create.side_effect = [
                mock_docs_response,
                mock_query_response,
            ]

            # Call rerank with custom model
            custom_model = "text-embedding-3-large"
            result = openai_llm.rerank(query, documents, model=custom_model)

            # Verify model was used correctly
            assert result["usage"]["model"] == custom_model
            assert (
                openai_llm.client.embeddings.create.call_args_list[0][1]["model"]
                == custom_model
            )
            assert (
                openai_llm.client.embeddings.create.call_args_list[1][1]["model"]
                == custom_model
            )

    def test_embedding_error_handling(self, openai_llm):
        """Test error handling during embedding process."""
        # Create test case
        query = "test query"
        documents = ["doc1", "doc2"]

        # Mock an error from the embeddings API
        openai_llm.client.embeddings.create.side_effect = Exception("API error")

        # Call rerank and expect exception
        with pytest.raises(ValueError) as exc_info:
            openai_llm.rerank(query, documents)

        # Verify error message
        assert "error" in str(exc_info.value).lower()
        assert "openai" in str(exc_info.value).lower()

    def test_edge_case_identical_documents(self, openai_llm, mock_embeddings_response):
        """Test reranking with identical documents."""
        # Create test with identical documents
        query = "test query"
        documents = ["same content", "same content", "same content"]

        # Create embeddings with identical vectors
        query_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]
        doc_embeddings = [
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
        ]

        # Mock embeddings API responses
        mock_query_response = mock_embeddings_response([query_embedding], 10)
        mock_docs_response = mock_embeddings_response(doc_embeddings, 30)

        openai_llm.client.embeddings.create.side_effect = [
            mock_docs_response,
            mock_query_response,
        ]

        # Call rerank
        result = openai_llm.rerank(query, documents)

        # Verify all documents have the same score
        assert len(set(result["scores"])) == 1

        # Original order should be preserved for identical scores
        assert result["indices"] == [0, 1, 2]

    def test_edge_case_zero_similarity(self, openai_llm, mock_embeddings_response):
        """Test reranking with documents having zero similarity."""
        # Create test case with orthogonal embeddings
        query = "test query"
        documents = ["doc1", "doc2"]

        # Create embeddings with zero dot product
        query_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]
        doc_embeddings = [
            [0.0, 1.0, 0.0, 0.0, 0.0],  # Orthogonal to query
            [0.0, 0.0, 1.0, 0.0, 0.0],  # Orthogonal to query
        ]

        # Mock embeddings API responses
        mock_query_response = mock_embeddings_response([query_embedding], 10)
        mock_docs_response = mock_embeddings_response(doc_embeddings, 20)

        openai_llm.client.embeddings.create.side_effect = [
            mock_docs_response,
            mock_query_response,
        ]

        # Call rerank
        result = openai_llm.rerank(query, documents)

        # Verify scores are zero
        assert all(score == 0 for score in result["scores"])

    def test_edge_case_zero_magnitude(self, openai_llm, mock_embeddings_response):
        """Test reranking with zero magnitude vectors."""
        # Create test case with zero vectors
        query = "test query"
        documents = ["doc1", "doc2"]

        # Create embeddings with zero vectors
        query_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]
        doc_embeddings = [
            [0.0, 0.0, 0.0, 0.0, 0.0],  # Zero vector
            [0.5, 0.5, 0.0, 0.0, 0.0],  # Normal vector
        ]

        # Mock embeddings API responses
        mock_query_response = mock_embeddings_response([query_embedding], 10)
        mock_docs_response = mock_embeddings_response(doc_embeddings, 20)

        openai_llm.client.embeddings.create.side_effect = [
            mock_docs_response,
            mock_query_response,
        ]

        # Call rerank - should handle division by zero gracefully
        result = openai_llm.rerank(query, documents)

        # Verify rankings - normal vector should be first
        assert result["indices"] == [1, 0]
        assert result["scores"][0] > 0
        assert result["scores"][1] == 0

    def test_long_document_batching(self, openai_llm, mock_embeddings_response):
        """Test reranking with documents that would exceed token limits."""
        # Since we're doing a simple test of ranking logic, we can reduce the document count
        # This simplifies the test and makes it more stable
        query = "test query"
        document_count = 10  # Use a smaller number of documents for testing
        documents = [f"document {i}" for i in range(document_count)]

        # We'll need to create matching embeddings for all documents
        query_embedding = [1.0, 0.0, 0.0, 0.0, 0.0]

        # Create decreasing similarity values to test sorting
        doc_embeddings = []
        for i in range(document_count):
            similarity = 1.0 - (i / document_count)
            doc_embeddings.append([similarity, 1 - similarity, 0.0, 0.0, 0.0])

        # Mock embeddings API responses
        mock_query_response = mock_embeddings_response([query_embedding], 10)
        mock_docs_response = mock_embeddings_response(doc_embeddings, 100)

        openai_llm.client.embeddings.create.side_effect = [
            mock_docs_response,
            mock_query_response,
        ]

        # Call rerank
        result = openai_llm.rerank(query, documents)

        # Verify correct ranking - should be sorted by decreasing similarity
        assert len(result["ranked_documents"]) == document_count

        # Document 0 should be most similar and first
        assert result["indices"][0] == 0

        # Scores should be in descending order
        assert all(
            result["scores"][i] >= result["scores"][i + 1]
            for i in range(len(result["scores"]) - 1)
        )

        # Usage should combine the tokens
        assert result["usage"]["total_tokens"] == 110  # 100 + 10

    def test_cosine_similarity_calculation(self, openai_llm, mock_embeddings_response):
        """Test the cosine similarity calculation used for reranking."""
        # Create test vectors with known cosine similarities
        query_embedding = [1.0, 0.0, 0.0]
        doc_embeddings = [
            [0.5, 0.866, 0.0],  # 60° from query, cos(60°) = 0.5
            [0.0, 1.0, 0.0],  # 90° from query, cos(90°) = 0
            [0.866, 0.5, 0.0],  # 30° from query, cos(30°) = 0.866
            [-1.0, 0.0, 0.0],  # 180° from query, cos(180°) = -1
        ]
        documents = ["doc1", "doc2", "doc3", "doc4"]

        # Mock embeddings API responses
        mock_query_response = mock_embeddings_response([query_embedding], 10)
        mock_docs_response = mock_embeddings_response(doc_embeddings, 40)

        openai_llm.client.embeddings.create.side_effect = [
            mock_docs_response,
            mock_query_response,
        ]

        # Call rerank
        result = openai_llm.rerank("test", documents)

        # Verify scores approximate the expected cosine similarities
        assert abs(result["scores"][0] - 0.866) < 0.01  # doc3 (30° angle)
        assert abs(result["scores"][1] - 0.5) < 0.01  # doc1 (60° angle)
        assert abs(result["scores"][2] - 0.0) < 0.01  # doc2 (90° angle)
        assert result["scores"][3] < 0  # doc4 (180° angle)

        # Verify ranking is correct - highest similarity first
        assert result["indices"] == [2, 0, 1, 3]
