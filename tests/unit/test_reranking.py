"""
Tests for the document reranking functionality.
"""

import unittest
from unittest.mock import MagicMock, patch

from lluminary import get_llm_from_model
from lluminary.models import set_provider_config


class TestReranking(unittest.TestCase):
    """Tests for document reranking functionality."""

    def setUp(self):
        """Set up test environment."""
        # Mock API keys for testing
        self.openai_key = "test-openai-key"
        self.cohere_key = "test-cohere-key"

        # Configure providers with mock keys
        set_provider_config("openai", {"api_key": self.openai_key})
        set_provider_config("cohere", {"api_key": self.cohere_key})

        # Test documents
        self.documents = [
            "Python is a popular programming language.",
            "The Eiffel Tower is in Paris, France.",
            "Machine learning is a subset of AI.",
            "Cats are common pets worldwide.",
        ]

        # Test query
        self.query = "Which programming language is popular?"

    def test_openai_supports_reranking(self):
        """Test that an OpenAI reranking model correctly reports support."""
        llm = get_llm_from_model("text-embedding-3-small", provider="openai")
        self.assertTrue(llm.supports_reranking())

        # Non-reranking model should report no support
        llm = get_llm_from_model("gpt-3.5-turbo", provider="openai")
        self.assertFalse(llm.supports_reranking())

    def test_cohere_supports_reranking(self):
        """Test that a Cohere reranking model correctly reports support."""
        llm = get_llm_from_model("rerank-english-v3.0", provider="cohere")
        self.assertTrue(llm.supports_reranking())

        # Non-reranking model should report no support
        llm = get_llm_from_model("command", provider="cohere")
        self.assertFalse(llm.supports_reranking())

    @patch("openai.OpenAI")
    def test_openai_reranking(self, mock_openai):
        """Test OpenAI reranking functionality with mocked client."""
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock embedding responses
        mock_doc_response = MagicMock()
        mock_doc_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.2, 0.3, 0.4]),
            MagicMock(embedding=[0.3, 0.4, 0.5]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_doc_response.usage = MagicMock(total_tokens=100)

        mock_query_response = MagicMock()
        mock_query_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_query_response.usage = MagicMock(total_tokens=10)

        # Set up the mock to return our mock responses
        mock_client.embeddings.create.side_effect = [
            mock_doc_response,
            mock_query_response,
        ]

        # Create OpenAI LLM and run reranking
        llm = get_llm_from_model("text-embedding-3-small", provider="openai")

        # Monkeypatch the client
        llm.client = mock_client

        # Run reranking
        result = llm.rerank(query=self.query, documents=self.documents, top_n=2)

        # Verify the result structure
        self.assertIn("ranked_documents", result)
        self.assertIn("indices", result)
        self.assertIn("scores", result)
        self.assertIn("usage", result)

        # Verify top_n parameter was respected
        self.assertEqual(len(result["ranked_documents"]), 2)
        self.assertEqual(len(result["indices"]), 2)
        self.assertEqual(len(result["scores"]), 2)

    @patch("cohere.Client")
    def test_cohere_reranking(self, mock_cohere_client):
        """Test Cohere reranking functionality with mocked client."""
        # Setup mock client
        mock_client = MagicMock()
        mock_cohere_client.return_value = mock_client

        # Mock rerank response
        mock_result1 = MagicMock(index=0, relevance_score=0.8)
        mock_result2 = MagicMock(index=2, relevance_score=0.6)
        mock_result3 = MagicMock(index=1, relevance_score=0.4)
        mock_result4 = MagicMock(index=3, relevance_score=0.2)

        mock_response = MagicMock()
        mock_response.results = [mock_result1, mock_result2, mock_result3, mock_result4]

        # Set up the mock to return our mock response
        mock_client.rerank.return_value = mock_response

        # Create Cohere LLM and run reranking
        llm = get_llm_from_model("rerank-english-v3.0", provider="cohere")

        # Run reranking
        result = llm.rerank(query=self.query, documents=self.documents, top_n=2)

        # Verify the result structure
        self.assertIn("ranked_documents", result)
        self.assertIn("indices", result)
        self.assertIn("scores", result)
        self.assertIn("usage", result)

        # Verify top_n parameter was respected
        self.assertEqual(len(result["ranked_documents"]), 2)
        self.assertEqual(len(result["indices"]), 2)
        self.assertEqual(len(result["scores"]), 2)

        # Verify the order matches the relevance scores
        self.assertEqual(result["indices"][0], 0)
        self.assertEqual(result["indices"][1], 2)
        self.assertEqual(result["scores"][0], 0.8)
        self.assertEqual(result["scores"][1], 0.6)

    def test_empty_documents(self):
        """Test reranking with empty document list."""
        llm = get_llm_from_model("text-embedding-3-small", provider="openai")

        result = llm.rerank(query=self.query, documents=[])

        # Verify empty results
        self.assertEqual(result["ranked_documents"], [])
        self.assertEqual(result["indices"], [])
        self.assertEqual(result["scores"], [])
        self.assertEqual(result["usage"]["total_tokens"], 0)
        self.assertEqual(result["usage"]["total_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
