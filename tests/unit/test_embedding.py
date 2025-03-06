"""
Unit tests for embedding functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from lluminary import get_llm_from_model
from lluminary.models.providers.cohere import CohereLLM
from lluminary.models.providers.openai import OpenAILLM


@pytest.mark.unit
class TestEmbeddings:
    """Unit tests for embedding functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.test_text = "This is a test sentence for embeddings."
        self.test_texts = [
            "This is the first test sentence.",
            "Here is another test sentence for embeddings.",
            "And this is a third one to test batch processing.",
        ]

    @patch("openai.OpenAI")
    def test_openai_embed_single(self, mock_openai):
        """Test OpenAI embedding for a single text."""
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Create mock embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])]
        mock_response.usage = MagicMock(total_tokens=5)

        # Set up the mock to return our mock response
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(OpenAILLM, "auth", return_value=None):
            # Create OpenAI LLM
            llm = get_llm_from_model("text-embedding-3-small", provider="openai")

            # Monkeypatch the client
            llm.client = mock_client

            # Get embeddings - note that embed expects a list of strings
            embeddings, usage = llm.embed([self.test_text])

            # Verify the result structure
            assert isinstance(embeddings, list)
            assert isinstance(usage, dict)

            # Verify embeddings
            assert len(embeddings) == 1  # One text input
            assert len(embeddings[0]) == 5  # Embedding dimension
            assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]

            # Verify usage
            assert usage["total_tokens"] == 5
            assert "total_cost" in usage

            # Verify the API was called correctly
            mock_client.embeddings.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_openai_embed_batch(self, mock_openai):
        """Test OpenAI embedding for a batch of texts."""
        # Setup mock response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Create mock embedding response
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
            MagicMock(embedding=[0.7, 0.8, 0.9]),
        ]
        mock_response.usage = MagicMock(total_tokens=15)

        # Set up the mock to return our mock response
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(OpenAILLM, "auth", return_value=None):
            # Create OpenAI LLM
            llm = get_llm_from_model("text-embedding-3-small", provider="openai")

            # Monkeypatch the client
            llm.client = mock_client

            # Get embeddings
            embeddings, usage = llm.embed(self.test_texts)

            # Verify the result structure
            assert isinstance(embeddings, list)
            assert isinstance(usage, dict)

            # Verify embeddings
            assert len(embeddings) == 3  # Number of texts
            assert all(
                len(emb) == 3 for emb in embeddings
            )  # Each embedding has dimension 3
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]
            assert embeddings[2] == [0.7, 0.8, 0.9]

            # Verify usage
            assert usage["total_tokens"] == 15
            assert "total_cost" in usage

            # Verify the API was called correctly
            mock_client.embeddings.create.assert_called_once()

    @patch("cohere.Client")
    def test_cohere_embed_single(self, mock_cohere_client):
        """Test Cohere embedding for a single text."""
        # Setup mock client
        mock_client = MagicMock()
        mock_cohere_client.return_value = mock_client

        # Create mock embedding response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_response.meta = MagicMock(billed_units=5)

        # Set up the mock to return our mock response
        mock_client.embed.return_value = mock_response

        with patch.object(CohereLLM, "auth", return_value=None):
            # Create Cohere LLM
            llm = get_llm_from_model("embed-english-v3.0", provider="cohere")

            # Monkeypatch the client
            llm.client = mock_client

            # Get embeddings - note that embed expects a list of strings
            embeddings, usage = llm.embed([self.test_text])

            # Verify the result structure
            assert isinstance(embeddings, list)
            assert isinstance(usage, dict)

            # Verify embeddings
            assert len(embeddings) == 1  # One text input
            assert len(embeddings[0]) == 5  # Embedding dimension
            assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]

            # Verify usage
            assert usage["total_tokens"] == 5
            assert "total_cost" in usage

            # Verify the API was called correctly
            mock_client.embed.assert_called_once()

    @patch("cohere.Client")
    def test_cohere_embed_batch(self, mock_cohere_client):
        """Test Cohere embedding for a batch of texts."""
        # Setup mock client
        mock_client = MagicMock()
        mock_cohere_client.return_value = mock_client

        # Create mock embedding response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        mock_response.meta = MagicMock(billed_units=15)

        # Set up the mock to return our mock response
        mock_client.embed.return_value = mock_response

        with patch.object(CohereLLM, "auth", return_value=None):
            # Create Cohere LLM
            llm = get_llm_from_model("embed-english-v3.0", provider="cohere")

            # Monkeypatch the client
            llm.client = mock_client

            # Get embeddings
            embeddings, usage = llm.embed(self.test_texts)

            # Verify the result structure
            assert isinstance(embeddings, list)
            assert isinstance(usage, dict)

            # Verify embeddings
            assert len(embeddings) == 3  # Number of texts
            assert all(
                len(emb) == 3 for emb in embeddings
            )  # Each embedding has dimension 3
            assert embeddings[0] == [0.1, 0.2, 0.3]
            assert embeddings[1] == [0.4, 0.5, 0.6]
            assert embeddings[2] == [0.7, 0.8, 0.9]

            # Verify usage
            assert usage["total_tokens"] == 15
            assert "total_cost" in usage

            # Verify the API was called correctly
            mock_client.embed.assert_called_once()

    def test_supports_embeddings(self):
        """Test the base LLM.supports_embeddings method."""
        from lluminary.models.base import LLM

        # Create a mock LLM instance
        llm = MagicMock(spec=LLM)
        llm.EMBEDDING_MODELS = ["text-embedding-3-small", "embed-english-v3.0"]

        # Test when model is in embedding models list
        llm.model_name = "text-embedding-3-small"
        assert llm.supports_embeddings() is True

        # Test when model is not in embedding models list
        llm.model_name = "gpt-4"
        assert llm.supports_embeddings() is False

        # Test when embedding models list is empty
        llm.EMBEDDING_MODELS = []
        llm.model_name = "any-model"
        assert llm.supports_embeddings() is False

    @patch("openai.OpenAI")
    def test_empty_text_embedding(self, mock_openai):
        """Test embedding with empty text."""
        # Setup mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Create mock embedding response for empty text
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[])]
        mock_response.usage = MagicMock(total_tokens=0)

        # Set up the mock to return our mock response
        mock_client.embeddings.create.return_value = mock_response

        with patch.object(OpenAILLM, "auth", return_value=None):
            # Create OpenAI LLM
            llm = get_llm_from_model("text-embedding-3-small", provider="openai")

            # Monkeypatch the client
            llm.client = mock_client

            # Get embeddings for empty text
            embeddings, usage = llm.embed([""])

            # Verify the result structure
            assert isinstance(embeddings, list)
            assert isinstance(usage, dict)

            # Verify embeddings - result should be a list with one empty embedding
            assert len(embeddings) == 1
            assert embeddings[0] == []

            # Verify usage
            assert usage["total_tokens"] == 0
            assert usage["total_cost"] == 0.0

    @patch("openai.OpenAI")
    def test_embedding_error_handling(self, mock_openai):
        """Test error handling in embedding."""
        # Setup mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Set up the mock to raise an exception
        mock_client.embeddings.create.side_effect = Exception("API error")

        with patch.object(OpenAILLM, "auth", return_value=None):
            # Create OpenAI LLM
            llm = get_llm_from_model("text-embedding-3-small", provider="openai")

            # Monkeypatch the client
            llm.client = mock_client

            # Get embeddings should raise an exception
            with pytest.raises(Exception):
                llm.embed([self.test_text])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
