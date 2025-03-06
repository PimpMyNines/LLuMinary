"""
Tests for OpenAI provider reranking functionality.

This module focuses specifically on testing the document reranking
capabilities of the OpenAI provider.
"""
import math
from unittest.mock import patch, MagicMock

import pytest

from src.llmhandler.models.providers.openai import OpenAILLM


@pytest.fixture
def openai_llm():
    """Fixture for OpenAI LLM instance."""
    with patch.object(OpenAILLM, "auth") as mock_auth, patch("openai.OpenAI") as mock_openai:
        # Mock authentication to avoid API errors
        mock_auth.return_value = None

        # Create the LLM instance with mock API key
        llm = OpenAILLM("gpt-4o", api_key="test-key")

        # Initialize client attribute directly for tests
        llm.client = MagicMock()
        
        # Ensure config exists
        if not hasattr(llm, 'config'):
            llm.config = {}
        
        # Add client to config as expected by implementation
        llm.config["client"] = llm.client

        yield llm


def test_reranking_basic_functionality(openai_llm):
    """Test basic document reranking functionality."""
    # Test data
    query = "What is machine learning?"
    documents = [
        "Machine learning is a branch of artificial intelligence.",
        "Deep learning is a subset of machine learning.",
        "Python is a programming language often used for data science.",
        "Natural language processing deals with text data."
    ]
    
    # Mock embedding response
    mock_embedding_data = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4]},  # Query embedding
        ],
        "usage": {"total_tokens": 5, "prompt_tokens": 5}
    }
    
    # Mock embeddings for documents with different similarity scores
    mock_doc_embeddings = {
        "data": [
            {"embedding": [0.11, 0.21, 0.31, 0.41]},  # High similarity to query
            {"embedding": [0.12, 0.22, 0.32, 0.42]},  # Medium similarity
            {"embedding": [0.5, 0.6, 0.7, 0.8]},      # Low similarity
            {"embedding": [0.3, 0.3, 0.3, 0.3]},      # Medium-low similarity
        ],
        "usage": {"total_tokens": 20, "prompt_tokens": 20}
    }
    
    with patch.object(openai_llm, "client") as mock_client:
        # Set up mock embedding responses
        mock_client.embeddings.create.side_effect = [
            MagicMock(**mock_embedding_data),
            MagicMock(**mock_doc_embeddings)
        ]
        
        # Call rerank method
        results, usage = openai_llm.rerank(query=query, documents=documents)
        
        # Verify embeddings API was called twice (once for query, once for documents)
        assert mock_client.embeddings.create.call_count == 2
        
        # Verify results structure
        assert isinstance(results, list)
        assert len(results) == len(documents)
        assert all("document" in item and "score" in item for item in results)
        
        # Verify scores are between 0 and 1
        assert all(0 <= item["score"] <= 1 for item in results)
        
        # Verify usage information
        assert "total_tokens" in usage
        assert "total_cost" in usage
        

def test_reranking_top_n_parameter(openai_llm):
    """Test limiting reranking results with top_n parameter."""
    # Test data
    query = "Python programming"
    documents = [
        "Python is a programming language.",
        "Java is another programming language.",
        "Python has simple syntax.",
        "JavaScript is used for web development.",
        "Python is popular for data science."
    ]
    
    # Mock embedding responses
    mock_embedding_data = {
        "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}],
        "usage": {"total_tokens": 4, "prompt_tokens": 4}
    }
    
    mock_doc_embeddings = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3, 0.4]},
            {"embedding": [0.3, 0.3, 0.3, 0.3]},
            {"embedding": [0.1, 0.2, 0.3, 0.4]},
            {"embedding": [0.5, 0.5, 0.5, 0.5]},
            {"embedding": [0.1, 0.2, 0.3, 0.4]},
        ],
        "usage": {"total_tokens": 20, "prompt_tokens": 20}
    }
    
    with patch.object(openai_llm, "client") as mock_client:
        # Set up mock embedding responses
        mock_client.embeddings.create.side_effect = [
            MagicMock(**mock_embedding_data),
            MagicMock(**mock_doc_embeddings)
        ]
        
        # Test with top_n=2
        results, usage = openai_llm.rerank(query=query, documents=documents, top_n=2)
        
        # Verify only top 2 results are returned
        assert len(results) == 2
        
        # Test with top_n=3
        mock_client.embeddings.create.side_effect = [
            MagicMock(**mock_embedding_data),
            MagicMock(**mock_doc_embeddings)
        ]
        results, usage = openai_llm.rerank(query=query, documents=documents, top_n=3)
        
        # Verify only top 3 results are returned
        assert len(results) == 3


def test_reranking_error_handling(openai_llm):
    """Test error handling during reranking process."""
    query = "Test query"
    documents = ["Document 1", "Document 2"]
    
    with patch.object(openai_llm, "client") as mock_client:
        # Test API error
        mock_client.embeddings.create.side_effect = Exception("API error")
        
        with pytest.raises(Exception) as excinfo:
            openai_llm.rerank(query=query, documents=documents)
        
        assert "API error" in str(excinfo.value)
        
        # Test empty documents list
        mock_client.embeddings.create.side_effect = None
        results, usage = openai_llm.rerank(query=query, documents=[])
        
        assert len(results) == 0
        
        # Test invalid document type
        with pytest.raises(ValueError):
            openai_llm.rerank(query=query, documents=[1, 2, 3])


def test_reranking_cost_calculation(openai_llm):
    """Test cost calculation for reranking operations."""
    query = "Short query"
    documents = ["Short document 1", "Short document 2"]
    
    # Mock embedding responses with token usage
    mock_embedding_data = {
        "data": [{"embedding": [0.1, 0.2, 0.3]}],
        "usage": {"total_tokens": 3, "prompt_tokens": 3}
    }
    
    mock_doc_embeddings = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ],
        "usage": {"total_tokens": 6, "prompt_tokens": 6}
    }
    
    with patch.object(openai_llm, "client") as mock_client:
        # Set up mock embedding responses
        mock_client.embeddings.create.side_effect = [
            MagicMock(**mock_embedding_data),
            MagicMock(**mock_doc_embeddings)
        ]
        
        # Perform reranking
        results, usage = openai_llm.rerank(query=query, documents=documents)
        
        # Verify token usage is tracked
        assert usage["total_tokens"] == 9  # 3 + 6
        
        # Verify cost is calculated
        assert "total_cost" in usage
        assert usage["total_cost"] > 0
