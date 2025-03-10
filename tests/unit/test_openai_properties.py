"""
Tests for OpenAI provider model properties and capabilities.

This module focuses on testing properties of the OpenAI provider that don't require mocking
complex dependencies, including model lists, costs, and supported features.
"""

from unittest.mock import patch

import pytest
from lluminary.models.providers.openai import OpenAILLM


# Simple mocking of auth method to prevent API calls
@pytest.fixture
def openai_llm():
    """Create a mocked OpenAI LLM instance."""
    with patch.object(OpenAILLM, "auth") as _:
        llm = OpenAILLM("gpt-4o", api_key="test_key")
        yield llm


def test_model_list(openai_llm):
    """Test that the provider has expected model lists."""
    # Check supported models
    assert len(openai_llm.SUPPORTED_MODELS) > 0
    assert "gpt-4o" in openai_llm.SUPPORTED_MODELS

    # Check embedding models
    assert len(openai_llm.EMBEDDING_MODELS) > 0
    assert "text-embedding-3-small" in openai_llm.EMBEDDING_MODELS
    assert "text-embedding-3-large" in openai_llm.EMBEDDING_MODELS

    # Check thinking models
    assert len(openai_llm.THINKING_MODELS) > 0
    assert "gpt-4o" in openai_llm.THINKING_MODELS

    # Check reranking models
    assert len(openai_llm.RERANKING_MODELS) > 0
    assert "text-embedding-3-small" in openai_llm.RERANKING_MODELS


def test_cost_structure(openai_llm):
    """Test that cost structures are correctly defined."""
    # Check embedding costs
    assert len(openai_llm.embedding_costs) > 0
    assert openai_llm.embedding_costs["text-embedding-3-small"] > 0
    assert openai_llm.embedding_costs["text-embedding-3-large"] > 0

    # Check reranking costs
    assert len(openai_llm.reranking_costs) > 0
    assert openai_llm.reranking_costs["text-embedding-3-small"] > 0

    # Check image generation costs
    assert len(openai_llm.IMAGE_GENERATION_COSTS) > 0
    assert "dall-e-3" in openai_llm.IMAGE_GENERATION_COSTS
    assert "1024x1024" in openai_llm.IMAGE_GENERATION_COSTS["dall-e-3"]
    assert openai_llm.IMAGE_GENERATION_COSTS["dall-e-3"]["1024x1024"] > 0


def test_default_values(openai_llm):
    """Test that default values are correctly set."""
    # Check default embedding model
    assert openai_llm.DEFAULT_EMBEDDING_MODEL == "text-embedding-3-small"

    # Check default reranking model
    assert openai_llm.DEFAULT_RERANKING_MODEL == "text-embedding-3-small"

    # Check default timeout
    assert openai_llm.timeout == 60


def test_capability_methods(openai_llm):
    """Test capability reporting methods."""
    # Test supports_embeddings
    assert openai_llm.supports_embeddings() is True

    # Test supports_image_input
    assert openai_llm.supports_image_input() is True

    # Test is_thinking_model
    assert openai_llm.is_thinking_model("gpt-4o") is True
    assert openai_llm.is_thinking_model("unknown-model") is False

    # Test get_supported_models
    models = openai_llm.get_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert all(isinstance(model, str) for model in models)


def test_context_window(openai_llm):
    """Test context window reporting."""
    # Test the provider has a CONTEXT_WINDOW attribute
    assert hasattr(openai_llm, "CONTEXT_WINDOW")
    assert isinstance(openai_llm.CONTEXT_WINDOW, dict)
    assert len(openai_llm.CONTEXT_WINDOW) > 0

    # Test that context windows are positive integers for any defined models
    for model, window in openai_llm.CONTEXT_WINDOW.items():
        assert isinstance(window, int)
        assert window > 0


def test_initialization_config():
    """Test various initialization config options."""
    # We need to patch auth to prevent actual API calls
    with patch.object(OpenAILLM, "auth") as _:
        # Test basic initialization
        llm = OpenAILLM("gpt-4o", api_key="test_key")
        assert llm.model_name == "gpt-4o"
        assert llm.config["api_key"] == "test_key"

        # Test with API base
        api_base = "https://test.openai.com"
        llm = OpenAILLM("gpt-4o", api_key="test_key", api_base=api_base)
        assert llm.api_base == api_base

        # Test with organization ID
        org_id = "org-test-123"
        llm = OpenAILLM("gpt-4o", api_key="test_key", organization_id=org_id)
        assert llm.config["organization_id"] == org_id

        # Test with custom timeout
        timeout = 120
        llm = OpenAILLM("gpt-4o", api_key="test_key", timeout=timeout)
        assert llm.timeout == timeout
