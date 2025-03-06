"""
Basic tests for OpenAI provider class.

This module verifies basic properties and behaviors of the OpenAI provider.
"""

from unittest.mock import patch

from lluminary.models.providers.openai import OpenAILLM


def test_provider_properties():
    """Verify basic provider properties."""
    # Create an instance
    llm = OpenAILLM("gpt-4o", api_key="test-key")

    # Check properties
    assert llm.model_name == "gpt-4o"
    assert llm.provider_name == "openai"

    # Check supported models list
    assert len(llm.SUPPORTED_MODELS) > 0
    assert "gpt-4o" in llm.SUPPORTED_MODELS

    # Check embedding models
    assert len(llm.EMBEDDING_MODELS) > 0
    assert "text-embedding-3-small" in llm.EMBEDDING_MODELS

    # Check cost structure
    assert len(llm.embedding_costs) > 0
    assert llm.embedding_costs["text-embedding-3-small"] > 0


def test_provider_capabilities():
    """Test provider capability flags."""
    llm = OpenAILLM("gpt-4o", api_key="test-key")

    # Check capabilities
    assert llm.supports_embeddings() is True
    assert llm.supports_image_input() is True

    # Check thinking models
    assert llm.is_thinking_model("gpt-4o") is True
    assert llm.is_thinking_model("unknown-model") is False


@patch("src.lluminary.models.providers.openai.OpenAI")
def test_basic_configuration(mock_openai):
    """Test basic configuration storage."""
    # Create instance with various config options
    api_key = "test-api-key"
    api_base = "https://test-base.example.com"
    organization_id = "test-org"
    timeout = 90

    llm = OpenAILLM(
        "gpt-4o",
        api_key=api_key,
        api_base=api_base,
        organization_id=organization_id,
        timeout=timeout,
    )

    # Verify config storage
    assert llm.config["api_key"] == api_key
    assert llm.api_base == api_base
    assert llm.config["organization_id"] == organization_id
    assert llm.timeout == timeout
