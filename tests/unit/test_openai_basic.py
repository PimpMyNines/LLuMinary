"""
Basic tests for OpenAI provider class.

This module verifies basic properties and behaviors of the OpenAI provider.
"""

from unittest.mock import patch, MagicMock

from lluminary.models.providers.openai import OpenAILLM


@patch("lluminary.models.providers.openai.OpenAI")
def test_provider_properties(mock_openai_class):
    """Verify basic provider properties."""
    print("Starting test_provider_properties...")
    
    # Setup mock OpenAI client
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Create a list models response
    mock_client.models.list.return_value = MagicMock()
    
    # Create an instance
    try:
        print("Creating OpenAILLM instance...")
        llm = OpenAILLM("gpt-4o", api_key="test-key")
        print(f"OpenAILLM instance created: {llm}")

        # Check properties
        print(f"Checking model_name: {llm.model_name}")
        assert llm.model_name == "gpt-4o"
        print(f"Checking provider_name: {llm.provider_name}")
        assert llm.provider_name == "openai"

        # Check supported models list
        print(f"Checking SUPPORTED_MODELS: {llm.SUPPORTED_MODELS}")
        assert len(llm.SUPPORTED_MODELS) > 0
        assert "gpt-4o" in llm.SUPPORTED_MODELS

        # Check embedding models
        print(f"Checking EMBEDDING_MODELS: {llm.EMBEDDING_MODELS}")
        assert len(llm.EMBEDDING_MODELS) > 0
        assert "text-embedding-3-small" in llm.EMBEDDING_MODELS

        # Check cost structure
        print(f"Checking embedding_costs: {llm.embedding_costs if hasattr(llm, 'embedding_costs') else 'Not found'}")
        assert len(llm.embedding_costs) > 0
        assert llm.embedding_costs["text-embedding-3-small"] > 0
    except Exception as e:
        print(f"Exception in test_provider_properties: {type(e).__name__}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise


@patch("lluminary.models.providers.openai.OpenAI")
def test_provider_capabilities(mock_openai_class):
    """Test provider capability flags."""
    # Setup mock OpenAI client
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Create a list models response
    mock_client.models.list.return_value = MagicMock()
    
    llm = OpenAILLM("gpt-4o", api_key="test-key")

    # Check capabilities
    assert llm.supports_embeddings() is True
    assert llm.supports_image_input() is True

    # Check thinking models
    assert "gpt-4o" in llm.THINKING_MODELS
    
    # Use the is_thinking_model method
    assert llm.is_thinking_model("gpt-4o") is True
    assert llm.is_thinking_model("unknown-model") is False


@patch("lluminary.models.providers.openai.OpenAI")
def test_basic_configuration(mock_openai_class):
    """Test basic configuration storage."""
    # Setup mock OpenAI client
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Create a list models response
    mock_client.models.list.return_value = MagicMock()
    
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
    assert llm.config.get("organization_id") == organization_id
    assert llm.timeout == timeout
