"""
Unit tests for the router module.
"""

from typing import ClassVar, Dict, List
from unittest.mock import patch

import pytest

from lluminary.models.base import LLM
from lluminary.models.router import (
    MODEL_MAPPINGS,
    PROVIDER_REGISTRY,
    get_llm_from_model,
    list_available_models,
    register_model,
    register_provider,
)


class MockLLM(LLM):
    """Mock LLM implementation for testing."""

    SUPPORTED_MODELS: ClassVar[List[str]] = [
        "test-model",
        "test-model-1",
        "test-model-2",
    ]
    THINKING_MODELS: ClassVar[List[str]] = []
    EMBEDDING_MODELS: ClassVar[List[str]] = []
    RERANKING_MODELS: ClassVar[List[str]] = []

    CONTEXT_WINDOW: ClassVar[Dict[str, int]] = {
        "test-model": 4096,
        "test-model-1": 4096,
        "test-model-2": 4096,
    }

    COST_PER_MODEL: ClassVar[Dict[str, Dict[str, float]]] = {
        "test-model": {"read_token": 0.001, "write_token": 0.002, "image_cost": 0.01},
        "test-model-1": {"read_token": 0.001, "write_token": 0.002, "image_cost": 0.01},
        "test-model-2": {"read_token": 0.001, "write_token": 0.002, "image_cost": 0.01},
    }

    def __init__(self, model_name):
        super().__init__(model_name)

    def _format_messages_for_model(self, messages):
        """Mock implementation of message formatting."""
        return messages

    def auth(self):
        """Mock implementation of authentication."""
        pass

    def _raw_generate(
        self,
        event_id,
        system_prompt,
        messages,
        max_tokens=1000,
        temp=0.0,
        top_k=200,
        tools=None,
        thinking_budget=None,
    ):
        """Mock implementation of generation."""
        return "test response", {
            "read_tokens": 10,
            "write_tokens": 5,
            "total_tokens": 15,
            "read_cost": 0.01,
            "write_cost": 0.01,
            "total_cost": 0.02,
        }


def test_register_provider():
    """Test registering a custom provider."""
    # Save original registry state
    original_registry = PROVIDER_REGISTRY.copy()

    try:
        # Register a mock provider
        register_provider("mock_provider", MockLLM)

        # Check if provider was registered
        assert "mock_provider" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["mock_provider"] == MockLLM

    finally:
        # Restore original registry state
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(original_registry)


def test_register_provider_invalid():
    """Test registering an invalid provider."""
    # Try to register a non-LLM class
    with pytest.raises(TypeError):
        register_provider("invalid", dict)


def test_register_model():
    """Test registering a custom model."""
    # Save original mappings state
    original_mappings = MODEL_MAPPINGS.copy()
    original_registry = PROVIDER_REGISTRY.copy()

    try:
        # Register a mock provider
        register_provider("mock_provider", MockLLM)

        # Register a custom model
        register_model("custom-model", "mock_provider", "test-model")

        # Check if model was registered
        assert "custom-model" in MODEL_MAPPINGS
        assert MODEL_MAPPINGS["custom-model"]["provider"] == "mock_provider"
        assert MODEL_MAPPINGS["custom-model"]["model"] == "test-model"

    finally:
        # Restore original states
        MODEL_MAPPINGS.clear()
        MODEL_MAPPINGS.update(original_mappings)
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(original_registry)


def test_register_model_invalid_provider():
    """Test registering a model with an invalid provider."""
    with pytest.raises(ValueError):
        register_model("custom-model", "nonexistent_provider", "test-model")


def test_get_llm_from_registered_model():
    """Test getting an LLM from a registered model."""
    # Save original states
    original_mappings = MODEL_MAPPINGS.copy()
    original_registry = PROVIDER_REGISTRY.copy()

    try:
        # Register a mock provider
        register_provider("mock_provider", MockLLM)

        # Register a custom model
        register_model("custom-model", "mock_provider", "test-model")

        # Get LLM from the registered model
        with patch("src.lluminary.models.router._load_default_providers"):
            llm = get_llm_from_model("custom-model")

            # Check if correct LLM was returned
            assert isinstance(llm, MockLLM)
            assert llm.model_name == "test-model"

    finally:
        # Restore original states
        MODEL_MAPPINGS.clear()
        MODEL_MAPPINGS.update(original_mappings)
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(original_registry)


def test_list_available_models():
    """Test listing available models."""
    # Save original states
    original_mappings = MODEL_MAPPINGS.copy()
    original_registry = PROVIDER_REGISTRY.copy()

    try:
        # Register a mock provider
        register_provider("mock_provider", MockLLM)

        # Register custom models
        register_model("custom-model-1", "mock_provider", "test-model-1")
        register_model("custom-model-2", "mock_provider", "test-model-2")

        # Get available models
        models = list_available_models()

        # Check if models are listed correctly
        assert "mock_provider" in models
        assert "custom-model-1" in models["mock_provider"]
        assert "custom-model-2" in models["mock_provider"]

    finally:
        # Restore original states
        MODEL_MAPPINGS.clear()
        MODEL_MAPPINGS.update(original_mappings)
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(original_registry)


def test_get_llm_from_model_invalid():
    """Test getting an LLM from an invalid model."""
    with pytest.raises(ValueError):
        get_llm_from_model("nonexistent-model")
