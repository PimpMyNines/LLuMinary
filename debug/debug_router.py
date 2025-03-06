"""
Debug router tests
"""

import traceback
from unittest.mock import patch

from lluminary.models.base import LLM
from lluminary.models.router import (
    MODEL_MAPPINGS,
    PROVIDER_REGISTRY,
    get_llm_from_model,
    register_model,
    register_provider,
)


class MockLLM(LLM):
    """Mock LLM implementation for testing."""

    SUPPORTED_MODELS = ["test-model", "test-model-1", "test-model-2"]
    THINKING_MODELS = []
    EMBEDDING_MODELS = []
    RERANKING_MODELS = []

    CONTEXT_WINDOW = {"test-model": 4096, "test-model-1": 4096, "test-model-2": 4096}

    COST_PER_MODEL = {
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


def debug_register_model():
    """Debug the register_model test"""
    print("\nDebugging register_model...")

    # Save original states
    original_mappings = MODEL_MAPPINGS.copy()
    original_registry = PROVIDER_REGISTRY.copy()

    try:
        # Register a mock provider
        print("Registering mock provider...")
        register_provider("mock_provider", MockLLM)

        print(f"Provider registry after registration: {PROVIDER_REGISTRY}")

        # Register a custom model
        print("Registering custom model...")
        register_model("custom-model", "mock_provider", "test-model")

        print(f"Model mappings after registration: {MODEL_MAPPINGS}")

        # Check if model was registered
        print(
            f"Is custom-model in MODEL_MAPPINGS? {('custom-model' in MODEL_MAPPINGS)}"
        )
        if "custom-model" in MODEL_MAPPINGS:
            print(f"Value: {MODEL_MAPPINGS['custom-model']}")

        # Check mapping
        expected = ("mock_provider", "test-model")
        actual = MODEL_MAPPINGS.get("custom-model")
        print(f"Expected: {expected}")
        print(f"Actual: {actual}")
        print(f"Match: {expected == actual}")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

    finally:
        # Restore original states
        MODEL_MAPPINGS.clear()
        MODEL_MAPPINGS.update(original_mappings)
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(original_registry)


def debug_get_llm_from_registered_model():
    """Debug the get_llm_from_registered_model test"""
    print("\nDebugging get_llm_from_registered_model...")

    # Save original states
    original_mappings = MODEL_MAPPINGS.copy()
    original_registry = PROVIDER_REGISTRY.copy()

    try:
        # Register a mock provider
        print("Registering mock provider...")
        register_provider("mock_provider", MockLLM)

        # Register a custom model
        print("Registering custom model...")
        register_model("custom-model", "mock_provider", "test-model")

        # Get LLM from the registered model
        print("Getting LLM from registered model...")
        with patch("src.lluminary.models.router._load_default_providers"):
            llm = get_llm_from_model("custom-model")

            # Check if correct LLM was returned
            print(f"Type of returned LLM: {type(llm)}")
            print(f"Is instance of MockLLM: {isinstance(llm, MockLLM)}")
            print(f"Model name: {llm.model_name}")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

    finally:
        # Restore original states
        MODEL_MAPPINGS.clear()
        MODEL_MAPPINGS.update(original_mappings)
        PROVIDER_REGISTRY.clear()
        PROVIDER_REGISTRY.update(original_registry)


if __name__ == "__main__":
    debug_register_model()
    debug_get_llm_from_registered_model()
