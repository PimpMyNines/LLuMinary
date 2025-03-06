"""
Provider capabilities module for LLuMinary.

This module provides a structured way to document and detect the capabilities
of different LLM providers and models.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Set


class Capability(Enum):
    """Enum representing different LLM capabilities."""

    TEXT_GENERATION = auto()
    TEXT_EMBEDDINGS = auto()
    IMAGE_INPUT = auto()
    IMAGE_GENERATION = auto()
    STREAMING = auto()
    FUNCTIONS = auto()
    RERANKING = auto()
    THINKING_BUDGET = auto()
    JSON_MODE = auto()
    TOOL_CALLING = auto()
    VISION = auto()


class CapabilityRegistry:
    """Registry for tracking provider and model capabilities."""

    # Provider-level capabilities
    _provider_capabilities: Dict[str, Set[Capability]] = {
        "openai": {
            Capability.TEXT_GENERATION,
            Capability.TEXT_EMBEDDINGS,
            Capability.IMAGE_INPUT,
            Capability.IMAGE_GENERATION,
            Capability.STREAMING,
            Capability.FUNCTIONS,
            Capability.RERANKING,
            Capability.JSON_MODE,
            Capability.TOOL_CALLING,
            Capability.VISION,
        },
        "anthropic": {
            Capability.TEXT_GENERATION,
            Capability.TEXT_EMBEDDINGS,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.THINKING_BUDGET,
            Capability.TOOL_CALLING,
            Capability.VISION,
        },
        "google": {
            Capability.TEXT_GENERATION,
            Capability.TEXT_EMBEDDINGS,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.FUNCTIONS,
            Capability.VISION,
        },
        "cohere": {
            Capability.TEXT_GENERATION,
            Capability.TEXT_EMBEDDINGS,
            Capability.RERANKING,
            Capability.STREAMING,
        },
        "bedrock": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.VISION,
        },
    }

    # Model-specific capabilities
    _model_capabilities: Dict[str, Set[Capability]] = {
        # OpenAI models
        "gpt-4o": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.FUNCTIONS,
            Capability.JSON_MODE,
            Capability.TOOL_CALLING,
            Capability.VISION,
        },
        "gpt-4o-mini": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.FUNCTIONS,
            Capability.JSON_MODE,
            Capability.TOOL_CALLING,
            Capability.VISION,
        },
        "o1": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.FUNCTIONS,
            Capability.JSON_MODE,
            Capability.TOOL_CALLING,
            Capability.VISION,
        },
        "o3-mini": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.FUNCTIONS,
            Capability.JSON_MODE,
            Capability.TOOL_CALLING,
            Capability.VISION,
        },
        # Anthropic models
        "claude-sonnet-3.5": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.THINKING_BUDGET,
            Capability.TOOL_CALLING,
            Capability.VISION,
        },
        "claude-haiku-3.5": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.THINKING_BUDGET,
            Capability.TOOL_CALLING,
            Capability.VISION,
        },
        # Google models
        "gemini-pro": {
            Capability.TEXT_GENERATION,
            Capability.STREAMING,
            Capability.FUNCTIONS,
        },
        "gemini-pro-vision": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.FUNCTIONS,
            Capability.VISION,
        },
        "gemini-ultra": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.FUNCTIONS,
            Capability.VISION,
        },
        # Bedrock models
        "bedrock-claude-haiku-3.5": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.VISION,
        },
        "bedrock-claude-sonnet-3.5-v1": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.VISION,
        },
        "bedrock-claude-sonnet-3.5-v2": {
            Capability.TEXT_GENERATION,
            Capability.IMAGE_INPUT,
            Capability.STREAMING,
            Capability.VISION,
        },
        # Cohere models
        "cohere-embed": {
            Capability.TEXT_EMBEDDINGS,
        },
        "cohere-rerank": {
            Capability.RERANKING,
        },
    }

    @classmethod
    def provider_has_capability(cls, provider: str, capability: Capability) -> bool:
        """
        Check if a provider supports a specific capability.

        Args:
            provider: Provider name
            capability: Capability to check

        Returns:
            bool: True if the provider supports the capability
        """
        provider = provider.lower()
        return (
            provider in cls._provider_capabilities
            and capability in cls._provider_capabilities[provider]
        )

    @classmethod
    def model_has_capability(cls, model: str, capability: Capability) -> bool:
        """
        Check if a specific model supports a capability.

        Args:
            model: Model name
            capability: Capability to check

        Returns:
            bool: True if the model supports the capability
        """
        return (
            model in cls._model_capabilities
            and capability in cls._model_capabilities[model]
        )

    @classmethod
    def get_provider_capabilities(cls, provider: str) -> Set[Capability]:
        """
        Get all capabilities supported by a provider.

        Args:
            provider: Provider name

        Returns:
            Set[Capability]: Set of supported capabilities
        """
        provider = provider.lower()
        return cls._provider_capabilities.get(provider, set())

    @classmethod
    def get_model_capabilities(cls, model: str) -> Set[Capability]:
        """
        Get all capabilities supported by a model.

        Args:
            model: Model name

        Returns:
            Set[Capability]: Set of supported capabilities
        """
        return cls._model_capabilities.get(model, set())

    @classmethod
    def register_model_capabilities(
        cls, model: str, capabilities: Set[Capability]
    ) -> None:
        """
        Register capabilities for a new model.

        Args:
            model: Model name
            capabilities: Set of capabilities the model supports
        """
        cls._model_capabilities[model] = capabilities

    @classmethod
    def register_provider_capabilities(
        cls, provider: str, capabilities: Set[Capability]
    ) -> None:
        """
        Register capabilities for a new provider.

        Args:
            provider: Provider name
            capabilities: Set of capabilities the provider supports
        """
        provider = provider.lower()
        cls._provider_capabilities[provider] = capabilities

    @classmethod
    def get_models_with_capability(cls, capability: Capability) -> List[str]:
        """
        Get all models that support a specific capability.

        Args:
            capability: The capability to check for

        Returns:
            List[str]: List of model names with the capability
        """
        return [
            model
            for model, caps in cls._model_capabilities.items()
            if capability in caps
        ]

    @classmethod
    def get_providers_with_capability(cls, capability: Capability) -> List[str]:
        """
        Get all providers that support a specific capability.

        Args:
            capability: The capability to check for

        Returns:
            List[str]: List of provider names with the capability
        """
        return [
            provider
            for provider, caps in cls._provider_capabilities.items()
            if capability in caps
        ]

    @classmethod
    def capability_exists(cls, capability_name: str) -> bool:
        """
        Check if a capability exists by name.

        Args:
            capability_name: Name of the capability

        Returns:
            bool: True if the capability exists
        """
        try:
            Capability[capability_name.upper()]
            return True
        except KeyError:
            return False

    @classmethod
    def get_capability_by_name(cls, capability_name: str) -> Optional[Capability]:
        """
        Get a capability enum value by name.

        Args:
            capability_name: Name of the capability

        Returns:
            Optional[Capability]: The capability enum value if it exists, None otherwise
        """
        try:
            return Capability[capability_name.upper()]
        except KeyError:
            return None

    @classmethod
    def capability_to_string(cls, capability: Capability) -> str:
        """
        Convert a capability enum to a human-readable string.

        Args:
            capability: The capability enum value

        Returns:
            str: Human-readable string
        """
        capability_mapping = {
            Capability.TEXT_GENERATION: "Text Generation",
            Capability.TEXT_EMBEDDINGS: "Text Embeddings",
            Capability.IMAGE_INPUT: "Image Input Processing",
            Capability.IMAGE_GENERATION: "Image Generation",
            Capability.STREAMING: "Streaming Responses",
            Capability.FUNCTIONS: "Function Calling",
            Capability.RERANKING: "Document Reranking",
            Capability.THINKING_BUDGET: "Thinking Budget",
            Capability.JSON_MODE: "JSON Mode",
            Capability.TOOL_CALLING: "Tool Calling",
            Capability.VISION: "Vision Capabilities",
        }
        return capability_mapping.get(
            capability, capability.name.replace("_", " ").title()
        )
