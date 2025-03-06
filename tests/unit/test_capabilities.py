"""Tests for the capabilities module."""

from lluminary.models.capabilities import Capability, CapabilityRegistry


class TestCapabilities:
    """Test the capabilities module."""

    def test_capability_enum(self):
        """Test the Capability enum."""
        assert Capability.TEXT_GENERATION is not None
        assert Capability.TEXT_EMBEDDINGS is not None
        assert Capability.IMAGE_INPUT is not None
        assert Capability.IMAGE_GENERATION is not None
        assert Capability.STREAMING is not None
        assert Capability.FUNCTIONS is not None
        assert Capability.RERANKING is not None
        assert Capability.THINKING_BUDGET is not None
        assert Capability.JSON_MODE is not None
        assert Capability.TOOL_CALLING is not None
        assert Capability.VISION is not None

    def test_provider_has_capability(self):
        """Test provider_has_capability method."""
        # OpenAI has TEXT_GENERATION capability
        assert (
            CapabilityRegistry.provider_has_capability(
                "openai", Capability.TEXT_GENERATION
            )
            is True
        )

        # OpenAI has TEXT_EMBEDDINGS capability
        assert (
            CapabilityRegistry.provider_has_capability(
                "openai", Capability.TEXT_EMBEDDINGS
            )
            is True
        )

        # Anthropic has THINKING_BUDGET capability
        assert (
            CapabilityRegistry.provider_has_capability(
                "anthropic", Capability.THINKING_BUDGET
            )
            is True
        )

        # Anthropic does not have JSON_MODE capability
        assert (
            CapabilityRegistry.provider_has_capability(
                "anthropic", Capability.JSON_MODE
            )
            is False
        )

        # Case insensitivity
        assert (
            CapabilityRegistry.provider_has_capability(
                "OPENAI", Capability.TEXT_GENERATION
            )
            is True
        )

        # Unknown provider
        assert (
            CapabilityRegistry.provider_has_capability(
                "unknown", Capability.TEXT_GENERATION
            )
            is False
        )

    def test_model_has_capability(self):
        """Test model_has_capability method."""
        # gpt-4o has IMAGE_INPUT capability
        assert (
            CapabilityRegistry.model_has_capability("gpt-4o", Capability.IMAGE_INPUT)
            is True
        )

        # claude-sonnet-3.5 has THINKING_BUDGET capability
        assert (
            CapabilityRegistry.model_has_capability(
                "claude-sonnet-3.5", Capability.THINKING_BUDGET
            )
            is True
        )

        # claude-sonnet-3.5 does not have JSON_MODE capability
        assert (
            CapabilityRegistry.model_has_capability(
                "claude-sonnet-3.5", Capability.JSON_MODE
            )
            is False
        )

        # Unknown model
        assert (
            CapabilityRegistry.model_has_capability(
                "unknown-model", Capability.TEXT_GENERATION
            )
            is False
        )

    def test_get_provider_capabilities(self):
        """Test get_provider_capabilities method."""
        # Get OpenAI capabilities
        openai_capabilities = CapabilityRegistry.get_provider_capabilities("openai")
        assert Capability.TEXT_GENERATION in openai_capabilities
        assert Capability.IMAGE_GENERATION in openai_capabilities
        assert len(openai_capabilities) > 0

        # Get Anthropic capabilities
        anthropic_capabilities = CapabilityRegistry.get_provider_capabilities(
            "anthropic"
        )
        assert Capability.TEXT_GENERATION in anthropic_capabilities
        assert Capability.THINKING_BUDGET in anthropic_capabilities
        assert len(anthropic_capabilities) > 0

        # Unknown provider
        unknown_capabilities = CapabilityRegistry.get_provider_capabilities("unknown")
        assert len(unknown_capabilities) == 0

    def test_get_model_capabilities(self):
        """Test get_model_capabilities method."""
        # Get gpt-4o capabilities
        gpt4o_capabilities = CapabilityRegistry.get_model_capabilities("gpt-4o")
        assert Capability.TEXT_GENERATION in gpt4o_capabilities
        assert Capability.IMAGE_INPUT in gpt4o_capabilities
        assert len(gpt4o_capabilities) > 0

        # Get claude-sonnet-3.5 capabilities
        claude_capabilities = CapabilityRegistry.get_model_capabilities(
            "claude-sonnet-3.5"
        )
        assert Capability.TEXT_GENERATION in claude_capabilities
        assert Capability.THINKING_BUDGET in claude_capabilities
        assert len(claude_capabilities) > 0

        # Unknown model
        unknown_capabilities = CapabilityRegistry.get_model_capabilities(
            "unknown-model"
        )
        assert len(unknown_capabilities) == 0

    def test_register_capabilities(self):
        """Test registering new capabilities."""
        # Register new model capabilities
        CapabilityRegistry.register_model_capabilities(
            "custom-model", {Capability.TEXT_GENERATION, Capability.STREAMING}
        )
        assert (
            CapabilityRegistry.model_has_capability(
                "custom-model", Capability.TEXT_GENERATION
            )
            is True
        )
        assert (
            CapabilityRegistry.model_has_capability(
                "custom-model", Capability.STREAMING
            )
            is True
        )
        assert (
            CapabilityRegistry.model_has_capability(
                "custom-model", Capability.IMAGE_INPUT
            )
            is False
        )

        # Register new provider capabilities
        CapabilityRegistry.register_provider_capabilities(
            "custom-provider", {Capability.TEXT_GENERATION, Capability.TEXT_EMBEDDINGS}
        )
        assert (
            CapabilityRegistry.provider_has_capability(
                "custom-provider", Capability.TEXT_GENERATION
            )
            is True
        )
        assert (
            CapabilityRegistry.provider_has_capability(
                "custom-provider", Capability.TEXT_EMBEDDINGS
            )
            is True
        )
        assert (
            CapabilityRegistry.provider_has_capability(
                "custom-provider", Capability.IMAGE_GENERATION
            )
            is False
        )

    def test_get_models_with_capability(self):
        """Test get_models_with_capability method."""
        # Get models with TEXT_GENERATION capability
        text_generation_models = CapabilityRegistry.get_models_with_capability(
            Capability.TEXT_GENERATION
        )
        assert "gpt-4o" in text_generation_models
        assert "claude-sonnet-3.5" in text_generation_models
        assert len(text_generation_models) > 0

        # Get models with RERANKING capability
        reranking_models = CapabilityRegistry.get_models_with_capability(
            Capability.RERANKING
        )
        assert "cohere-rerank" in reranking_models

        # Custom model
        CapabilityRegistry.register_model_capabilities(
            "custom-reranking-model", {Capability.RERANKING}
        )
        updated_reranking_models = CapabilityRegistry.get_models_with_capability(
            Capability.RERANKING
        )
        assert "custom-reranking-model" in updated_reranking_models

    def test_get_providers_with_capability(self):
        """Test get_providers_with_capability method."""
        # Get providers with TEXT_GENERATION capability
        text_generation_providers = CapabilityRegistry.get_providers_with_capability(
            Capability.TEXT_GENERATION
        )
        assert "openai" in text_generation_providers
        assert "anthropic" in text_generation_providers
        assert len(text_generation_providers) > 0

        # Get providers with RERANKING capability
        reranking_providers = CapabilityRegistry.get_providers_with_capability(
            Capability.RERANKING
        )
        assert "openai" in reranking_providers
        assert "cohere" in reranking_providers

        # Custom provider
        CapabilityRegistry.register_provider_capabilities(
            "custom-reranking-provider", {Capability.RERANKING}
        )
        updated_reranking_providers = CapabilityRegistry.get_providers_with_capability(
            Capability.RERANKING
        )
        assert "custom-reranking-provider" in updated_reranking_providers

    def test_capability_existence_and_name(self):
        """Test capability_exists and get_capability_by_name methods."""
        assert CapabilityRegistry.capability_exists("TEXT_GENERATION") is True
        assert CapabilityRegistry.capability_exists("NONEXISTENT") is False

        assert (
            CapabilityRegistry.get_capability_by_name("TEXT_GENERATION")
            == Capability.TEXT_GENERATION
        )
        assert CapabilityRegistry.get_capability_by_name("NONEXISTENT") is None

    def test_capability_to_string(self):
        """Test capability_to_string method."""
        assert (
            CapabilityRegistry.capability_to_string(Capability.TEXT_GENERATION)
            == "Text Generation"
        )
        assert (
            CapabilityRegistry.capability_to_string(Capability.IMAGE_INPUT)
            == "Image Input Processing"
        )
        assert (
            CapabilityRegistry.capability_to_string(Capability.THINKING_BUDGET)
            == "Thinking Budget"
        )
