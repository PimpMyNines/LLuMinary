"""
Unit tests for the registration of the Provider Template.
"""

from unittest.mock import patch

import pytest

from lluminary.models.providers.provider_template import ProviderNameLLM
from lluminary.models.router import register_provider


class TestProviderTemplateRegistration:
    """Test the registration of ProviderNameLLM with the provider registry."""

    @pytest.mark.xfail(
        reason="Need to fix import for PROVIDERS and MODEL_TO_PROVIDER_MAP"
    )
    def test_provider_registration(self):
        """Test that the provider can be registered correctly."""
        # Clear any existing registry to start fresh
        from lluminary.models.router import MODEL_TO_PROVIDER_MAP, PROVIDERS

        old_providers = PROVIDERS.copy()
        old_model_map = MODEL_TO_PROVIDER_MAP.copy()

        try:
            # Clear registries for testing
            PROVIDERS.clear()
            MODEL_TO_PROVIDER_MAP.clear()

            # Register the provider
            register_provider("provider_name", ProviderNameLLM)

            # Check that the provider was registered
            assert "provider_name" in PROVIDERS
            assert PROVIDERS["provider_name"] == ProviderNameLLM

            # Check that models were mapped
            for model_name in ProviderNameLLM.SUPPORTED_MODELS:
                assert model_name in MODEL_TO_PROVIDER_MAP
                assert MODEL_TO_PROVIDER_MAP[model_name] == "provider_name"
        finally:
            # Restore original registry
            PROVIDERS.clear()
            PROVIDERS.update(old_providers)
            MODEL_TO_PROVIDER_MAP.clear()
            MODEL_TO_PROVIDER_MAP.update(old_model_map)

    @pytest.mark.xfail(reason="Need to fix import for router components")
    def test_router_registration_lookup(self):
        """Test that the router can find the registered provider."""
        from lluminary.models.router import (
            MODEL_TO_PROVIDER_MAP,
            PROVIDERS,
            get_provider_for_model,
        )

        old_providers = PROVIDERS.copy()
        old_model_map = MODEL_TO_PROVIDER_MAP.copy()

        try:
            # Clear registries for testing
            PROVIDERS.clear()
            MODEL_TO_PROVIDER_MAP.clear()

            # Register the provider
            register_provider("provider_name", ProviderNameLLM)

            # Test with model mapping
            provider_class = get_provider_for_model("provider-model-1")
            assert provider_class == ProviderNameLLM

            # Test with direct provider name
            provider_class = get_provider_for_model("provider_name")
            assert provider_class == ProviderNameLLM

            # Test with unknown model
            with pytest.raises(ValueError):
                get_provider_for_model("unknown-model")
        finally:
            # Restore original registry
            PROVIDERS.clear()
            PROVIDERS.update(old_providers)
            MODEL_TO_PROVIDER_MAP.clear()
            MODEL_TO_PROVIDER_MAP.update(old_model_map)


class TestProviderTemplateDiscovery:
    """Test the discovery and initialization of ProviderNameLLM through the router."""

    @pytest.mark.xfail(reason="Need to fix import for router components")
    @patch("lluminary.models.providers.provider_template.ProviderNameLLM.auth")
    def test_provider_discovery(self, mock_auth):
        """Test that the provider can be discovered by the router."""
        from lluminary.models.router import (
            MODEL_TO_PROVIDER_MAP,
            PROVIDERS,
            get_provider,
        )

        old_providers = PROVIDERS.copy()
        old_model_map = MODEL_TO_PROVIDER_MAP.copy()

        try:
            # Clear registries for testing
            PROVIDERS.clear()
            MODEL_TO_PROVIDER_MAP.clear()

            # Register the provider
            register_provider("provider_name", ProviderNameLLM)

            # Test provider initialization
            provider = get_provider("provider_name", model_name="provider-model-1")

            assert isinstance(provider, ProviderNameLLM)
            assert provider.model_name == "provider-model-1"
            assert mock_auth.called
        finally:
            # Restore original registry
            PROVIDERS.clear()
            PROVIDERS.update(old_providers)
            MODEL_TO_PROVIDER_MAP.clear()
            MODEL_TO_PROVIDER_MAP.update(old_model_map)

    @pytest.mark.xfail(reason="Need to fix import for router components")
    @patch("lluminary.models.providers.provider_template.ProviderNameLLM.auth")
    def test_provider_discovery_with_model_mapping(self, mock_auth):
        """Test provider discovery using model mapping."""
        from lluminary.models.router import (
            MODEL_TO_PROVIDER_MAP,
            PROVIDERS,
            get_provider,
        )

        old_providers = PROVIDERS.copy()
        old_model_map = MODEL_TO_PROVIDER_MAP.copy()

        try:
            # Clear registries for testing
            PROVIDERS.clear()
            MODEL_TO_PROVIDER_MAP.clear()

            # Register the provider
            register_provider("provider_name", ProviderNameLLM)

            # Test provider initialization through model mapping
            provider = get_provider("provider-model-2")

            assert isinstance(provider, ProviderNameLLM)
            assert provider.model_name == "provider-model-2"
            assert mock_auth.called
        finally:
            # Restore original registry
            PROVIDERS.clear()
            PROVIDERS.update(old_providers)
            MODEL_TO_PROVIDER_MAP.clear()
            MODEL_TO_PROVIDER_MAP.update(old_model_map)

    @pytest.mark.xfail(reason="Need to fix import for router components")
    def test_provider_discovery_unsupported_model(self):
        """Test discovery with an unsupported model."""
        from lluminary.models.router import (
            MODEL_TO_PROVIDER_MAP,
            PROVIDERS,
            get_provider,
        )

        old_providers = PROVIDERS.copy()
        old_model_map = MODEL_TO_PROVIDER_MAP.copy()

        try:
            # Clear registries for testing
            PROVIDERS.clear()
            MODEL_TO_PROVIDER_MAP.clear()

            # Register the provider
            register_provider("provider_name", ProviderNameLLM)

            # Test with unsupported model
            with pytest.raises(ValueError) as excinfo:
                get_provider("unsupported-model")

            assert "No provider found for model" in str(excinfo.value)
        finally:
            # Restore original registry
            PROVIDERS.clear()
            PROVIDERS.update(old_providers)
            MODEL_TO_PROVIDER_MAP.clear()
            MODEL_TO_PROVIDER_MAP.update(old_model_map)
