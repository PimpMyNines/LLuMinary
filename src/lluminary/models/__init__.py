"""Models module that provides various LLM implementations."""

from .base import LLM
from .router import get_llm_from_model, list_available_models, register_provider

# Provider configuration
_PROVIDER_CONFIGS = {}

def set_provider_config(provider_name, config):
    """Set configuration for a specific provider."""
    _PROVIDER_CONFIGS[provider_name] = config

def get_provider_config(provider_name):
    """Get configuration for a specific provider."""
    return _PROVIDER_CONFIGS.get(provider_name, {})

__all__ = [
    "LLM", 
    "get_llm_from_model", 
    "list_available_models",
    "register_provider",
    "set_provider_config",
    "get_provider_config"
]
