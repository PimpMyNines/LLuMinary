"""
Router for LLM model selection with extensible provider registration.
"""

from typing import Any, Dict, List, Type

from .base import LLM

# Default provider classes
_DEFAULT_PROVIDERS: Dict[str, Type[LLM]] = {}

# Registry to hold provider classes dynamically
PROVIDER_REGISTRY: Dict[str, Type[LLM]] = {}

# Mapping of friendly names to model providers and their specific model names
MODEL_MAPPINGS: Dict[str, Dict[str, str]] = {
    # OpenAI Models
    "gpt-4.5": {"provider": "openai", "model": "gpt-4.5-preview"},
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
    "o1": {"provider": "openai", "model": "o1"},
    "o3-mini": {"provider": "openai", "model": "o3-mini"},
    # Anthropic Models
    "claude-haiku-3.5": {"provider": "anthropic",
                         "model": "claude-3-5-haiku-20241022"},
    "claude-sonnet-3.5": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
    },
    "claude-sonnet-3.7": {
        "provider": "anthropic",
        "model": "claude-3-7-sonnet-20250219",
    },
    # Bedrock Models
    "bedrock-claude-haiku-3.5": {
        "provider": "bedrock",
        "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    },
    "bedrock-claude-sonnet-3.5-v1": {
        "provider": "bedrock",
        "model": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    },
    "bedrock-claude-sonnet-3.5-v2": {
        "provider": "bedrock",
        "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    },
    "bedrock-claude-sonnet-3.7": {
        "provider": "bedrock",
        "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    },
    # Google Models
    "gemini-2.0-flash": {"provider": "google", "model": "gemini-2.0-flash"},
    "gemini-2.0-flash-lite": {
        "provider": "google",
        "model": "gemini-2.0-flash-lite-preview-02-05",
    },
    "gemini-2.0-pro": {"provider": "google",
                       "model": "gemini-2.0-pro-exp-02-05"},
    "gemini-2.0-flash-thinking": {
        "provider": "google",
        "model": "gemini-2.0-flash-thinking-exp-01-21",
    },
    # Cohere Models
    "cohere-command": {"provider": "cohere", "model": "command"},
    "cohere-command-light": {"provider": "cohere", "model": "command-light"},
    "cohere-command-r": {"provider": "cohere", "model": "command-r"},
    "cohere-command-r-plus": {"provider": "cohere", "model": "command-r-plus"},
}


def register_provider(provider_name: str, provider_class: Type[LLM]) -> None:
    """
    Register a new LLM provider class.

    Args:
        provider_name (str): Name of the provider (e.g., 'openai', 'anthropic')
        provider_class (Type[LLM]): The provider class, must extend LLM base class

    Raises:
        TypeError: If provider_class is not a subclass of LLM
    """
    if not issubclass(provider_class, LLM):
        raise TypeError(
            f"Provider class must be a subclass of LLM, got {provider_class}"
        )

    PROVIDER_REGISTRY[provider_name] = provider_class


def get_llm_from_model(model_name: str, **kwargs: Any) -> LLM:
    """
    Get the appropriate LLM instance based on the friendly model name.

    Args:
        model_name (str): Friendly name of the model
        **kwargs: Additional configuration parameters for the LLM

    Returns:
        LLM: An instance of the appropriate LLM provider

    Raises:
        ValueError: If the model name is not recognized or provider isn't registered
    """
    # Import default providers here to avoid circular imports
    if not PROVIDER_REGISTRY:
        _load_default_providers()

    if model_name not in MODEL_MAPPINGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(MODEL_MAPPINGS.keys())}"
        )

    provider_info = MODEL_MAPPINGS[model_name]
    provider_name = provider_info["provider"]
    actual_model = provider_info["model"]

    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Provider {provider_name} not registered. Available providers: {list(PROVIDER_REGISTRY.keys())}"
        )

    provider_class = PROVIDER_REGISTRY[provider_name]
    return provider_class(actual_model, **kwargs)


def _load_default_providers() -> None:
    """Load the default provider classes."""
    from .providers import AnthropicLLM, BedrockLLM, GoogleLLM, OpenAILLM

    register_provider("openai", OpenAILLM)
    register_provider("anthropic", AnthropicLLM)
    register_provider("bedrock", BedrockLLM)
    register_provider("google", GoogleLLM)


def list_available_models() -> Dict[str, List[str]]:
    """
    Get a dictionary of all available models grouped by provider.

    Returns:
        Dict[str, List[str]]: Dictionary mapping provider names to lists of their friendly model names
    """
    # Ensure providers are loaded
    if not PROVIDER_REGISTRY:
        _load_default_providers()

    models_by_provider: Dict[str, List[str]] = {}
    for friendly_name, info in MODEL_MAPPINGS.items():
        provider_name = info["provider"]
        if provider_name not in models_by_provider:
            models_by_provider[provider_name] = []
        models_by_provider[provider_name].append(friendly_name)
    return models_by_provider


def register_model(friendly_name: str, provider_name: str,
                   model_id: str) -> None:
    """
    Register a new model with the system.

    Args:
        friendly_name (str): User-friendly name for the model
        provider_name (str): Name of the provider (must be registered)
        model_id (str): Provider-specific model identifier

    Raises:
        ValueError: If the provider is not registered
    """
    # Ensure providers are loaded
    if not PROVIDER_REGISTRY:
        _load_default_providers()

    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Provider {provider_name} not registered. Available providers: {list(PROVIDER_REGISTRY.keys())}"
        )

    MODEL_MAPPINGS[friendly_name] = {"provider": provider_name,
                                     "model": model_id}
