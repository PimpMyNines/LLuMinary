"""
LLuMinary: A versatile interface for illuminating the path to multiple LLM providers.

LLuMinary provides a clean, extensible interface for working with various LLM providers
including OpenAI, Anthropic, Google, and Cohere. It handles all the complexity of
provider-specific implementations, message formatting, and error handling, allowing
you to focus on building applications.
"""

from .exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMMistake,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMValidationError,
)
from .handler import LLMHandler as LLuMinary
from .models.router import get_llm_from_model, list_available_models, register_provider
from .version import __version__

__all__ = [
    "LLMAuthenticationError",
    "LLMConnectionError",
    "LLMError",
    "LLMMistake",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMValidationError",
    "LLuMinary",
    "__version__",
    "get_llm_from_model",
    "list_available_models",
    "register_provider",
]
