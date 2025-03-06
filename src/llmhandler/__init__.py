"""
LLMHandler - A Python package for handling LLM operations across multiple providers.
"""

__version__ = "0.1.0"

from .exceptions import LLMMistake
from .models.router import get_llm_from_model

__all__ = ["get_llm_from_model", "LLMMistake"]
