"""Tools package for LLM handler.

This package provides tool registration and management functionality.
"""

from .registry import ToolMetadata, ToolRegistry, ToolValidationError
from .validators import (json_serializable, validate_params,
                         validate_return_type, validate_tool)

__all__ = [
    "ToolRegistry",
    "ToolValidationError",
    "ToolMetadata",
    "validate_tool",
    "validate_return_type",
    "validate_params",
    "json_serializable",
]
