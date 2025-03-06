"""Tool registry for managing and validating LLM tools.

This module provides a centralized registry for managing tools that can be used
by the LLM handler. It includes validation, monitoring, and metadata management.
"""

import inspect
import json
import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Optional, Optional, Optional, Callable, Dict, List, Optional

from ..exceptions import LLMError

logger = logging.getLogger(__name__)


class ToolValidationError(LLMError):
    """Raised when tool validation fails."""

    pass


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""

    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    return_type: Any
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0


class ToolRegistry:
    """Registry for managing and validating LLM tools."""

    def __init__(self):
        """Initialize an empty tool registry."""
        self._tools: Dict[str, ToolMetadata] = {}

    def _validate_tool(self, tool: Callable, name: str) -> None:
        """Validate a tool's signature and documentation.

        Args:
            tool: The tool function to validate
            name: The name of the tool

        Raises:
            ToolValidationError: If validation fails
        """
        # Check for docstring
        if not tool.__doc__:
            raise ToolValidationError(f"Tool '{name}' must have a docstring")

        # Get signature
        try:
            sig = inspect.signature(tool)
        except ValueError as e:
            raise ToolValidationError(f"Could not inspect tool '{name}': {e}")

        # Validate return type annotation
        if sig.return_annotation == inspect.Signature.empty:
            raise ToolValidationError(
                f"Tool '{name}' must have a return type annotation"
            )

        # Validate parameter type hints
        for param_name, param in sig.parameters.items():
            if param.annotation == inspect.Parameter.empty:
                raise ToolValidationError(
                    f"Parameter '{param_name}' of tool '{name}' must have a type annotation"
                )

        # Validate parameters are JSON-serializable
        parameters = {
            name: str(param.annotation) for name, param in
            sig.parameters.items()
        }
        try:
            json.dumps(parameters)
        except TypeError:
            raise ToolValidationError(
                f"Tool '{name}' parameters must be JSON-serializable"
            )

    def _create_tool_metadata(
            self, tool: Callable, name: str, description: str = None
    ) -> ToolMetadata:
        """Create metadata for a tool.

        Args:
            tool: The tool function
            name: Name of the tool
            description: Optional description (defaults to docstring)

        Returns:
            ToolMetadata object
        """
        sig = inspect.signature(tool)

        return ToolMetadata(
            name=name,
            description=description or tool.__doc__ or "",
            function=tool,
            parameters={
                name: str(param.annotation) for name, param in
                sig.parameters.items()
            },
            return_type=str(sig.return_annotation),
        )

    def _wrap_tool(self, tool: Callable, name: str) -> Callable:
        """Wrap a tool to add monitoring.

        Args:
            tool: The tool function to wrap
            name: Name of the tool

        Returns:
            Wrapped tool function
        """

        @wraps(tool)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            try:
                result = tool(*args, **kwargs)
                self._tools[name].success_count += 1
                return result
            except Exception:
                self._tools[name].failure_count += 1
                raise
            finally:
                execution_time = time.time() - start_time
                self._tools[name].total_execution_time += execution_time
                total_calls = (
                        self._tools[name].success_count + self._tools[
                    name].failure_count
                )
                self._tools[name].average_execution_time = (
                        self._tools[name].total_execution_time / total_calls
                )

        return wrapped

    def register_tool(
            self, tool: Callable, name: str = None, description: str = None
    ) -> None:
        """Register a tool with optional custom name and description.

        Args:
            tool: The tool function to register
            name: Optional custom name (defaults to function name)
            description: Optional description (defaults to docstring)

        Raises:
            ToolValidationError: If tool validation fails
        """
        name = name or tool.__name__

        # Validate the tool
        self._validate_tool(tool, name)

        # Create metadata
        metadata = self._create_tool_metadata(tool, name, description)

        # Wrap the tool for monitoring
        wrapped_tool = self._wrap_tool(tool, name)
        metadata.function = wrapped_tool

        # Store in registry
        self._tools[name] = metadata
        logger.info(f"Registered tool '{name}'")

    def register_tools(self, tools: List[Callable]) -> None:
        """Register multiple tools at once.

        Args:
            tools: List of tool functions to register
        """
        for tool in tools:
            self.register_tool(tool)

    def get_tool(self, name: str) -> Optional[Callable]:
        """Retrieve a registered tool by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            The tool function if found, None otherwise
        """
        if name in self._tools:
            return self._tools[name].function
        return None

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their metadata.

        Returns:
            List of tool metadata dictionaries
        """
        return [
            {
                "name": metadata.name,
                "description": metadata.description,
                "parameters": metadata.parameters,
                "return_type": metadata.return_type,
                "success_count": metadata.success_count,
                "failure_count": metadata.failure_count,
                "average_execution_time": metadata.average_execution_time,
            }
            for metadata in self._tools.values()
        ]

    def get_tool_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get execution statistics for a tool.

        Args:
            name: Name of the tool

        Returns:
            Dictionary of tool statistics if found, None otherwise
        """
        if name in self._tools:
            metadata = self._tools[name]
            return {
                "success_count": metadata.success_count,
                "failure_count": metadata.failure_count,
                "average_execution_time": metadata.average_execution_time,
                "total_execution_time": metadata.total_execution_time,
            }
        return None
