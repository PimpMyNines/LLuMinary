"""Validators for tool functions.

This module provides decorators and utilities for validating tool functions.
"""

import inspect
import json
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .registry import ToolValidationError


def validate_tool(tool: Callable) -> Callable:
    """Decorator to validate a tool function.

    Args:
        tool: The tool function to validate

    Returns:
        The validated tool function

    Raises:
        ToolValidationError: If validation fails
    """

    @wraps(tool)
    def wrapper(*args, **kwargs):
        # Validate docstring
        if not tool.__doc__:
            raise ToolValidationError(
                f"Tool '{tool.__name__}' must have a docstring")

        # Get and validate signature
        try:
            sig = inspect.signature(tool)
        except ValueError as e:
            raise ToolValidationError(
                f"Could not inspect tool '{tool.__name__}': {e}")

        # Validate return type
        if sig.return_annotation == inspect.Signature.empty:
            raise ToolValidationError(
                f"Tool '{tool.__name__}' must have a return type annotation"
            )

        # Validate parameter types
        for param_name, param in sig.parameters.items():
            if param.annotation == inspect.Parameter.empty:
                raise ToolValidationError(
                    f"Parameter '{param_name}' of tool '{tool.__name__}' "
                    "must have a type annotation"
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
                f"Tool '{tool.__name__}' parameters must be JSON-serializable"
            )

        # Execute the tool
        return tool(*args, **kwargs)

    return wrapper


def validate_return_type(expected_type: Type) -> Callable:
    """Decorator to validate a tool's return type at runtime.

    Args:
        expected_type: The expected return type

    Returns:
        Decorator function

    Example:
        @validate_return_type(Dict[str, Any])
        def my_tool() -> Dict[str, Any]:
            return {"result": "success"}
    """

    def decorator(tool: Callable) -> Callable:
        @wraps(tool)
        def wrapper(*args, **kwargs):
            result = tool(*args, **kwargs)

            if not _validate_type_structure(result, expected_type):
                raise ToolValidationError(
                    f"Tool '{tool.__name__}' returned {type(result)}, "
                    f"expected {expected_type}"
                )

            return result

        return wrapper

    return decorator


def _validate_type_structure(value: Any, expected_type: Type) -> bool:
    """Validate value against expected type, handling complex nested types.

    Args:
        value: The value to validate
        expected_type: The expected type

    Returns:
        True if value matches expected_type, False otherwise
    """
    # Special case for Any
    if expected_type is Any:
        return True

    # Handle None with Optional types
    if value is None:
        # Check if expected_type is Optional[...] (Union[..., None])
        origin = get_origin(expected_type)
        if origin is Union:
            args = get_args(expected_type)
            return type(None) in args
        return False

    # Get the origin and args for generic types (like List, Dict, etc.)
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Handle basic types
    if origin is None:
        return isinstance(value, expected_type)

    # Handle Union types
    if origin is Union:
        return any(_validate_type_structure(value, arg) for arg in args)

    # Handle List, Set, Tuple types
    if origin is list or origin is List:
        if not isinstance(value, list):
            return False
        if not args or args[0] is Any:
            return True
        return all(_validate_type_structure(item, args[0]) for item in value)

    if origin is set or origin is Set:
        if not isinstance(value, set):
            return False
        if not args or args[0] is Any:
            return True
        return all(_validate_type_structure(item, args[0]) for item in value)

    if origin is tuple or origin is Tuple:
        if not isinstance(value, tuple):
            return False

        # Handle Tuple[T, ...] - variable length tuple
        if args and len(args) == 2 and args[1] == Ellipsis:
            return all(
                _validate_type_structure(item, args[0]) for item in value)

        # Handle fixed length tuples
        if not args:
            return True

        # Check if lengths match for fixed length tuples
        if len(args) != len(value):
            return False

        # Validate each element against its expected type
        return all(
            _validate_type_structure(val, typ) for val, typ in zip(value, args))

    # Handle Dict types
    if origin is dict or origin is Dict:
        if not isinstance(value, dict):
            return False

        if not args:
            return True

        key_type, val_type = args

        if key_type is Any and val_type is Any:
            return True

        # Validate keys
        if key_type is not Any:
            key_valid = all(isinstance(k, key_type) for k in value.keys())
            if not key_valid:
                return False

        # Validate values
        if val_type is not Any:
            val_valid = all(
                _validate_type_structure(v, val_type) for v in value.values()
            )
            if not val_valid:
                return False

        return True

    # Fall back to simple isinstance check for other cases
    return isinstance(value, origin)


def validate_params(*types: Type, **named_types: Type) -> Callable:
    """Decorator to validate parameter types at runtime.

    Args:
        *types: Expected types for positional arguments
        **named_types: Expected types for keyword arguments

    Returns:
        Decorator function

    Example:
        @validate_params(str, int, optional_param=bool)
        def my_tool(text: str, number: int, optional_param: bool = False):
            return {"result": "success"}
    """

    def decorator(tool: Callable) -> Callable:
        sig = inspect.signature(tool)
        type_hints = get_type_hints(tool)

        @wraps(tool)
        def wrapper(*args, **kwargs):
            # Validate positional arguments
            for arg, expected_type in zip(args, types):
                if not _validate_type_structure(arg, expected_type):
                    raise ToolValidationError(
                        f"Argument {arg} is type {type(arg)}, "
                        f"expected {expected_type}"
                    )

            # Validate keyword arguments
            for name, value in kwargs.items():
                if name in named_types:
                    expected_type = named_types[name]
                    if not _validate_type_structure(value, expected_type):
                        raise ToolValidationError(
                            f"Argument '{name}' is type {type(value)}, "
                            f"expected {expected_type}"
                        )

            return tool(*args, **kwargs)

        return wrapper

    return decorator


def validate_nested_structure(value: Any) -> bool:
    """Validate that a value's nested structure is JSON-serializable.

    Args:
        value: The value to validate

    Returns:
        True if the value is JSON-serializable, False otherwise
    """
    if value is None:
        return True

    if isinstance(value, (str, int, float, bool)):
        return True

    if isinstance(value, (list, tuple)):
        return all(validate_nested_structure(item) for item in value)

    if isinstance(value, dict):
        # Check that all keys are strings (JSON requirement)
        if not all(isinstance(k, str) for k in value.keys()):
            return False

        # Check that all values are serializable
        return all(validate_nested_structure(v) for v in value.values())

    # Any other type is not JSON-serializable
    return False


def json_serializable(tool: Callable) -> Callable:
    """Decorator to validate that a tool's return value is JSON-serializable.

    Args:
        tool: The tool function to validate

    Returns:
        The validated tool function

    Example:
        @json_serializable
        def my_tool() -> Dict[str, Any]:
            return {"result": "success"}  # OK
            # return lambda x: x  # Would raise ToolValidationError
    """

    @wraps(tool)
    def wrapper(*args, **kwargs):
        result = tool(*args, **kwargs)

        # First check the nested structure without serializing
        if not validate_nested_structure(result):
            raise ToolValidationError(
                f"Tool '{tool.__name__}' return value contains non-JSON-serializable types"
            )

        # Then do the actual serialization
        try:
            json.dumps(result)
        except (TypeError, ValueError) as e:
            raise ToolValidationError(
                f"Tool '{tool.__name__}' return value is not JSON-serializable: {e}"
            )

        return result

    return wrapper
