"""Unit tests for tool validators.

This module provides comprehensive tests for the validators in the tools module.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytest

from lluminary.tools.registry import ToolValidationError
from lluminary.tools.validators import (
    _validate_type_structure,
    json_serializable,
    validate_nested_structure,
    validate_params,
    validate_return_type,
    validate_tool,
)


# Test functions for validate_tool decorator
@validate_tool
def valid_tool(param1: str, param2: int) -> Dict[str, Any]:
    """A valid tool with proper documentation and type hints."""
    return {"param1": param1, "param2": param2}


def test_validate_tool_success():
    """Test validate_tool decorator with valid tool."""
    result = valid_tool("test", 42)
    assert result == {"param1": "test", "param2": 42}


def test_validate_tool_missing_docstring():
    """Test validate_tool decorator with missing docstring."""
    with pytest.raises(ToolValidationError) as exc:

        @validate_tool
        def tool_no_doc(param: str) -> Dict[str, Any]:
            return {"param": param}

        tool_no_doc("test")
    assert "must have a docstring" in str(exc.value)


def test_validate_tool_missing_type_hints():
    """Test validate_tool decorator with missing type hints."""
    with pytest.raises(ToolValidationError) as exc:

        @validate_tool
        def tool_no_type_hints(param) -> Dict[str, Any]:
            """Tool without parameter type hints."""
            return {"param": param}

        tool_no_type_hints("test")
    assert "must have a type annotation" in str(exc.value)

    with pytest.raises(ToolValidationError) as exc:

        @validate_tool
        def tool_no_return_type(param: str):
            """Tool without return type hint."""
            return {"param": param}

        tool_no_return_type("test")
    assert "must have a return type annotation" in str(exc.value)


# Test functions for validate_return_type decorator
@validate_return_type(Dict[str, Any])
def tool_with_return_type() -> Dict[str, Any]:
    """Tool with validated return type."""
    return {"result": "success"}


@validate_return_type(Dict[str, Any])
def tool_with_wrong_return() -> Dict[str, Any]:
    """Tool that returns wrong type."""
    return ["wrong type"]


def test_validate_return_type_success():
    """Test validate_return_type with correct return type."""
    result = tool_with_return_type()
    assert result == {"result": "success"}


def test_validate_return_type_failure():
    """Test validate_return_type with incorrect return type."""
    with pytest.raises(ToolValidationError) as exc:
        tool_with_wrong_return()
    assert "expected" in str(exc.value)


# Test functions for validate_params decorator
@validate_params(str, int, optional_param=bool)
def tool_with_params(
    text: str, number: int, optional_param: bool = False
) -> Dict[str, Any]:
    """Tool with validated parameters."""
    return {"text": text, "number": number, "optional_param": optional_param}


def test_validate_params_success():
    """Test validate_params with correct parameter types."""
    result = tool_with_params("test", 42, optional_param=True)
    assert result == {"text": "test", "number": 42, "optional_param": True}


def test_validate_params_wrong_type():
    """Test validate_params with incorrect parameter type."""
    with pytest.raises(ToolValidationError) as exc:
        tool_with_params(123, 42)  # First param should be str
    assert "expected" in str(exc.value)


def test_validate_params_wrong_kwarg_type():
    """Test validate_params with incorrect keyword argument type."""
    with pytest.raises(ToolValidationError) as exc:
        tool_with_params("test", 42, optional_param="not a bool")
    assert "expected" in str(exc.value)


# Test functions for json_serializable decorator
@json_serializable
def tool_with_json_result() -> Dict[str, Any]:
    """Tool that returns JSON-serializable data."""
    return {"result": "success", "numbers": [1, 2, 3]}


@json_serializable
def tool_with_non_json_result() -> Dict[str, Any]:
    """Tool that returns non-JSON-serializable data."""
    return {"function": lambda x: x}


def test_json_serializable_success():
    """Test json_serializable with valid JSON data."""
    result = tool_with_json_result()
    assert result == {"result": "success", "numbers": [1, 2, 3]}


def test_json_serializable_failure():
    """Test json_serializable with non-JSON-serializable data."""
    with pytest.raises(ToolValidationError) as exc:
        tool_with_non_json_result()
    assert "non-JSON-serializable" in str(exc.value)


# Test combining multiple decorators
@validate_tool
@validate_return_type(Dict[str, Any])
@validate_params(str, int)
@json_serializable
def tool_with_all_validations(text: str, number: int) -> Dict[str, Any]:
    """Tool with all validations applied."""
    return {"text": text, "number": number}


def test_combined_decorators_success():
    """Test all decorators combined with valid input/output."""
    result = tool_with_all_validations("test", 42)
    assert result == {"text": "test", "number": 42}


def test_combined_decorators_type_error():
    """Test all decorators combined with type error."""
    with pytest.raises(ToolValidationError) as exc:
        tool_with_all_validations(42, "wrong type")  # Types swapped
    assert "expected" in str(exc.value)


def test_combined_decorators_non_json():
    """Test all decorators combined with non-JSON result."""

    @validate_tool
    @validate_return_type(Dict[str, Any])
    @validate_params(str)
    @json_serializable
    def bad_tool(text: str) -> Dict[str, Any]:
        """Tool that returns non-JSON data."""
        return {"func": lambda x: x}

    with pytest.raises(ToolValidationError) as exc:
        bad_tool("test")
    assert "non-JSON-serializable" in str(exc.value)


# Tests for validate_nested_structure function
def test_validate_nested_structure_primitive_types():
    """Test validate_nested_structure with primitive types."""
    assert validate_nested_structure(None) is True
    assert validate_nested_structure("text") is True
    assert validate_nested_structure(42) is True
    assert validate_nested_structure(3.14) is True
    assert validate_nested_structure(True) is True
    assert validate_nested_structure(False) is True


def test_validate_nested_structure_lists():
    """Test validate_nested_structure with lists."""
    assert validate_nested_structure([]) is True
    assert validate_nested_structure([1, 2, 3]) is True
    assert validate_nested_structure(["a", "b", "c"]) is True
    assert validate_nested_structure([1, "a", True]) is True
    assert validate_nested_structure([[1, 2], [3, 4]]) is True


def test_validate_nested_structure_dicts():
    """Test validate_nested_structure with dictionaries."""
    assert validate_nested_structure({}) is True
    assert validate_nested_structure({"a": 1, "b": 2}) is True
    assert validate_nested_structure({"a": [1, 2], "b": {"c": 3}}) is True


def test_validate_nested_structure_invalid():
    """Test validate_nested_structure with invalid structures."""
    # Dictionary with non-string keys (not JSON-serializable)
    assert validate_nested_structure({1: "value"}) is False

    # Function in a dictionary
    assert validate_nested_structure({"func": lambda x: x}) is False

    # Function in a list
    assert validate_nested_structure([1, lambda x: x, 3]) is False

    # Class instance that isn't a basic type
    class TestClass:
        pass

    assert validate_nested_structure({"instance": TestClass()}) is False


# Tests for _validate_type_structure function
def test_validate_type_structure_primitive_types():
    """Test _validate_type_structure with primitive types."""
    assert _validate_type_structure(42, int) is True
    assert _validate_type_structure("text", str) is True
    assert _validate_type_structure(3.14, float) is True
    assert _validate_type_structure(True, bool) is True

    # Type mismatches
    assert _validate_type_structure("text", int) is False
    assert _validate_type_structure(42, str) is False
    assert _validate_type_structure(3.14, int) is False


def test_validate_type_structure_any():
    """Test _validate_type_structure with Any type."""
    assert _validate_type_structure(42, Any) is True
    assert _validate_type_structure("text", Any) is True
    assert _validate_type_structure(None, Any) is True
    assert _validate_type_structure([1, 2, 3], Any) is True
    assert _validate_type_structure({"a": 1}, Any) is True
    assert _validate_type_structure(lambda x: x, Any) is True


def test_validate_type_structure_optional():
    """Test _validate_type_structure with Optional types."""
    # Optional[int] = Union[int, None]
    assert _validate_type_structure(42, Optional[int]) is True
    assert _validate_type_structure(None, Optional[int]) is True
    assert _validate_type_structure("text", Optional[int]) is False

    # Optional with complex type
    assert _validate_type_structure({"a": 1}, Optional[Dict[str, int]]) is True
    assert _validate_type_structure(None, Optional[Dict[str, int]]) is True


def test_validate_type_structure_union():
    """Test _validate_type_structure with Union types."""
    # Basic unions
    assert _validate_type_structure(42, Union[int, str]) is True
    assert _validate_type_structure("text", Union[int, str]) is True
    assert _validate_type_structure(3.14, Union[int, str]) is False

    # Complex unions
    assert _validate_type_structure({"a": 1}, Union[Dict[str, int], List[int]]) is True
    assert _validate_type_structure([1, 2, 3], Union[Dict[str, int], List[int]]) is True
    assert _validate_type_structure("text", Union[Dict[str, int], List[int]]) is False


def test_validate_type_structure_list():
    """Test _validate_type_structure with List types."""
    # Basic lists
    assert _validate_type_structure([1, 2, 3], List[int]) is True
    assert _validate_type_structure([], List[int]) is True  # Empty list is valid
    assert _validate_type_structure([1, "2", 3], List[int]) is False  # Mixed types
    assert _validate_type_structure("not a list", List[int]) is False

    # Lists with complex types
    assert _validate_type_structure([{"a": 1}, {"b": 2}], List[Dict[str, int]]) is True

    assert (
        _validate_type_structure([{"a": "1"}, {"b": 2}], List[Dict[str, int]]) is False
    )  # "1" is not an int


def test_validate_type_structure_set():
    """Test _validate_type_structure with Set types."""
    # Basic sets
    assert _validate_type_structure({1, 2, 3}, Set[int]) is True
    assert _validate_type_structure(set(), Set[int]) is True  # Empty set is valid
    assert _validate_type_structure({1, "2", 3}, Set[int]) is False  # Mixed types
    assert _validate_type_structure([1, 2, 3], Set[int]) is False  # Not a set

    # Sets with complex types not tested as they're not hashable


def test_validate_type_structure_tuple():
    """Test _validate_type_structure with Tuple types."""
    # Fixed length tuples
    assert _validate_type_structure((1, "text"), Tuple[int, str]) is True
    assert _validate_type_structure((1, 2), Tuple[int, str]) is False  # Wrong type
    assert _validate_type_structure((1,), Tuple[int, str]) is False  # Too short
    assert (
        _validate_type_structure((1, "text", True), Tuple[int, str]) is False
    )  # Too long

    # Variable length tuples (Tuple[T, ...])
    assert _validate_type_structure((1, 2, 3), Tuple[int, ...]) is True
    assert _validate_type_structure((), Tuple[int, ...]) is True  # Empty tuple
    assert (
        _validate_type_structure((1, "2", 3), Tuple[int, ...]) is False
    )  # Mixed types


def test_validate_type_structure_dict():
    """Test _validate_type_structure with Dict types."""
    # Basic dictionaries
    assert _validate_type_structure({"a": 1, "b": 2}, Dict[str, int]) is True
    assert _validate_type_structure({}, Dict[str, int]) is True  # Empty dict is valid
    assert (
        _validate_type_structure({"a": "1"}, Dict[str, int]) is False
    )  # Value not int
    assert _validate_type_structure({1: 1}, Dict[str, int]) is False  # Key not str
    assert _validate_type_structure([1, 2, 3], Dict[str, int]) is False  # Not a dict

    # Dictionaries with Any
    assert _validate_type_structure({"a": 1, "b": "text"}, Dict[str, Any]) is True
    assert _validate_type_structure({1: 1, "b": 2}, Dict[Any, int]) is True

    # Dictionaries with complex values
    assert (
        _validate_type_structure({"a": [1, 2], "b": [3, 4]}, Dict[str, List[int]])
        is True
    )

    assert (
        _validate_type_structure({"a": [1, "2"], "b": [3, 4]}, Dict[str, List[int]])
        is False
    )  # "2" is not an int


def test_validate_type_structure_complex_nested():
    """Test _validate_type_structure with complex nested types."""
    # A dictionary with string keys and values that are either integers or lists of strings
    complex_type = Dict[str, Union[int, List[str]]]

    assert (
        _validate_type_structure({"a": 1, "b": ["x", "y", "z"], "c": 3}, complex_type)
        is True
    )

    assert (
        _validate_type_structure(
            {"a": 1, "b": [1, 2, 3], "c": 3}, complex_type  # [1, 2, 3] not List[str]
        )
        is False
    )
