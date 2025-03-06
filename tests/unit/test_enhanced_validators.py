"""Tests for enhanced validators for complex nested types."""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytest

from lluminary.tools.registry import ToolValidationError
from lluminary.tools.validators import (
    _validate_type_structure,
    json_serializable,
    validate_nested_structure,
    validate_params,
    validate_return_type,
)


class TestTypeValidation:
    """Tests for type validation functions."""

    def test_simple_types(self):
        """Test validation of simple types."""
        assert _validate_type_structure(10, int) is True
        assert _validate_type_structure("test", str) is True
        assert _validate_type_structure(True, bool) is True
        assert _validate_type_structure(10.5, float) is True

        assert _validate_type_structure("test", int) is False
        assert _validate_type_structure(10, str) is False

    def test_list_types(self):
        """Test validation of list types."""
        assert _validate_type_structure([1, 2, 3], List[int]) is True
        assert _validate_type_structure([], List[int]) is True
        assert _validate_type_structure(["a", "b"], List[str]) is True
        assert _validate_type_structure([1, 2, 3], List[str]) is False
        assert _validate_type_structure(["a", 1], List[str]) is False
        assert _validate_type_structure({"a": 1}, List[Any]) is False

    def test_dict_types(self):
        """Test validation of dictionary types."""
        assert _validate_type_structure({"a": 1, "b": 2}, Dict[str, int]) is True
        assert _validate_type_structure({}, Dict[str, int]) is True
        assert _validate_type_structure({"a": "A", "b": "B"}, Dict[str, str]) is True
        assert (
            _validate_type_structure({"a": 1, "b": "2"}, Dict[str, Union[int, str]])
            is True
        )
        assert _validate_type_structure({1: "a", 2: "b"}, Dict[int, str]) is True
        assert _validate_type_structure({"a": 1, "b": "c"}, Dict[str, int]) is False
        assert _validate_type_structure(["a", "b"], Dict[Any, Any]) is False

    def test_set_types(self):
        """Test validation of set types."""
        assert _validate_type_structure({1, 2, 3}, Set[int]) is True
        assert _validate_type_structure(set(), Set[int]) is True
        assert _validate_type_structure({"a", "b"}, Set[str]) is True
        assert _validate_type_structure({1, 2, 3}, Set[str]) is False
        assert _validate_type_structure([1, 2, 3], Set[int]) is False

    def test_tuple_types(self):
        """Test validation of tuple types."""
        assert _validate_type_structure((1, "a"), Tuple[int, str]) is True
        # Using an ellipsis for variable-length tuples
        assert _validate_type_structure((1, 2, 3), Tuple[int, ...]) is True
        assert _validate_type_structure((), Tuple[int, ...]) is True
        assert _validate_type_structure((1, "a", True), Tuple[int, str, bool]) is True
        assert (
            _validate_type_structure((1, 2), Tuple[int, str]) is False
        )  # 2 is not str
        assert (
            _validate_type_structure([1, "a"], Tuple[int, str]) is False
        )  # list, not tuple

    def test_simple_nested_types(self):
        """Test validation of simple nested types."""
        # Simple nested dictionary
        simple_dict = {"key": "value", "count": 1}
        simple_dict_type = Dict[str, Union[str, int]]
        assert _validate_type_structure(simple_dict, simple_dict_type) is True

        # Nested list in dictionary
        nested_list_dict = {"items": [1, 2, 3]}
        nested_list_type = Dict[str, List[int]]
        assert _validate_type_structure(nested_list_dict, nested_list_type) is True

        # Simple string in place of list
        wrong_type = {"items": "not a list"}
        assert _validate_type_structure(wrong_type, nested_list_type) is False

    def test_complex_nested_types(self):
        """Test validation of complex nested types."""
        # List of dictionaries with mixed types
        complex_struct = [
            {"name": "Alice", "scores": [95, 87]},
            {"name": "Bob", "scores": [85, 92]},
        ]
        complex_type = List[Dict[str, Union[str, List[int]]]]
        assert _validate_type_structure(complex_struct, complex_type) is True

        # Dictionary with list of dictionaries (single nesting level)
        user_list = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
        user_list_type = Dict[str, List[Dict[str, Union[int, str]]]]
        assert _validate_type_structure(user_list, user_list_type) is True

    def test_optional_types(self):
        """Test validation of Optional types."""
        assert _validate_type_structure(None, Optional[int]) is True
        assert _validate_type_structure(10, Optional[int]) is True
        assert _validate_type_structure("test", Optional[str]) is True
        assert _validate_type_structure(None, Optional[str]) is True
        assert _validate_type_structure(None, int) is False

    def test_simple_union_types(self):
        """Test validation of simple Union types."""
        # Basic union types with simple values
        assert _validate_type_structure(10, Union[int, str]) is True
        assert _validate_type_structure("test", Union[int, str]) is True

        # Boolean is a subclass of int in Python, so this will actually be True
        # Note: In most type systems, bool is a separate type from int
        assert _validate_type_structure(True, Union[int, str]) is True

        # Test something that's definitely not in the union
        assert _validate_type_structure(3.14, Union[int, str]) is False

    def test_complex_union_types(self):
        """Test validation of complex Union types."""
        # Union of simple types
        simple_union = Union[int, str, bool]
        assert _validate_type_structure(10, simple_union) is True
        assert _validate_type_structure("test", simple_union) is True
        assert _validate_type_structure(True, simple_union) is True
        assert _validate_type_structure(10.5, simple_union) is False

        # Union of container types
        container_union = Union[List[int], Dict[str, str]]
        assert _validate_type_structure([1, 2, 3], container_union) is True
        assert _validate_type_structure({"a": "A"}, container_union) is True
        assert _validate_type_structure("not a container", container_union) is False


class TestNestedStructureValidation:
    """Tests for nested structure validation."""

    def test_simple_values(self):
        """Test validation of simple JSON-serializable values."""
        assert validate_nested_structure(None) is True
        assert validate_nested_structure(10) is True
        assert validate_nested_structure("test") is True
        assert validate_nested_structure(True) is True
        assert validate_nested_structure(10.5) is True

    def test_collections(self):
        """Test validation of collections."""
        assert validate_nested_structure([1, 2, 3]) is True
        assert validate_nested_structure({"a": 1, "b": 2}) is True
        assert validate_nested_structure((1, 2, 3)) is True

        # Sets are not directly JSON-serializable
        assert validate_nested_structure({1, 2, 3}) is False

    def test_nested_structures(self):
        """Test validation of nested structures."""
        assert (
            validate_nested_structure(
                {
                    "users": [
                        {
                            "name": "Alice",
                            "age": 30,
                            "active": True,
                            "scores": [95, 87, 91],
                        },
                        {
                            "name": "Bob",
                            "age": 25,
                            "active": False,
                            "scores": [88, 92, 85],
                        },
                    ],
                    "metadata": {"total": 2, "average_age": 27.5},
                }
            )
            is True
        )

    def test_non_serializable_values(self):
        """Test validation of non-serializable values."""
        assert validate_nested_structure(lambda x: x) is False
        assert validate_nested_structure({1: "value"}) is False  # Non-string keys
        assert validate_nested_structure({"func": lambda x: x}) is False
        assert validate_nested_structure([1, lambda x: x, 3]) is False


class TestDecorators:
    """Tests for validator decorators."""

    def test_validate_return_type_with_nested_types(self):
        """Test validate_return_type with nested types."""

        @validate_return_type(Dict[str, List[Dict[str, Any]]])
        def nested_return() -> Dict[str, List[Dict[str, Any]]]:
            """Function that returns a nested structure."""
            return {"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}

        result = nested_return()
        assert result == {
            "items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]
        }

        @validate_return_type(Dict[str, List[Dict[str, str]]])
        def invalid_nested_return() -> Dict[str, List[Dict[str, str]]]:
            """Function that returns an invalid nested structure."""
            return {
                "items": [
                    {"id": 1, "name": "Item 1"},  # id is int, not str
                    {"id": 2, "name": "Item 2"},
                ]
            }

        with pytest.raises(ToolValidationError):
            invalid_nested_return()

    def test_validate_params_with_nested_types(self):
        """Test validate_params with nested types."""

        @validate_params(Dict[str, List[Dict[str, Any]]])
        def nested_param(data: Dict[str, List[Dict[str, Any]]]) -> bool:
            """Function that takes a nested structure parameter."""
            return True

        assert (
            nested_param(
                {"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]}
            )
            is True
        )

        with pytest.raises(ToolValidationError):
            nested_param("not a dict")

    def test_json_serializable_with_nested_types(self):
        """Test json_serializable with nested types."""

        @json_serializable
        def serializable_return() -> Dict[str, Any]:
            """Function that returns a serializable nested structure."""
            return {
                "items": [
                    {"id": 1, "name": "Item 1", "tags": ["tag1", "tag2"]},
                    {"id": 2, "name": "Item 2", "tags": ["tag3"]},
                ],
                "metadata": {"count": 2, "categories": {"A": 1, "B": 1}},
            }

        result = serializable_return()
        assert "items" in result
        assert "metadata" in result

        @json_serializable
        def non_serializable_return() -> Dict[str, Any]:
            """Function that returns a non-serializable nested structure."""
            return {"items": [{"id": 1, "process": lambda x: x + 1}]}

        with pytest.raises(ToolValidationError) as exc:
            non_serializable_return()
        assert "contains non-JSON-serializable types" in str(exc.value)
