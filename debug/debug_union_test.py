"""Debug script for union type validation."""

from typing import Union

from lluminary.tools.validators import _validate_type_structure


def main():
    """Run debug test for union types."""
    union_type = Union[int, str]

    # Test with int (should be True)
    value1 = 10
    result1 = _validate_type_structure(value1, union_type)
    print(f"Value {value1} (type {type(value1)}) with Union[int, str]: {result1}")

    # Test with str (should be True)
    value2 = "test"
    result2 = _validate_type_structure(value2, union_type)
    print(f"Value {value2} (type {type(value2)}) with Union[int, str]: {result2}")

    # Test with bool (should be False)
    value3 = True
    result3 = _validate_type_structure(value3, union_type)
    print(f"Value {value3} (type {type(value3)}) with Union[int, str]: {result3}")

    # In Python, bool is a subclass of int, which might be the issue
    print(f"isinstance(True, int): {isinstance(True, int)}")


if __name__ == "__main__":
    main()
