"""Debug script for type validator."""

from typing import Any, Dict, List, Optional, Union, get_args, get_origin


def validate_type_structure(value, expected_type):
    """Validate value against expected type, handling complex nested types."""
    print(f"Validating value: {value!r} against type: {expected_type!r}")

    # Special case for Any
    if expected_type is Any:
        print("Type is Any, returning True")
        return True

    # Handle None with Optional types
    if value is None:
        # Check if expected_type is Optional[...] (Union[..., None])
        origin = get_origin(expected_type)
        if origin is Union:
            args = get_args(expected_type)
            result = type(None) in args
            print(f"Value is None, checking Union type, result: {result}")
            return result
        print("Value is None but type is not Optional, returning False")
        return False

    # Get the origin and args for generic types (like List, Dict, etc.)
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    print(f"Type origin: {origin}, args: {args}")

    # Handle basic types
    if origin is None:
        result = isinstance(value, expected_type)
        print(f"Basic type check, result: {result}")
        return result

    # Handle Union types
    if origin is Union:
        results = [validate_type_structure(value, arg) for arg in args]
        result = any(results)
        print(f"Union type check, results for each arg: {results}, final: {result}")
        return result

    # Handle List types
    if origin is list or origin is List:
        if not isinstance(value, list):
            print("Value is not a list, returning False")
            return False
        if not args or args[0] is Any:
            print("List with Any type or no args, returning True")
            return True

        results = [validate_type_structure(item, args[0]) for item in value]
        result = all(results)
        print(f"List item checks, results: {results}, final: {result}")
        return result

    # Handle Dict types
    if origin is dict or origin is Dict:
        if not isinstance(value, dict):
            print("Value is not a dict, returning False")
            return False

        if not args:
            print("Dict with no args, returning True")
            return True

        key_type, val_type = args

        print(f"Dict key_type: {key_type}, val_type: {val_type}")

        if key_type is Any and val_type is Any:
            print("Dict with Any for both key and value, returning True")
            return True

        key_results = []
        if key_type is not Any:
            key_results = [validate_type_structure(k, key_type) for k in value.keys()]
            if not all(key_results):
                print(f"Dict key checks failed: {key_results}")
                return False

        val_results = []
        if val_type is not Any:
            val_results = [validate_type_structure(v, val_type) for v in value.values()]
            if not all(val_results):
                print(f"Dict value checks failed: {val_results}")
                return False

        print(f"Dict checks: keys {key_results}, values {val_results}, returning True")
        return True

    # Fall back to simple isinstance check for other cases
    result = isinstance(value, origin)
    print(f"Fallback check, result: {result}")
    return result


def test_nested_types():
    """Test nested types validation."""

    print("\nTest 1: Simple nested dict with list of dicts")
    struct1 = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
    type1 = Dict[str, List[Dict[str, Any]]]
    assert validate_type_structure(struct1, type1) is True

    print("\nTest 2: String value where int expected")
    struct2 = {"users": [{"name": "Alice", "age": "30"}]}
    type2 = Dict[str, List[Dict[str, Union[str, int]]]]
    assert validate_type_structure(struct2, type2) is True

    print("\nTest 3: None value where Optional expected")
    struct3 = {"users": [{"name": "Alice", "age": None}]}
    type3 = Dict[str, List[Dict[str, Optional[int]]]]
    assert validate_type_structure(struct3, type3) is True

    print("\nTest 4: String where list expected")
    struct4 = {"users": "not a list"}
    type4 = Dict[str, List[Any]]
    assert validate_type_structure(struct4, type4) is False


if __name__ == "__main__":
    test_nested_types()
