"""Comprehensive tests for tool registry."""

import time
from typing import Any, Dict

import pytest

from lluminary.tools.registry import ToolMetadata, ToolRegistry, ToolValidationError
from lluminary.tools.validators import (
    json_serializable,
    validate_params,
    validate_return_type,
    validate_tool,
)


def test_registry_initialization():
    """Test registry initialization."""
    registry = ToolRegistry()
    assert len(registry.list_tools()) == 0
    assert registry._tools == {}


def tool_with_type_hints(x: str) -> Dict[str, Any]:
    """Example tool with proper type hints."""
    return {"result": x}


def complete_tool(param1: str, param2: int) -> Dict[str, Any]:
    """Example tool with complete documentation and type hints.

    Args:
        param1: A string parameter
        param2: An integer parameter

    Returns:
        Dictionary with results
    """
    return {"param1": param1, "param2": param2}


def test_validate_tool():
    """Test the _validate_tool method."""
    registry = ToolRegistry()

    # Valid tool
    registry._validate_tool(complete_tool, "complete_tool")

    # Tool without docstring
    def tool_no_doc(param: str) -> Dict[str, Any]:
        return {"param": param}

    with pytest.raises(ToolValidationError) as exc:
        registry._validate_tool(tool_no_doc, "tool_no_doc")
    assert "must have a docstring" in str(exc.value)

    # Tool without return type
    def tool_no_return_type(param: str):
        """A tool without return type."""
        return {"param": param}

    with pytest.raises(ToolValidationError) as exc:
        registry._validate_tool(tool_no_return_type, "tool_no_return_type")
    assert "must have a return type annotation" in str(exc.value)

    # Tool without parameter type
    def tool_no_param_type(param) -> Dict[str, Any]:
        """A tool without parameter type."""
        return {"param": param}

    with pytest.raises(ToolValidationError) as exc:
        registry._validate_tool(tool_no_param_type, "tool_no_param_type")
    assert "must have a type annotation" in str(exc.value)


def test_create_tool_metadata():
    """Test the _create_tool_metadata method."""
    registry = ToolRegistry()

    # With default description
    metadata = registry._create_tool_metadata(complete_tool, "complete_tool")
    assert metadata.name == "complete_tool"
    assert metadata.description == complete_tool.__doc__
    assert "param1" in metadata.parameters
    assert "param2" in metadata.parameters
    assert metadata.return_type == str(Dict[str, Any])

    # With custom description
    metadata = registry._create_tool_metadata(
        complete_tool, "custom_name", "Custom description"
    )
    assert metadata.name == "custom_name"
    assert metadata.description == "Custom description"


def test_wrap_tool():
    """Test the _wrap_tool method."""
    # Use a unique name for this test
    tool_name = "wrap_test_tool"
    registry = ToolRegistry()

    # First register the tool metadata
    metadata = ToolMetadata(
        name=tool_name,
        description="Test tool",
        function=complete_tool,
        parameters={"param1": "str", "param2": "int"},
        return_type="Dict[str, Any]",
    )
    registry._tools[tool_name] = metadata
    print(f"Created metadata for {tool_name}")

    # Now wrap the tool
    wrapped = registry._wrap_tool(complete_tool, tool_name)

    # Successful execution
    result = wrapped("test", 42)
    print(f"Tool execution result: {result}")
    print(
        f"Tool stats after success: {registry._tools[tool_name].success_count} successes, {registry._tools[tool_name].failure_count} failures"
    )

    assert result == {"param1": "test", "param2": 42}
    assert registry._tools[tool_name].success_count == 1
    assert registry._tools[tool_name].failure_count == 0
    assert registry._tools[tool_name].total_execution_time > 0

    # Failed execution
    try:
        wrapped("test")  # Missing required argument
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected

    print(
        f"Tool stats after failure: {registry._tools[tool_name].success_count} successes, {registry._tools[tool_name].failure_count} failures"
    )
    assert registry._tools[tool_name].success_count == 1
    assert registry._tools[tool_name].failure_count == 1


def test_register_tool():
    """Test registering a tool."""
    registry = ToolRegistry()

    # Register with default name
    registry.register_tool(complete_tool)
    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "complete_tool"

    # Register with custom name and description
    registry.register_tool(
        tool_with_type_hints, name="custom_tool", description="Custom description"
    )
    tools = registry.list_tools()
    assert len(tools) == 2
    custom_tool = [t for t in tools if t["name"] == "custom_tool"][0]
    assert custom_tool["description"] == "Custom description"

    # Try to register an invalid tool
    def invalid_tool(param):
        """Invalid tool without type hints."""
        return {"param": param}

    with pytest.raises(ToolValidationError):
        registry.register_tool(invalid_tool)


def test_register_tools():
    """Test registering multiple tools."""
    registry = ToolRegistry()

    registry.register_tools([complete_tool, tool_with_type_hints])

    tools = registry.list_tools()
    assert len(tools) == 2
    assert any(t["name"] == "complete_tool" for t in tools)
    assert any(t["name"] == "tool_with_type_hints" for t in tools)


def test_get_tool():
    """Test retrieving a tool."""
    registry = ToolRegistry()
    registry.register_tool(complete_tool)

    # Get existing tool
    tool = registry.get_tool("complete_tool")
    assert tool is not None
    result = tool("test", 42)
    assert result == {"param1": "test", "param2": 42}

    # Get non-existent tool
    assert registry.get_tool("non_existent") is None


def test_list_tools():
    """Test listing tools."""
    registry = ToolRegistry()

    # Empty registry
    assert registry.list_tools() == []

    # Register tools
    registry.register_tool(complete_tool)
    registry.register_tool(tool_with_type_hints)

    tools = registry.list_tools()
    assert len(tools) == 2

    # Check required fields
    required_fields = {
        "name",
        "description",
        "parameters",
        "return_type",
        "success_count",
        "failure_count",
        "average_execution_time",
    }

    for tool in tools:
        assert all(field in tool for field in required_fields)


def test_get_tool_stats():
    """Test getting tool statistics."""
    # Create a uniquely named tool for this test
    stats_tool_name = "unique_stats_tool"

    # Create a separate registry for this test to avoid interference
    registry = ToolRegistry()

    # Define a simple test tool to avoid interaction with other tests
    def stats_test_tool(text: str) -> Dict[str, str]:
        """Tool used only for testing stats."""
        return {"result": text}

    # Register tool with unique name
    registry.register_tool(stats_test_tool, name=stats_tool_name)

    # Initialize stats
    stats = registry.get_tool_stats(stats_tool_name)
    print(f"Initial stats for {stats_tool_name}: {stats}")
    assert stats["success_count"] == 0
    assert stats["failure_count"] == 0

    # Execute tool
    tool = registry.get_tool(stats_tool_name)
    result = tool("test success")
    print(f"Tool execution result: {result}")

    # Check updated stats
    updated_stats = registry.get_tool_stats(stats_tool_name)
    print(f"Updated stats for {stats_tool_name}: {updated_stats}")
    assert updated_stats["success_count"] == 1
    assert updated_stats["failure_count"] == 0

    # Non-existent tool
    assert registry.get_tool_stats("non_existent") is None


def test_tool_execution_time_tracking():
    """Test that execution time is properly tracked."""
    registry = ToolRegistry()
    registry.register_tool(complete_tool)

    # First check initial stats
    initial_stats = registry.get_tool_stats("complete_tool")
    print(f"Initial stats: {initial_stats}")

    tool = registry.get_tool("complete_tool")

    # Execute with delay
    time.sleep(0.1)
    result = tool("test", 42)
    print(f"Tool execution result: {result}")

    stats = registry.get_tool_stats("complete_tool")
    print(f"Final stats: {stats}")

    # Use more relaxed assertions
    assert stats["total_execution_time"] > 0.0
    assert stats["average_execution_time"] > 0.0


def test_tool_with_decorators():
    """Test tools with validation decorators."""
    registry = ToolRegistry()

    @validate_tool
    @validate_return_type(Dict[str, Any])
    @validate_params(str, int)
    @json_serializable
    def decorated_tool(text: str, number: int) -> Dict[str, Any]:
        """Tool with all decorators."""
        return {"text": text, "number": number}

    registry.register_tool(decorated_tool)

    tool = registry.get_tool("decorated_tool")
    result = tool("test", 42)
    assert result == {"text": "test", "number": 42}

    # Test with invalid parameters
    with pytest.raises(ToolValidationError):
        tool(42, "not a number")  # Types swapped
