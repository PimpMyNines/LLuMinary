"""Minimal test for tools module."""


def test_tool_imports():
    """Test that we can import from the tools module."""
    from lluminary.tools import ToolRegistry

    registry = ToolRegistry()
    assert isinstance(registry, ToolRegistry)
