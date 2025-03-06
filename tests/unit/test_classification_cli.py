"""
Unit tests for the classification CLI commands.
"""

import json
import os
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# Mark all tests in this file as classification tests
pytestmark = pytest.mark.classification

from lluminary.cli.classify import create, list_configs, test, validate
from lluminary.models.classification import ClassificationConfig


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return ClassificationConfig(
        name="test_config",
        description="Test configuration",
        categories={"cat1": "Category 1", "cat2": "Category 2"},
        examples=[
            {
                "user_input": "Test input",
                "doc_str": "Test reasoning",
                "selection": "cat1",
            }
        ],
        max_options=1,
        metadata={"version": "1.0"},
    )


@pytest.fixture
def temp_config_dir(test_config):
    """Create a temporary directory with test configuration files."""
    with TemporaryDirectory() as temp_dir:
        # Create a test config file
        config_path = os.path.join(temp_dir, "test_config.json")
        test_config.save(config_path)

        # Create an invalid file for testing
        with open(os.path.join(temp_dir, "not_a_config.txt"), "w") as f:
            f.write("This is not a JSON file")

        yield temp_dir


def test_list_configs_command(temp_config_dir):
    """Test the list_configs command."""
    runner = CliRunner()
    result = runner.invoke(list_configs, [temp_config_dir])

    # Verify output
    assert result.exit_code == 0
    assert "test_config" in result.output
    assert "Test configuration" in result.output
    assert "cat1, cat2" in result.output


def test_list_configs_empty_dir():
    """Test list_configs with an empty directory."""
    with TemporaryDirectory() as empty_dir:
        runner = CliRunner()
        result = runner.invoke(list_configs, [empty_dir])

        assert result.exit_code == 0
        assert "No classification configs found" in result.output


def test_validate_command(temp_config_dir):
    """Test the validate command with valid config."""
    config_path = os.path.join(temp_config_dir, "test_config.json")

    runner = CliRunner()
    result = runner.invoke(validate, [config_path])

    # Verify output
    assert result.exit_code == 0
    assert "✓ Configuration test_config is valid" in result.output
    assert "Categories: 2" in result.output
    assert "Examples: 1" in result.output


def test_validate_command_invalid():
    """Test the validate command with invalid config."""
    with TemporaryDirectory() as temp_dir:
        # Create an invalid config file
        invalid_path = os.path.join(temp_dir, "invalid.json")
        with open(invalid_path, "w") as f:
            f.write('{"name": "invalid", "categories": {}}')  # Empty categories

        runner = CliRunner()
        result = runner.invoke(validate, [invalid_path])

        # Verify error
        assert result.exit_code == 1
        assert "✗ Configuration is invalid" in result.output


@patch("src.lluminary.cli.classify.get_llm_from_model")
def test_test_command(mock_get_llm, temp_config_dir):
    """Test the test command."""
    # Setup mock LLM
    mock_llm = MagicMock()
    mock_llm.classify_from_file.return_value = (
        ["cat1"],
        {
            "read_tokens": 100,
            "write_tokens": 20,
            "total_tokens": 120,
            "total_cost": 0.01,
        },
    )
    mock_get_llm.return_value = mock_llm

    # Run test command
    config_path = os.path.join(temp_config_dir, "test_config.json")
    runner = CliRunner()
    result = runner.invoke(
        test, [config_path, "Test message", "--model", "claude-haiku-3.5"]
    )

    # Verify output
    assert result.exit_code == 0
    assert "Classification Results" in result.output
    assert "cat1" in result.output
    assert "Total tokens: 120" in result.output
    assert "Cost: $0.010000" in result.output

    # Verify LLM was called correctly
    mock_get_llm.assert_called_once_with("claude-haiku-3.5")
    mock_llm.classify_from_file.assert_called_once()
    args = mock_llm.classify_from_file.call_args[0]
    assert args[0] == config_path
    assert len(args[1]) == 1
    assert args[1][0]["message"] == "Test message"


@patch("src.lluminary.cli.classify.get_llm_from_model")
def test_test_command_error(mock_get_llm):
    """Test the test command error handling."""
    # Setup mock LLM with error
    mock_llm = MagicMock()
    mock_llm.classify_from_file.side_effect = Exception("Test error")
    mock_get_llm.return_value = mock_llm

    # Run test command
    runner = CliRunner()
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "tmp.json")
        with open(path, "w") as f:
            f.write("{}")  # Empty file

        result = runner.invoke(test, [path, "Test message"])

        # Verify error output
        assert result.exit_code == 1
        assert "Error" in result.output


@patch("click.prompt")
@patch("click.confirm")
def test_create_command(mock_confirm, mock_prompt):
    """Test the create command."""
    # Setup mock confirm for example question
    mock_confirm.return_value = True

    # Setup mock prompts
    mock_prompt.side_effect = [
        # Category 1
        "cat1",
        "Category 1",
        # Category 2
        "cat2",
        "Category 2",
        # Empty category name to finish
        "",
        # Example 1
        "Example input",
        "Example reasoning",
        "cat1",
        # Empty input to finish examples
        "",
        # Max options
        1,
        # Metadata
        "Test User",
        "2024-05-03",
        "test,classification",
    ]

    # Run create command
    with TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "new_config.json")

        runner = CliRunner()
        result = runner.invoke(
            create, [output_path, "--name", "new_config", "--description", "New config"]
        )

        # Verify output
        assert result.exit_code == 0
        assert "Configuration saved" in result.output

        # Verify created file
        assert os.path.exists(output_path)
        with open(output_path) as f:
            config_data = json.load(f)
            assert config_data["name"] == "new_config"
            assert config_data["categories"] == {
                "cat1": "Category 1",
                "cat2": "Category 2",
            }
            assert len(config_data["examples"]) == 1
            assert config_data["examples"][0]["selection"] == "cat1"
            assert config_data["max_options"] == 1
            assert "metadata" in config_data
            assert "tags" in config_data["metadata"]
            assert "test" in config_data["metadata"]["tags"]
