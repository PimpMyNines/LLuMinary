"""Unit tests for classification CLI commands."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from lluminary.cli.classify import create, list_configs, test, validate


@pytest.fixture
def mock_classification_library():
    """Fixture for mocking ClassificationLibrary."""
    with patch("lluminary.cli.classify.ClassificationLibrary") as mock:
        instance = mock.return_value
        instance.list_configs.return_value = [
            {
                "name": "test_config",
                "description": "Test configuration",
                "categories": ["category1", "category2"],
                "metadata": {"author": "tester"},
            }
        ]
        yield mock


@pytest.fixture
def mock_classification_config():
    """Fixture for mocking ClassificationConfig."""
    with patch("lluminary.cli.classify.ClassificationConfig") as mock:
        instance = mock.return_value
        instance.name = "test_config"
        instance.categories = {
            "category1": "First category",
            "category2": "Second category",
        }
        instance.examples = [
            {"user_input": "test", "doc_str": "test doc", "selection": "category1"}
        ]
        instance.max_options = 1

        # Add method mocks
        instance.validate.return_value = None
        instance.save.return_value = None

        # Setup from_file mock
        mock.from_file.return_value = instance

        yield mock


@pytest.fixture
def mock_llm():
    """Fixture for mocking LLM."""
    with patch("lluminary.cli.classify.get_llm_from_model") as mock:
        instance = mock.return_value
        instance.classify_from_file.return_value = (
            ["category1"],
            {"total_tokens": 150, "total_cost": 0.000123},
        )
        yield mock


def test_list_configs_command(mock_classification_library):
    """Test the list_configs command."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a test config directory
        Path("test_configs").mkdir()

        result = runner.invoke(list_configs, ["test_configs"])

        assert result.exit_code == 0
        assert "test_config" in result.output
        assert "Test configuration" in result.output
        assert "category1, category2" in result.output
        assert "author" in result.output

        mock_classification_library.assert_called_once_with("test_configs")
        mock_classification_library.return_value.load_configs.assert_called_once()
        mock_classification_library.return_value.list_configs.assert_called_once()


def test_list_configs_empty(mock_classification_library):
    """Test the list_configs command with no configs found."""
    mock_classification_library.return_value.list_configs.return_value = []

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("empty_dir").mkdir()

        result = runner.invoke(list_configs, ["empty_dir"])

        assert result.exit_code == 0
        assert "No classification configs found" in result.output


def test_validate_command(mock_classification_config):
    """Test the validate command."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a test config file
        Path("test_config.json").write_text('{"name": "test_config"}')

        result = runner.invoke(validate, ["test_config.json"])

        assert result.exit_code == 0
        assert "✓ Configuration test_config is valid" in result.output
        assert "Categories: 2" in result.output
        assert "Examples: 1" in result.output

        mock_classification_config.from_file.assert_called_once_with("test_config.json")
        mock_classification_config.return_value.validate.assert_called_once()


def test_validate_command_error(mock_classification_config):
    """Test the validate command with validation error."""
    mock_classification_config.return_value.validate.side_effect = ValueError(
        "Invalid config"
    )

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("invalid_config.json").write_text('{"name": "invalid"}')

        result = runner.invoke(validate, ["invalid_config.json"])

        assert result.exit_code == 1
        assert "✗ Configuration is invalid: Invalid config" in result.output


def test_test_command(mock_classification_config, mock_llm):
    """Test the test command."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("test_config.json").write_text('{"name": "test_config"}')

        result = runner.invoke(
            test,
            ["test_config.json", "This is a test message", "--model", "test-model"],
        )

        assert result.exit_code == 0
        assert "Classification Results:" in result.output
        assert "Categories: category1" in result.output
        assert "Total tokens: 150" in result.output
        assert "$0.000123" in result.output

        mock_llm.assert_called_once_with("test-model")
        mock_llm.return_value.classify_from_file.assert_called_once_with(
            "test_config.json",
            [
                {
                    "message_type": "human",
                    "message": "This is a test message",
                    "image_paths": [],
                    "image_urls": [],
                }
            ],
            system_prompt=None,
        )


def test_test_command_with_system_prompt(mock_classification_config, mock_llm):
    """Test the test command with system prompt."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("test_config.json").write_text('{"name": "test_config"}')

        result = runner.invoke(
            test,
            [
                "test_config.json",
                "This is a test message",
                "--system-prompt",
                "Custom system prompt",
            ],
        )

        assert result.exit_code == 0
        mock_llm.return_value.classify_from_file.assert_called_once_with(
            "test_config.json",
            [
                {
                    "message_type": "human",
                    "message": "This is a test message",
                    "image_paths": [],
                    "image_urls": [],
                }
            ],
            system_prompt="Custom system prompt",
        )


def test_test_command_error(mock_classification_config, mock_llm):
    """Test the test command with error."""
    mock_llm.return_value.classify_from_file.side_effect = Exception("Test error")

    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("test_config.json").write_text('{"name": "test_config"}')

        result = runner.invoke(test, ["test_config.json", "This is a test message"])

        assert result.exit_code == 1
        assert "Error: Test error" in result.output


def test_create_command(mock_classification_config):
    """Test the create command."""
    runner = CliRunner()

    # Simulate interactive input
    input_values = [
        "Test Classifier",  # name
        "Test description",  # description
        "category1",  # first category name
        "Category 1 desc",  # first category description
        "category2",  # second category name
        "Category 2 desc",  # second category description
        "",  # empty category name to finish
        "y",  # add examples?
        "Example input",  # example input
        "Example reasoning",  # example reasoning
        "category1",  # category selection
        "",  # empty input to finish examples
        "2",  # max options
        "Test Author",  # author
        "2024-03-05",  # date
    ]

    with runner.isolated_filesystem():
        result = runner.invoke(
            create, ["output_config.json"], input="\n".join(input_values)
        )

        assert result.exit_code == 0
        assert "Configuration saved to output_config.json" in result.output

        # Check that ClassificationConfig was created correctly
        mock_classification_config.assert_called_once()
        instance = mock_classification_config.return_value
        assert instance.validate.called
        assert instance.save.called


def test_create_command_error(mock_classification_config):
    """Test the create command with validation error."""
    mock_classification_config.return_value.validate.side_effect = ValueError(
        "Invalid config"
    )

    runner = CliRunner()
    input_values = [
        "Invalid",  # name
        "Test",  # description
        "cat1",  # category name
        "Cat 1 desc",  # category description
        "",  # empty to finish
        "n",  # no examples
        "1",  # max options
        "Test",  # author
        "2024-03-05",  # date
    ]

    with runner.isolated_filesystem():
        result = runner.invoke(
            create, ["output_config.json"], input="\n".join(input_values)
        )

        assert result.exit_code == 1
        assert "Error: Invalid config" in result.output
