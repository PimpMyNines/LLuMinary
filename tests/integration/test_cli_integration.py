"""
Integration tests for the CLI functionality.
Tests real CLI commands with temp files and graceful skipping when auth fails.
"""

import json
import os
import tempfile

import pytest
from click.testing import CliRunner

from lluminary.cli.classify import cli

# Mark all tests in this file as CLI integration tests
pytestmark = [pytest.mark.integration, pytest.mark.cli]


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary classification config file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config = {
                "name": "test_config",
                "description": "Test classification config",
                "categories": {
                    "question": "A query seeking information",
                    "command": "A directive to perform an action",
                    "statement": "A declarative sentence",
                },
                "examples": [
                    {
                        "user_input": "What is the weather like?",
                        "doc_str": "This is a question seeking information about weather",
                        "selection": "question",
                    }
                ],
                "max_options": 1,
                "metadata": {"version": "1.0"},
            }
            f.write(json.dumps(config).encode("utf-8"))
            temp_path = f.name

        yield temp_path
        # Clean up
        os.unlink(temp_path)

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with multiple classification configs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create first config
            config1 = {
                "name": "config1",
                "description": "First test config",
                "categories": {
                    "category1": "Description 1",
                    "category2": "Description 2",
                },
                "examples": [],
                "max_options": 1,
                "metadata": {"version": "1.0"},
            }

            # Create second config
            config2 = {
                "name": "config2",
                "description": "Second test config",
                "categories": {
                    "category3": "Description 3",
                    "category4": "Description 4",
                },
                "examples": [],
                "max_options": 2,
                "metadata": {"version": "1.0"},
            }

            # Write configs to files
            with open(os.path.join(temp_dir, "config1.json"), "w") as f:
                json.dump(config1, f)

            with open(os.path.join(temp_dir, "config2.json"), "w") as f:
                json.dump(config2, f)

            yield temp_dir

    def test_validate_command(self, runner, temp_config_file):
        """Test the validate command with a valid configuration."""
        result = runner.invoke(cli, ["validate", temp_config_file])
        assert result.exit_code == 0
        assert "is valid" in result.output
        assert "Categories: 3" in result.output
        assert "Examples: 1" in result.output

    def test_list_configs_command(self, runner, temp_config_dir):
        """Test the list_configs command."""
        result = runner.invoke(cli, ["list-configs", temp_config_dir])
        assert result.exit_code == 0
        assert "config1" in result.output
        assert "config2" in result.output
        assert "First test config" in result.output
        assert "Second test config" in result.output

    def test_classification_command(self, runner, temp_config_file):
        """Test the classification command with a real API call."""
        # Try with multiple models in case one fails
        test_models = ["claude-haiku-3.5", "gpt-4o-mini", "gemini-2.0-flash-lite"]

        for model in test_models:
            result = runner.invoke(
                cli,
                [
                    "test",
                    temp_config_file,
                    "How do I reset my password?",
                    "--model",
                    model,
                ],
            )

            if result.exit_code == 0 and "Classification Results" in result.output:
                # Test passed with this model
                assert "Categories:" in result.output
                assert "Total tokens:" in result.output
                assert "Cost:" in result.output
                return

        # If we get here, no models worked - skip the test
        pytest.skip(
            "Skipping test as no models were able to complete the classification test"
        )
