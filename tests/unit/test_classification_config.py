"""
Unit tests for the classification configuration classes.
"""

import json
import os
from tempfile import TemporaryDirectory

import pytest

# Mark all tests in this file as classification tests
pytestmark = pytest.mark.classification

from lluminary.models.classification.config import (
    ClassificationConfig,
    ClassificationLibrary,
)


def test_classification_config_initialization():
    """Test that ClassificationConfig initializes correctly."""
    # Basic initialization
    categories = {"cat1": "Category 1", "cat2": "Category 2"}
    examples = [{"user_input": "test", "doc_str": "test doc", "selection": "cat1"}]

    config = ClassificationConfig(
        categories=categories,
        examples=examples,
        max_options=1,
        name="test_config",
        description="Test config",
        metadata={"version": "1.0"},
    )

    # Check attributes
    assert config.categories == categories
    assert config.examples == examples
    assert config.max_options == 1
    assert config.name == "test_config"
    assert config.description == "Test config"
    assert config.metadata == {"version": "1.0"}


def test_classification_config_to_dict():
    """Test to_dict method with different options."""
    config = ClassificationConfig(
        categories={"cat1": "Category 1", "cat2": "Category 2"},
        examples=[],
        name="test_config",
    )

    # Test normal dictionary output
    data = config.to_dict()
    assert data["name"] == "test_config"
    assert data["categories"] == {"cat1": "Category 1", "cat2": "Category 2"}

    # Test list format for categories
    list_data = config.to_dict(for_list=True)
    assert list_data["categories"] == ["cat1", "cat2"]


def test_classification_config_validation():
    """Test validation rules for ClassificationConfig."""
    # Valid config
    valid_config = ClassificationConfig(
        categories={"cat1": "Category 1", "cat2": "Category 2"},
        examples=[],
        max_options=1,
    )
    valid_config.validate()  # Should not raise

    # Empty categories
    with pytest.raises(ValueError, match="Categories cannot be empty"):
        config = ClassificationConfig(categories={}, examples=[])
        config.validate()

    # Invalid max_options (too small)
    with pytest.raises(ValueError, match="max_options must be >= 1"):
        config = ClassificationConfig(
            categories={"cat1": "Category 1"}, examples=[], max_options=0
        )
        config.validate()

    # Invalid max_options (too large)
    with pytest.raises(
        ValueError, match="max_options cannot exceed number of categories"
    ):
        config = ClassificationConfig(
            categories={"cat1": "Category 1"}, examples=[], max_options=2
        )
        config.validate()

    # Invalid example format
    with pytest.raises(ValueError, match="Invalid example format"):
        config = ClassificationConfig(
            categories={"cat1": "Category 1"},
            examples=[{"incomplete": "example"}],
            max_options=1,
        )
        config.validate()

    # Invalid category in example
    with pytest.raises(ValueError, match="Invalid category in example"):
        config = ClassificationConfig(
            categories={"cat1": "Category 1"},
            examples=[
                {
                    "user_input": "test",
                    "doc_str": "test doc",
                    "selection": "non_existent_category",
                }
            ],
            max_options=1,
        )
        config.validate()


def test_classification_config_file_operations():
    """Test saving and loading ClassificationConfig from files."""
    config = ClassificationConfig(
        categories={"cat1": "Category 1", "cat2": "Category 2"},
        examples=[{"user_input": "test", "doc_str": "test doc", "selection": "cat1"}],
        max_options=1,
        name="test_config",
        description="Test config",
        metadata={"version": "1.0"},
    )

    with TemporaryDirectory() as temp_dir:
        # Test save method
        file_path = os.path.join(temp_dir, "test_config.json")
        config.save(file_path)
        assert os.path.exists(file_path)

        # Verify file contents
        with open(file_path) as f:
            data = json.load(f)
            assert data["name"] == "test_config"
            assert "cat1" in data["categories"]

        # Test loading from file
        loaded_config = ClassificationConfig.from_file(file_path)
        assert loaded_config.name == config.name
        assert loaded_config.categories == config.categories
        assert loaded_config.examples == config.examples


def test_classification_library():
    """Test ClassificationLibrary functionality."""
    with TemporaryDirectory() as temp_dir:
        # Create test configs
        config1 = ClassificationConfig(
            categories={"cat1": "Category 1"},
            examples=[],
            name="config1",
            description="First config",
        )

        config2 = ClassificationConfig(
            categories={"cat2": "Category 2"},
            examples=[],
            name="config2",
            description="Second config",
        )

        # Save configs to files
        config1.save(os.path.join(temp_dir, "config1.json"))
        config2.save(os.path.join(temp_dir, "config2.json"))

        # Add invalid file
        with open(os.path.join(temp_dir, "not_a_config.txt"), "w") as f:
            f.write("This is not a JSON file")

        # Initialize library
        library = ClassificationLibrary(temp_dir)
        library.load_configs()

        # Test list_configs
        configs = library.list_configs()
        assert len(configs) == 2
        config_names = [c["name"] for c in configs]
        assert "config1" in config_names
        assert "config2" in config_names

        # Test get_config
        retrieved = library.get_config("config1")
        assert retrieved.name == "config1"
        assert retrieved.categories == {"cat1": "Category 1"}

        # Test get_config with invalid name
        with pytest.raises(KeyError):
            library.get_config("nonexistent_config")

        # Test add_config
        config3 = ClassificationConfig(
            categories={"cat3": "Category 3"}, examples=[], name="config3"
        )
        library.add_config(config3)
        assert "config3" in [c["name"] for c in library.list_configs()]
        assert os.path.exists(os.path.join(temp_dir, "config3.json"))

        # Test remove_config
        library.remove_config("config1")
        assert "config1" not in [c["name"] for c in library.list_configs()]
        assert not os.path.exists(os.path.join(temp_dir, "config1.json"))
