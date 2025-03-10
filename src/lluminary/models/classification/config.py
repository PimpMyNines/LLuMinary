"""Configuration management for classification."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ClassificationConfig:
    """Configuration for classification operations."""

    categories: Dict[str, str]
    examples: List[Dict[str, str]]
    max_options: int = 1
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict] = None

    @classmethod
    def from_file(cls, file_path: str) -> "ClassificationConfig":
        """Load configuration from a JSON file."""
        with open(file_path) as f:
            data = json.load(f)

        return cls(
            categories=data["categories"],
            examples=data.get("examples", []),
            max_options=data.get("max_options", 1),
            name=data.get("name"),
            description=data.get("description"),
            metadata=data.get("metadata"),
        )

    def to_dict(self, for_list: bool = False) -> Dict[str, Any]:
        """Convert config to dictionary format.

        Args:
            for_list: If True, returns categories as a list of keys
        """
        data = {
            "name": self.name,
            "description": self.description,
            "categories": list(self.categories.keys()) if for_list else self.categories,
            "examples": self.examples,
            "max_options": self.max_options,
            "metadata": self.metadata or {},
        }
        return data

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.categories:
            raise ValueError("Categories cannot be empty")

        if self.max_options < 1:
            raise ValueError("max_options must be >= 1")

        if self.max_options > len(self.categories):
            raise ValueError("max_options cannot exceed number of categories")

        for example in self.examples:
            if not all(k in example for k in ["user_input", "doc_str", "selection"]):
                raise ValueError("Invalid example format")
            if example["selection"] not in self.categories:
                raise ValueError(f"Invalid category in example: {example['selection']}")

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self.to_dict(for_list=False), f, indent=4)

    def to_file(self, file_path: str) -> None:
        """Save configuration to a JSON file."""
        self.save(file_path)


class ClassificationLibrary:
    """Manages a collection of classification configurations."""

    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.configs: Dict[str, ClassificationConfig] = {}

    def load_configs(self) -> None:
        """Load all configuration files from the config directory."""
        if not os.path.exists(self.config_dir):
            return

        for file_name in os.listdir(self.config_dir):
            if file_name.endswith(".json"):
                path = os.path.join(self.config_dir, file_name)
                config = ClassificationConfig.from_file(path)
                if config.name:
                    self.configs[config.name] = config

    def get_config(self, name: str) -> ClassificationConfig:
        """Get a configuration by name.

        Args:
            name: Name of the configuration to retrieve

        Returns:
            The requested configuration

        Raises:
            KeyError: If the configuration is not found
        """
        if name not in self.configs:
            raise KeyError(f"Classification config not found: {name}")
        return self.configs[name]

    def add_config(self, config: ClassificationConfig) -> None:
        """Add a new configuration to the library."""
        if not config.name:
            raise ValueError("Configuration must have a name")
        self.configs[config.name] = config

        # Save to file
        file_path = os.path.join(self.config_dir, f"{config.name}.json")
        config.save(file_path)

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all available configurations with their details."""
        return [config.to_dict(for_list=True) for config in self.configs.values()]

    def remove_config(self, name: str) -> None:
        """Remove a configuration."""
        if name not in self.configs:
            raise KeyError(f"Classification config not found: {name}")

        config_path = Path(self.config_dir) / f"{name}.json"
        if config_path.exists():
            config_path.unlink()

        del self.configs[name]
