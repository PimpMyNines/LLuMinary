"""Configuration handling for GitHub Project Management Tool."""

import os
import yaml
from typing import Any, Dict, Optional, Union, List

class Config:
    """Configuration manager.

    Handles loading and accessing configuration from multiple sources:
    - Config files
    - Environment variables
    - Command line arguments
    """

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_file: Path to config file (optional)
        """
        self.config_data = {}
        self._load_config(config_file)

    def _load_config(self, config_file: Optional[str] = None) -> None:
        """Load configuration from file.

        Args:
            config_file: Path to config file
        """
        # Default config locations
        potential_config_files = [
            config_file,
            os.environ.get('GITHUB_PM_CONFIG'),
            os.path.join(os.getcwd(), 'config.yaml'),
            os.path.expanduser('~/.github-pm/config.yaml'),
        ]
        
        # Try each location
        for file_path in potential_config_files:
            if file_path and os.path.isfile(file_path):
                try:
                    with open(file_path, 'r') as f:
                        self.config_data = yaml.safe_load(f) or {}
                    break
                except Exception as e:
                    print(f"Warning: Failed to load config file {file_path}: {str(e)}")
        
        # If no config found, initialize with empty dict
        if not self.config_data:
            self.config_data = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key (dot notation)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        # Try environment variable first (convert dots to underscores)
        env_key = f"GITHUB_PM_{key.replace('.', '_').upper()}"
        if env_key in os.environ:
            return os.environ[env_key]
            
        # If not in environment, look in config data
        parts = key.split('.')
        current = self.config_data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current

    def has(self, key: str) -> bool:
        """Check if configuration key exists.

        Args:
            key: Configuration key (dot notation)

        Returns:
            True if key exists, False otherwise
        """
        # Check env var first
        env_key = f"GITHUB_PM_{key.replace('.', '_').upper()}"
        if env_key in os.environ:
            return True
            
        # Check config data
        parts = key.split('.')
        current = self.config_data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
                
        return True

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key (dot notation)
            value: Value to set
        """
        parts = key.split('.')
        current = self.config_data
        
        # Navigate to the right level
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
            
        # Set the value
        current[parts[-1]] = value
