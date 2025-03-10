from typing import Any, Dict, List, Optional, Tuple

from .config import ClassificationConfig
from .validators import validate_classification_response


class Classifier:
    """Handles classification operations across different LLM providers."""

    def __init__(self, llm):
        self.llm = llm

    def classify(
        self,
        messages: List[Dict[str, Any]],
        categories: Dict[str, str],
        examples: Optional[List[Dict[str, Any]]] = None,
        max_options: int = 1,
        system_prompt: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Classify messages into predefined categories.

        Args:
            messages: List of message dictionaries
            categories: Dictionary mapping category names to descriptions
            examples: Optional list of example classifications
            max_options: Maximum number of categories to select
            system_prompt: Optional custom system prompt

        Returns:
            Tuple of (selected categories, usage statistics)
        """
        config = ClassificationConfig(
            categories=categories, examples=examples or [], max_options=max_options
        )

        # Load default system prompt if none provided
        if not system_prompt:
            system_prompt = self._load_default_system_prompt()

        response, usage = self.llm.generate(
            event_id=f"classification_{id(messages)}",
            system_prompt=system_prompt,
            messages=messages,
            result_processing_function=lambda x: validate_classification_response(
                x, len(categories), max_options
            ),
        )

        # Convert numeric selections to category names
        selections = self._convert_to_category_names(response, list(categories.keys()))
        return selections, usage

    def _load_default_system_prompt(self) -> str:
        """Load the default classification system prompt."""
        import os

        import yaml

        # Get the path to the prompt template
        # Path is now at the root lluminary level
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        prompt_path = os.path.join(base_dir, "prompts", "classification", "base.yaml")

        try:
            with open(prompt_path) as f:
                data = yaml.safe_load(f)
                return data.get("system_prompt", "You are a classification system...")
        except (FileNotFoundError, yaml.YAMLError):
            # Fallback to hardcoded prompt if file can't be loaded
            return "You are a classification system that categorizes messages into predefined categories."

    def _convert_to_category_names(
        self, selections: List[int], category_names: List[str]
    ) -> List[str]:
        """Convert numeric selections to category names."""
        return [category_names[i - 1] for i in selections]
