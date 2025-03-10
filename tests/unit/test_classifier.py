"""
Unit tests for the Classifier class.
"""

import pytest

# Mark all tests in this file as classification tests
pytestmark = pytest.mark.classification
from unittest.mock import MagicMock, patch

from lluminary.models.classification.classifier import Classifier
from lluminary.models.classification.validators import ValidationError


class TestClassifier:
    """Test suite for the Classifier class."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create a mock LLM
        self.mock_llm = MagicMock()
        # The result should be a tuple of (result, usage_stats)
        # where result will be the parsed list of selected indices [1]
        self.mock_llm.generate.return_value = ([1], {"total_cost": 0.01})

        # Create classifier with mock LLM
        self.classifier = Classifier(self.mock_llm)

        # Common test data
        self.test_messages = [{"message_type": "human", "message": "Test message"}]
        self.test_categories = {"cat1": "Category 1", "cat2": "Category 2"}

    def test_classify_basic(self):
        """Test basic classification functionality."""
        # Call classify
        selections, usage = self.classifier.classify(
            messages=self.test_messages, categories=self.test_categories
        )

        # Verify selections and usage stats
        assert selections == ["cat1"]
        assert usage == {"total_cost": 0.01}

        # Verify generate was called with correct parameters
        self.mock_llm.generate.assert_called_once()
        call_args = self.mock_llm.generate.call_args[1]
        assert call_args["messages"] == self.test_messages
        assert isinstance(call_args["result_processing_function"], type(lambda: None))

    def test_classify_with_examples(self):
        """Test classification with examples."""
        # Test data
        examples = [
            {
                "user_input": "Example input",
                "doc_str": "Example reasoning",
                "selection": "cat1",
            }
        ]

        # Call classify with examples
        selections, usage = self.classifier.classify(
            messages=self.test_messages,
            categories=self.test_categories,
            examples=examples,
        )

        # Verify selections
        assert selections == ["cat1"]

        # Verify examples were passed correctly
        call_args = self.mock_llm.generate.call_args[1]
        assert (
            "examples" not in call_args
        )  # Should be part of config, not directly passed

    def test_classify_multiple_categories(self):
        """Test classification with multiple category selection."""
        # Configure mock to return multiple selections
        self.mock_llm.generate.return_value = ([1, 2], {"total_cost": 0.01})

        # Call classify with max_options=2
        selections, usage = self.classifier.classify(
            messages=self.test_messages, categories=self.test_categories, max_options=2
        )

        # Verify both categories were selected
        assert len(selections) == 2
        assert "cat1" in selections
        assert "cat2" in selections

    def test_classify_with_custom_system_prompt(self):
        """Test classification with a custom system prompt."""
        # Custom prompt
        custom_prompt = "You are a specialized classifier..."

        # Call classify with custom prompt
        self.classifier.classify(
            messages=self.test_messages,
            categories=self.test_categories,
            system_prompt=custom_prompt,
        )

        # Verify custom prompt was used
        call_args = self.mock_llm.generate.call_args[1]
        assert call_args["system_prompt"] == custom_prompt

    def test_classify_error_handling(self):
        """Test error handling in classification."""
        # Configure mock to return invalid response
        self.mock_llm.generate.side_effect = ValidationError("Invalid response")

        # Call classify should propagate the error
        with pytest.raises(ValidationError):
            self.classifier.classify(
                messages=self.test_messages, categories=self.test_categories
            )

    def test_load_default_system_prompt(self):
        """Test loading of default system prompt."""
        # Test the prompt loading (currently returns a placeholder)
        prompt = self.classifier._load_default_system_prompt()
        assert isinstance(prompt, str)
        assert "classification" in prompt.lower()

    def test_convert_to_category_names(self):
        """Test conversion from numeric indices to category names."""
        # Test data
        indices = [1, 2]
        category_names = ["cat1", "cat2", "cat3"]

        # Convert indices to names
        result = self.classifier._convert_to_category_names(indices, category_names)

        # Verify conversion
        assert result == ["cat1", "cat2"]

        # Test boundary case
        result = self.classifier._convert_to_category_names([3], category_names)
        assert result == ["cat3"]

    @patch(
        "src.lluminary.models.classification.classifier.validate_classification_response"
    )
    def test_result_processing(self, mock_validate):
        """Test the result processing function passed to generate."""
        # Setup mock validator
        mock_validate.return_value = [1]

        # Call classify to trigger result processing
        self.classifier.classify(
            messages=self.test_messages, categories=self.test_categories
        )

        # Extract the result processing function
        generate_call = self.mock_llm.generate.call_args
        result_processor = generate_call[1]["result_processing_function"]

        # Call the result processor directly
        processed = result_processor("test response")

        # Verify validator was called with correct parameters
        mock_validate.assert_called_once_with("test response", 2, 1)
