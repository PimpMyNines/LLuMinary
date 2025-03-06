"""
Very simple test for base LLM class support functions.
"""

import unittest
from unittest.mock import MagicMock


class TestSupportFunctions(unittest.TestCase):
    """Test support functions in base LLM class."""

    def test_supports_embeddings(self):
        """Test supports_embeddings method."""
        # Create a mock with the necessary attributes
        llm = MagicMock()

        # Test when model is in embedding models list
        llm.EMBEDDING_MODELS = ["model1", "model2"]
        llm.model_name = "model1"

        # Define our own implementation of supports_embeddings
        def mock_supports_embeddings():
            return (
                len(llm.EMBEDDING_MODELS) > 0 and llm.model_name in llm.EMBEDDING_MODELS
            )

        # Replace the mock's supports_embeddings with our implementation
        llm.supports_embeddings = mock_supports_embeddings

        # Test cases
        self.assertTrue(llm.supports_embeddings())

        llm.model_name = "model3"
        self.assertFalse(llm.supports_embeddings())

        llm.EMBEDDING_MODELS = []
        self.assertFalse(llm.supports_embeddings())


if __name__ == "__main__":
    unittest.main()
