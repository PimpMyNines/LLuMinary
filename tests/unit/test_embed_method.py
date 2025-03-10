"""
Simple unit tests for embedding functionality.
"""

import unittest
from unittest.mock import MagicMock, patch


# To avoid import errors during testing
class MockOpenAI:
    def __init__(self, *args, **kwargs):
        self.embeddings = MagicMock()


class TestEmbedFunctionality(unittest.TestCase):
    def setUp(self):
        # Apply the patch for the import
        self.patcher = patch("openai.OpenAI", MockOpenAI)
        self.patcher.start()

        # Now we can import the class
        from lluminary.models.providers.openai import OpenAILLM

        self.OpenAILLM = OpenAILLM

    def tearDown(self):
        self.patcher.stop()

    def test_embed_method(self):
        # Create an OpenAILLM instance
        llm = self.OpenAILLM("text-embedding-3-small", api_key="test-key")

        # Define texts for embedding
        texts = ["This is a test."]

        # Mock the embedding models and the default model
        llm.EMBEDDING_MODELS = ["text-embedding-3-small"]
        llm.DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

        # Create mock embedding response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage = MagicMock(total_tokens=10)

        # Set up mock client's response
        llm.client = MagicMock()
        llm.client.embeddings.create.return_value = mock_response

        # Set up embedding costs
        llm.embedding_costs = {"text-embedding-3-small": 0.0001}

        # Call the embed method
        embeddings, usage = llm.embed(texts)

        # Verify the results
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])

        # Verify the usage information
        self.assertEqual(usage.get("tokens"), 10)
        self.assertEqual(usage.get("model"), "text-embedding-3-small")
        self.assertIn("cost", usage)

        # Verify the API was called with expected arguments
        llm.client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=texts, encoding_format="float"
        )


if __name__ == "__main__":
    unittest.main()
