"""
Tests for the streaming functionality.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


from lluminary import get_llm_from_model
from lluminary.models.providers.google import GoogleLLM


class TestStreaming(unittest.TestCase):
    """Tests for streaming functionality across providers."""

    def setUp(self):
        """Set up test environment."""
        # Mock API keys for testing
        self.openai_key = "test-openai-key"
        self.anthropic_key = "test-anthropic-key"
        self.google_key = "test-google-key"

        # Test messages
        self.messages = [
            {"message_type": "human", "message": "What is machine learning?"}
        ]

        # Test system prompt
        self.system_prompt = "You are a helpful assistant."

    @patch("openai.OpenAI")
    def test_openai_streaming(self, mock_openai):
        """Test OpenAI streaming functionality with mocked client."""
        # Setup mock client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Create mock chunks for the stream
        mock_chunks = []
        for text in [
            "Machine ",
            "learning ",
            "is ",
            "a ",
            "subset ",
            "of ",
            "artificial ",
            "intelligence.",
        ]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta = MagicMock()
            chunk.choices[0].delta.content = text
            mock_chunks.append(chunk)

        # Set the mock client to return our mock chunks
        mock_client.chat.completions.create.return_value = mock_chunks

        # Create OpenAI LLM
        llm = get_llm_from_model(
            "gpt-4o-mini", provider="openai", api_key=self.openai_key
        )

        # Monkeypatch the client
        llm.client = mock_client

        # Test callback
        received_chunks = []

        def test_callback(chunk, usage_data):
            received_chunks.append(chunk)

        # Test streaming
        for chunk, usage_data in llm.stream_generate(
            event_id="test_stream",
            system_prompt=self.system_prompt,
            messages=self.messages,
            callback=test_callback,
        ):
            # Verify usage data structure
            self.assertIn("event_id", usage_data)
            self.assertIn("model", usage_data)
            self.assertIn("read_tokens", usage_data)
            self.assertIn("write_tokens", usage_data)
            self.assertIn("total_tokens", usage_data)

            # Verify chunk is a string
            self.assertIsInstance(chunk, str)

        # Verify we got all chunks
        self.assertEqual(
            len(received_chunks), len(mock_chunks) + 1
        )  # +1 for the empty completion chunk

        # Verify the concatenated result
        expected_text = "Machine learning is a subset of artificial intelligence."
        received_text = "".join(chunk for chunk in received_chunks if chunk)
        self.assertEqual(received_text, expected_text)

    @patch("anthropic.Anthropic")
    def test_anthropic_streaming(self, mock_anthropic):
        """Test Anthropic streaming functionality with mocked client."""
        # Setup mock client and response
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Create mock message create method
        mock_client.messages.create = MagicMock()

        # Create mock chunks for the stream
        mock_chunks = []
        for text in [
            "Machine ",
            "learning ",
            "is ",
            "a ",
            "subset ",
            "of ",
            "artificial ",
            "intelligence.",
        ]:
            chunk = MagicMock()
            chunk.delta = MagicMock()
            chunk.delta.text = text
            mock_chunks.append(chunk)

        # Set the mock client to return our mock chunks
        mock_client.messages.create.return_value = mock_chunks

        # Create Anthropic LLM
        llm = get_llm_from_model(
            "claude-haiku-3.5", provider="anthropic", api_key=self.anthropic_key
        )

        # Monkeypatch the client
        llm.client = mock_client

        # Test callback
        received_chunks = []

        def test_callback(chunk, usage_data):
            received_chunks.append(chunk)

        # Test streaming
        for chunk, usage_data in llm.stream_generate(
            event_id="test_stream",
            system_prompt=self.system_prompt,
            messages=self.messages,
            callback=test_callback,
        ):
            # Verify usage data structure
            self.assertIn("event_id", usage_data)
            self.assertIn("model", usage_data)
            self.assertIn("read_tokens", usage_data)
            self.assertIn("write_tokens", usage_data)
            self.assertIn("total_tokens", usage_data)

            # Verify chunk is a string
            self.assertIsInstance(chunk, str)

        # Verify we got all chunks
        self.assertEqual(
            len(received_chunks), len(mock_chunks) + 1
        )  # +1 for the empty completion chunk

        # Verify the concatenated result
        expected_text = "Machine learning is a subset of artificial intelligence."
        received_text = "".join(chunk for chunk in received_chunks if chunk)
        self.assertEqual(received_text, expected_text)

    @patch("google.generativeai.GenerativeModel")
    def test_google_streaming(self, mock_generative_model):
        """Test Google streaming functionality with mocked client."""
        # Setup mock model and response
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model

        # Create mock chunks for the stream
        mock_chunks = []
        for text in [
            "Machine ",
            "learning ",
            "is ",
            "a ",
            "subset ",
            "of ",
            "artificial ",
            "intelligence.",
        ]:
            chunk = MagicMock()
            chunk.text = text
            mock_chunks.append(chunk)

        # Set the mock model to return our mock chunks
        mock_model.generate_content.return_value = mock_chunks

        # Create Google LLM with mocked auth
        with patch.object(GoogleLLM, "auth", return_value=None):
            llm = get_llm_from_model(
                "gemini-2.0-flash", provider="google", api_key=self.google_key
            )

        # Test callback
        received_chunks = []

        def test_callback(chunk, usage_data):
            received_chunks.append(chunk)

        # Patch the _format_messages_for_model method to avoid dependencies
        with patch.object(GoogleLLM, "_format_messages_for_model", return_value=[]):
            # Test streaming
            for chunk, usage_data in llm.stream_generate(
                event_id="test_stream",
                system_prompt=self.system_prompt,
                messages=self.messages,
                callback=test_callback,
            ):
                # Verify usage data structure
                self.assertIn("event_id", usage_data)
                self.assertIn("model", usage_data)
                self.assertIn("read_tokens", usage_data)
                self.assertIn("write_tokens", usage_data)
                self.assertIn("total_tokens", usage_data)

                # Verify chunk is a string
                self.assertIsInstance(chunk, str)

        # Verify we got all chunks
        self.assertEqual(
            len(received_chunks), len(mock_chunks) + 1
        )  # +1 for the empty completion chunk

        # Verify the concatenated result
        expected_text = "Machine learning is a subset of artificial intelligence."
        received_text = "".join(chunk for chunk in received_chunks if chunk)
        self.assertEqual(received_text, expected_text)

    def test_empty_messages(self):
        """Test streaming with empty messages."""
        # Create OpenAI LLM with a mock client
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.chat.completions.create.return_value = []

            llm = get_llm_from_model(
                "gpt-4o-mini", provider="openai", api_key=self.openai_key
            )
            llm.client = mock_client

            # Test with empty messages
            empty_messages = []

            chunks = list(
                llm.stream_generate(
                    event_id="test_empty",
                    system_prompt=self.system_prompt,
                    messages=empty_messages,
                )
            )

            # Should only get the final empty chunk with usage data
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0][0], "")
            self.assertTrue(chunks[0][1]["is_complete"])


if __name__ == "__main__":
    unittest.main()
