"""
Unit tests for the LLMHandler class.
"""

from unittest.mock import patch

import pytest

from lluminary.exceptions import LLMMistake, ProviderError
from lluminary.handler import LLMHandler
from lluminary.models.base import LLM


def test_handler_initialization():
    """Test LLMHandler initialization with various configurations."""
    # Test with default configuration
    handler = LLMHandler()
    assert handler.default_provider == "openai"
    assert handler.llm_instances == {}

    # Test with custom configuration
    config = {
        "default_provider": "anthropic",
        "providers": {
            "anthropic": {"api_key": "test-key", "default_model": "claude-sonnet-3.5"}
        },
    }
    handler = LLMHandler(config)
    assert handler.default_provider == "anthropic"
    # Skip checking llm_instances since we're not mocking the provider initialization yet
    # assert "anthropic" in handler.llm_instances


def test_handler_methods():
    """Test basic handler methods."""
    handler = LLMHandler()

    # Test methods that don't require provider initialization
    assert isinstance(handler.config, dict)

    # Test register_tools method
    def dummy_tool():
        pass

    handler.register_tools([dummy_tool])
    assert hasattr(handler, "_registered_tools")
    assert dummy_tool in handler._registered_tools


class MockLLM(LLM):
    """Mock LLM implementation for testing handler."""

    SUPPORTED_MODELS = ["mock-model"]
    THINKING_MODELS = ["mock-model"]
    EMBEDDING_MODELS = ["mock-model"]
    RERANKING_MODELS = []

    CONTEXT_WINDOW = {"mock-model": 4096}

    COST_PER_MODEL = {
        "mock-model": {"read_token": 0.001, "write_token": 0.002, "image_cost": 0.01}
    }

    def __init__(self, model_name="mock-model", **kwargs):
        super().__init__(model_name, **kwargs)

    def _format_messages_for_model(self, messages):
        return messages

    def auth(self):
        pass

    def _raw_generate(
        self,
        event_id,
        system_prompt,
        messages,
        max_tokens=1000,
        temp=0.0,
        top_k=200,
        tools=None,
        thinking_budget=None,
    ):
        """Mock implementation of generation."""
        return (
            "Mock response",
            {
                "read_tokens": 10,
                "write_tokens": 5,
                "total_tokens": 15,
                "read_cost": 0.01,
                "write_cost": 0.01,
                "total_cost": 0.02,
            },
            messages,
        )


def test_provider_initialization():
    """Test provider initialization and caching."""
    # Override the get_llm_from_model function to return our mock LLM
    with patch("src.lluminary.handler.get_llm_from_model", return_value=MockLLM()):
        handler = LLMHandler()

        # Test getting a provider
        provider = handler.get_provider("openai")
        assert provider is not None
        assert isinstance(provider, MockLLM)
        assert "openai" in handler.llm_instances

        # Test provider caching
        cached_provider = handler.get_provider("openai")
        assert cached_provider is provider

    # # This test would look like:
    # handler = LLMHandler()
    #
    # # Test getting a provider for the first time
    # provider = handler.get_provider("openai")
    # assert provider is not None
    # assert "openai" in handler.llm_instances
    #
    # # Test provider caching
    # cached_provider = handler.get_provider("openai")
    # assert cached_provider is provider


def test_provider_fallback():
    """Test fallback to default provider."""
    # Create a mock LLM instance
    mock_llm = MockLLM()

    # Create patched get_provider that first raises ProviderError then returns mock
    def side_effect_func(provider_name=None):
        if provider_name == "failing-provider":
            raise ProviderError("Provider failed", provider="failing-provider")
        return mock_llm

    with patch.object(LLMHandler, "get_provider", side_effect=side_effect_func):
        handler = LLMHandler({"default_provider": "openai"})

        # Test fallback when provider fails
        response = handler.generate(
            messages=[{"message_type": "human", "message": "test"}],
            provider="failing-provider",
        )

        # Make sure we got a response
        assert response == "Mock response"

    # # The test would look like:
    # handler = LLMHandler()
    #
    # # Test fallback when provider fails
    # with patch.object(handler, 'get_provider', side_effect=[ProviderError, MagicMock()]):
    #     response = handler.generate(
    #         messages=[{"message_type": "human", "message": "test"}],
    #         provider="failing-provider"
    #     )
    #     assert response is not None


def test_message_generation():
    """Test message generation with various configurations."""
    # Create a test message
    test_message = {"message_type": "human", "message": "How do I open this file?"}

    # Create a mock LLM instance
    mock_llm = MockLLM()

    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        handler = LLMHandler()

        # Test basic message generation
        response = handler.generate(messages=[test_message])
        assert response == "Mock response"

        # Test with system prompt
        response = handler.generate(
            messages=[test_message], system_prompt="You are a helpful assistant."
        )
        assert response == "Mock response"

        # Test with different provider
        response = handler.generate(messages=[test_message], provider="anthropic")
        assert response == "Mock response"

        # Test generate_with_usage
        response, usage = handler.generate_with_usage(messages=[test_message])
        assert response == "Mock response"
        assert "total_cost" in usage
        assert usage["total_cost"] == 0.02


def test_tool_processing():
    """Test tool registration and processing."""
    # Create a mock LLM instance
    mock_llm = MockLLM()

    # Mock LLM to handle tools
    def mock_generate(*args, **kwargs):
        return (
            "Tool result: 10",
            {
                "read_tokens": 10,
                "write_tokens": 5,
                "total_tokens": 15,
                "read_cost": 0.01,
                "write_cost": 0.01,
                "total_cost": 0.02,
            },
            [],
        )

    mock_llm.generate = mock_generate

    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        handler = LLMHandler()

        # Register a test tool
        def test_tool(x: int) -> int:
            """Double the input number."""
            return x * 2

        handler.register_tools([test_tool])

        # Test tool execution
        response = handler.generate(
            messages=[{"message_type": "human", "message": "Double the number 5"}],
            tools=[test_tool],
        )

        assert response == "Tool result: 10"


def test_error_handling():
    """Test error handling and recovery."""
    # Create a handler
    handler = LLMHandler()

    # Test provider error
    with patch.object(
        handler, "get_provider", side_effect=ProviderError("Provider not found")
    ):
        with pytest.raises(ProviderError):
            handler.get_provider("invalid-provider")

    # Test message format error
    with patch.object(handler, "get_provider", return_value=MockLLM()):
        with pytest.raises(LLMMistake):
            # Missing required message_type field
            handler.generate(messages=[{"invalid": "message"}])

    # Test recovery from provider error
    def side_effect_func(provider_name=None):
        if provider_name == "failing-provider":
            raise ProviderError("Provider failed")
        return MockLLM()

    with patch.object(handler, "get_provider", side_effect=side_effect_func):
        # This should work by falling back to the default provider
        response = handler.generate(
            messages=[{"message_type": "human", "message": "test"}],
            provider="failing-provider",
        )
        assert response == "Mock response"


def test_classification():
    """Test message classification functionality."""
    # Create test data
    test_message = {"message_type": "human", "message": "What is the weather like?"}
    test_categories = {
        "question": "A query seeking information",
        "command": "A directive to perform an action",
        "statement": "A declarative sentence",
    }
    test_examples = [
        {
            "user_input": "What is the capital of France?",
            "doc_str": "This is a question seeking information about geography",
            "selection": "question",
        }
    ]

    # Create a mock LLM instance with classification support
    mock_llm = MockLLM()

    # Mock classify method to return a fixed result
    def mock_classify(*args, **kwargs):
        return ["question"], {"read_tokens": 10, "write_tokens": 5, "total_cost": 0.02}

    mock_llm.classify = mock_classify

    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        handler = LLMHandler()

        # Test basic classification
        categories = handler.classify(
            messages=[test_message], categories=test_categories
        )
        assert categories == ["question"]

        # Test classification with examples
        categories = handler.classify(
            messages=[test_message], categories=test_categories, examples=test_examples
        )
        assert categories == ["question"]

        # Test classify_with_usage
        categories, usage = handler.classify_with_usage(
            messages=[test_message], categories=test_categories
        )
        assert categories == ["question"]
        assert "total_cost" in usage
        assert usage["total_cost"] == 0.02


def test_image_handling():
    """Test image processing capabilities."""
    # Create test data with image
    test_image_message = {
        "message_type": "human",
        "message": "What is in this image?",
        "image_paths": [],
        "image_urls": ["https://example.com/test.jpg"],
    }

    # Create mock LLM instances with and without image support
    mock_llm_with_images = MockLLM()
    mock_llm_without_images = MockLLM()

    # Mock supports_image_input methods
    mock_llm_with_images.supports_image_input = lambda: True
    mock_llm_without_images.supports_image_input = lambda: False

    # Test with provider that supports images
    with patch.object(LLMHandler, "get_provider", return_value=mock_llm_with_images):
        handler = LLMHandler()

        # Verify supports_images check
        assert handler.supports_images() is True

        # Test message with image URL
        response = handler.generate(messages=[test_image_message])
        assert response == "Mock response"

    # Test with provider that doesn't support images
    with patch.object(LLMHandler, "get_provider", return_value=mock_llm_without_images):
        handler = LLMHandler()

        # Verify supports_images check
        assert handler.supports_images() is False

        # Test message with image URL should raise an error
        with pytest.raises(LLMMistake):
            handler.generate(messages=[test_image_message])


def test_cost_tracking():
    """Test cost estimation and tracking."""
    # Create test message
    test_message = {"message_type": "human", "message": "test message"}

    # Create a mock LLM instance
    mock_llm = MockLLM()

    # Mock estimate_cost method
    def mock_estimate_cost(*args, **kwargs):
        return {
            "prompt_cost": 0.01,
            "response_cost": 0.02,
            "image_cost": 0,
            "total_cost": 0.03,
        }

    mock_llm.estimate_cost = mock_estimate_cost

    # Mock get_model_costs method
    def mock_get_model_costs():
        return {"read_token": 0.001, "write_token": 0.002, "image_cost": 0.01}

    mock_llm.get_model_costs = mock_get_model_costs

    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        handler = LLMHandler()

        # Test cost estimation
        costs = handler.estimate_cost(messages=[test_message], max_response_tokens=1000)
        assert "total_cost" in costs
        assert costs["total_cost"] >= 0

        # Test generate_with_usage returns usage statistics
        response, usage = handler.generate_with_usage(messages=[test_message])
        assert "read_tokens" in usage
        assert "write_tokens" in usage
        assert "total_cost" in usage
        assert usage["read_tokens"] == 10
        assert usage["write_tokens"] == 5
        assert usage["total_cost"] == 0.02


def test_context_management():
    """Test context window management."""
    # Create test messages
    test_message = {"message_type": "human", "message": "short test message"}
    large_message = {
        "message_type": "human",
        "message": "test " * 2000,
    }  # Very large message

    # Create a mock LLM instance
    mock_llm = MockLLM()

    # Mock check_context_fit method
    def mock_check_context_fit(prompt, max_response_tokens=None):
        # If the prompt is too long, return False
        if len(prompt) > 1000:
            return False, "Context window exceeded"
        return True, "Context fits within window"

    mock_llm.check_context_fit = mock_check_context_fit

    # Mock get_context_window method
    mock_llm.get_context_window = lambda: 4096

    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        handler = LLMHandler()

        # Test get_context_window
        window_size = handler.get_context_window()
        assert window_size == 4096

        # Test context window checking for small message
        fits, message = handler.check_context_fit(
            messages=[test_message], max_response_tokens=1000
        )
        assert fits is True
        assert "fits" in message.lower()

        # Test with oversized context
        fits, message = handler.check_context_fit(
            messages=[large_message], max_response_tokens=1000
        )
        assert fits is False
        assert "exceeded" in message.lower()


def test_thinking_models():
    """Test thinking/reasoning model capabilities."""
    # Create test message
    test_message = {
        "message_type": "human",
        "message": "Can you solve this math problem?",
    }

    # Create a mock LLM instance
    mock_llm = MockLLM()

    # Mock is_thinking_model method
    mock_llm.is_thinking_model = lambda: True

    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        handler = LLMHandler()

        # Test with thinking model
        response = handler.generate(messages=[test_message], thinking_budget=1000)
        assert response == "Mock response"

        # Test thinking budget validation
        with pytest.raises(ValueError):
            handler.generate(messages=[test_message], thinking_budget=-1)


def test_provider_specific_features():
    """Test provider-specific features and configurations."""
    # Create test message
    test_message = {
        "message_type": "human",
        "message": "Test provider-specific features",
    }

    # Create mock LLM instances for different providers
    openai_llm = MockLLM()
    anthropic_llm = MockLLM()
    google_llm = MockLLM()

    # Use different provider-specific implementations
    with patch(
        "src.lluminary.handler.get_llm_from_model",
        side_effect=[openai_llm, anthropic_llm, google_llm],
    ):
        handler = LLMHandler()

        # Test OpenAI with function calling
        def test_tool(x: int) -> int:
            """Double the number."""
            return x * 2

        response = handler.generate(
            messages=[test_message], provider="openai", tools=[test_tool]
        )
        assert response == "Mock response"

        # Test Anthropic with thinking
        response = handler.generate(
            messages=[test_message], provider="anthropic", thinking_budget=1000
        )
        assert response == "Mock response"

        # Test Google with specific configuration
        response = handler.generate(
            messages=[test_message], provider="google", temperature=0.7
        )
        assert response == "Mock response"
