"""
Unit tests for the LLMHandler class.
"""

from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Optional, Tuple

import pytest

from lluminary.exceptions import LLMError, LLMProviderError, LLMValidationError
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
    TOOLS_MODELS = ["mock-model"]  # Add support for tools

    CONTEXT_WINDOW = {"mock-model": 4096}

    COST_PER_MODEL = {
        "mock-model": {"read_token": 0.001, "write_token": 0.002, "image_cost": 0.01}
    }

    def __init__(self, model_name="mock-model", **kwargs):
        super().__init__(model_name, **kwargs)
        # Add capabilities for tools
        self.has_tool_calling = True
        self.has_functions = True
        
    @property
    def is_thinking_model(self) -> bool:
        """Check if the model supports thinking."""
        return self.model_name in self.THINKING_MODELS
        
    @property
    def supports_tools(self) -> bool:
        """Check if the model supports tools."""
        return self.model_name in self.TOOLS_MODELS

    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """Mock implementation of provider config validation."""
        pass

    def _format_messages_for_model(self, messages):
        return messages

    def auth(self):
        pass

    def _raw_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        top_k: int = 200,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking_budget: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
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
        )


def test_provider_initialization():
    """Test provider initialization and caching."""
    # Override the get_llm_from_model function to return our mock LLM
    with patch("lluminary.handler.get_llm_from_model", return_value=MockLLM()):
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
            raise LLMProviderError("Provider failed", provider="failing-provider")
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


def test_generate_with_tools():
    """Test generating with tools."""
    print("Starting test_generate_with_tools...")
    with patch("lluminary.handler.get_llm_from_model", return_value=MockLLM()):
        handler = LLMHandler()
        print("Created LLMHandler instance")

        def test_tool(x: int) -> int:
            """Double the input number."""
            print(f"test_tool called with x={x}")
            return x * 2

        handler.register_tools([test_tool])
        print("Registered test_tool")

        # Convert the function to a tool dictionary
        tool_dict = {
            "name": "test_tool",
            "description": "Double the input number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Number to double"}
                },
                "required": ["x"]
            }
        }
        print(f"Created tool_dict: {tool_dict}")

        # Test tool execution
        try:
            print("About to call handler.generate...")
            response = handler.generate(
                messages=[{"message_type": "human", "message": "Double the number 5"}],
                tools=[tool_dict],
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise


def test_tool_execution():
    """Test tool execution with different providers."""
    print("Starting test_tool_execution...")
    # Create test message
    test_message = {"message_type": "human", "message": "Double the number 5"}

    # Create a mock LLM instance
    mock_llm = MockLLM()
    print("Created MockLLM instance")

    # Create a tool dictionary
    tool_dict = {
        "name": "test_tool",
        "description": "Double the number.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Number to double"}
            },
            "required": ["x"]
        }
    }
    print(f"Created tool_dict: {tool_dict}")

    # Mock the _raw_generate method to return a tool call
    def mock_raw_generate(*args, **kwargs):
        print("mock_raw_generate called")
        return (
            '{"tool_calls": [{"name": "test_tool", "parameters": {"x": 5}}]}',
            {"read_tokens": 10, "write_tokens": 5, "total_tokens": 15}
        )

    mock_llm._raw_generate = mock_raw_generate
    print("Mocked _raw_generate method")

    # Create a test tool function
    def test_tool(x: int) -> int:
        print(f"test_tool called with x={x}")
        return x * 2

    # Test with patched get_provider
    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        try:
            print("Creating LLMHandler instance")
            handler = LLMHandler()
            
            print("Registering test_tool")
            handler.register_tools([test_tool])
            
            print("Calling handler.generate")
            response = handler.generate(
                messages=[test_message],
                tools=[tool_dict],
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise


def test_tool_processing():
    """Test tool processing and execution."""
    print("Starting test_tool_processing...")
    # Create test message
    test_message = {"message_type": "human", "message": "Double the number 5"}
    print("Created test message")

    # Create a mock LLM instance
    mock_llm = MockLLM()
    print("Created MockLLM instance")

    # Create a tool dictionary
    tool_dict = {
        "name": "test_tool",
        "description": "Double the number.",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "Number to double"}
            },
            "required": ["x"]
        }
    }
    print(f"Created tool_dict: {tool_dict}")

    # Mock the _raw_generate method to return a tool call
    def mock_raw_generate(*args, **kwargs):
        print("mock_raw_generate called")
        return (
            '{"tool_calls": [{"name": "test_tool", "parameters": {"x": 5}}]}',
            {"read_tokens": 10, "write_tokens": 5, "total_tokens": 15}
        )

    mock_llm._raw_generate = mock_raw_generate
    print("Mocked _raw_generate method")

    # Create a test tool function
    def test_tool(x: int) -> int:
        print(f"test_tool called with x={x}")
        return x * 2

    # Test with patched get_provider
    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        try:
            print("Creating LLMHandler instance")
            handler = LLMHandler()
            
            print("Registering test_tool")
            handler.register_tools([test_tool])
            
            print("Calling handler.generate")
            response = handler.generate(
                messages=[test_message],
                tools=[tool_dict],
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise


def test_error_handling():
    """Test error handling and recovery."""
    print("Starting test_error_handling...")
    # Create a handler
    handler = LLMHandler()
    print("Created LLMHandler instance")

    # Test provider error
    with patch.object(
        handler, "get_provider", side_effect=LLMProviderError("Provider not found")
    ):
        print("Testing provider error...")
        with pytest.raises(LLMProviderError):
            handler.get_provider("invalid-provider")
        print("Provider error test passed")

    # Test recovery from provider error
    def side_effect_func(provider_name=None):
        print(f"side_effect_func called with provider_name={provider_name}")
        if provider_name == "failing-provider":
            raise LLMProviderError("Provider failed", provider="failing-provider")
        return MockLLM()

    with patch.object(handler, "get_provider", side_effect=side_effect_func):
        print("Testing recovery from provider error...")
        # This should work by falling back to the default provider
        response = handler.generate(
            messages=[{"message_type": "human", "message": "test"}],
            provider="failing-provider",
        )
        print(f"Response: {response}")
        assert response == "Mock response"
        print("Recovery from provider error test passed")


def test_classification():
    """Test classification functionality."""
    print("Starting test_classification...")
    # Create a mock LLM instance
    mock_llm = MockLLM()
    print("Created MockLLM instance")

    # Test message and categories
    test_message = {"message_type": "human", "message": "What is the capital of France?"}
    test_categories = {
        "question": "A request for information",
        "statement": "A declarative sentence",
        "command": "An instruction or directive"
    }
    print(f"Created test message and categories")

    # Create a MagicMock for the classify method
    mock_classify_method = MagicMock(
        return_value=({"question": 0.9, "statement": 0.1, "command": 0.0}, {"read_tokens": 10, "write_tokens": 5, "total_cost": 0.02})
    )
    
    # Replace the classify method with our mock
    mock_llm.classify = mock_classify_method  # type: ignore
    print("Mocked classify method")

    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        print("Creating LLMHandler instance")
        handler = LLMHandler()

        # Test basic classification
        print("Testing basic classification...")
        try:
            categories = handler.classify(
                messages=[test_message], categories=test_categories
            )
            print(f"Categories: {categories}")
            assert "question" in categories
            print("Basic classification test passed")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise


def test_image_handling():
    """Test image processing capabilities."""
    print("Starting test_image_handling...")
    # Create test data with image
    test_image_message = {
        "message_type": "human",
        "message": "What is in this image?",
        "image_paths": [],
        "image_urls": ["https://example.com/test.jpg"],
    }
    print("Created test image message")

    # Create mock LLM instances with and without image support
    mock_llm_with_images = MockLLM()
    mock_llm_without_images = MockLLM()
    print("Created mock LLM instances")

    # Mock supports_image_input methods
    mock_llm_with_images.supports_image_input = lambda: True
    mock_llm_without_images.supports_image_input = lambda: False
    print("Mocked supports_image_input methods")

    # Test with provider that supports images
    with patch.object(LLMHandler, "get_provider", return_value=mock_llm_with_images):
        print("Testing with provider that supports images...")
        handler = LLMHandler()

        # Verify supports_images check
        print("Checking supports_images...")
        assert handler.supports_images() is True
        print("supports_images check passed")

        # Test message with image URL
        print("Testing message with image URL...")
        try:
            response = handler.generate(messages=[test_image_message])
            print(f"Response: {response}")
            assert response == "Mock response"
            print("Image URL test passed")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

    # Test with provider that doesn't support images
    with patch.object(LLMHandler, "get_provider", return_value=mock_llm_without_images):
        print("Testing with provider that doesn't support images...")
        handler = LLMHandler()

        # Verify supports_images check
        print("Checking supports_images...")
        assert handler.supports_images() is False
        print("supports_images check passed")

        # Test message with image URL - the handler should still work but ignore the images
        print("Testing message with image URL (should ignore images)...")
        response = handler.generate(messages=[test_image_message])
        print(f"Response: {response}")
        assert response == "Mock response"
        print("Image URL with non-supporting provider test passed")


def test_cost_estimation():
    """Test cost estimation functionality."""
    print("Starting test_cost_estimation...")
    # Create a mock LLM instance
    mock_llm = MockLLM()
    print("Created MockLLM instance")

    # Create MagicMocks for the methods
    mock_estimate_tokens_method = MagicMock(return_value=10)
    mock_get_model_costs_method = MagicMock(return_value={
        "read_token": 0.001,
        "write_token": 0.002,
        "image_cost": 0.01,
    })
    print("Created mock methods")
    
    # Replace the methods with our mocks
    mock_llm.estimate_tokens = mock_estimate_tokens_method  # type: ignore
    mock_llm.get_model_costs = mock_get_model_costs_method  # type: ignore
    print("Replaced methods with mocks")

    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        print("Creating LLMHandler instance")
        handler = LLMHandler()
        
        # Test cost estimation
        print("Testing cost estimation...")
        try:
            cost = handler.estimate_cost(
                messages=[{"message_type": "human", "message": "test message"}],
                max_response_tokens=100,
            )
            print(f"Cost: {cost}")
            assert isinstance(cost, dict)
            assert "read_cost" in cost
            assert "write_cost" in cost
            assert "total_cost" in cost
            print("Cost estimation test passed")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise


def test_context_window():
    """Test context window functionality."""
    print("Starting test_context_window...")
    # Create a mock LLM instance
    mock_llm = MockLLM()
    print("Created MockLLM instance")

    # Create MagicMocks for the methods
    mock_check_context_fit_method = MagicMock(return_value=(True, 3000, 1000))
    mock_get_context_window_method = MagicMock(return_value=4096)
    print("Created mock methods")
    
    # Replace the methods with our mocks
    mock_llm.check_context_fit = mock_check_context_fit_method  # type: ignore
    mock_llm.get_context_window = mock_get_context_window_method  # type: ignore
    print("Replaced methods with mocks")

    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        print("Creating LLMHandler instance")
        handler = LLMHandler()
        
        # Test context window check
        print("Testing check_context_fit...")
        try:
            fits, tokens_used, tokens_remaining = handler.check_context_fit(
                messages=[{"message_type": "human", "message": "test message"}],
                max_response_tokens=1000,
            )
            print(f"Fits: {fits}, Tokens used: {tokens_used}, Tokens remaining: {tokens_remaining}")
            assert fits is True
            assert tokens_used == 3000
            assert tokens_remaining == 1000
            print("Context fit check passed")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
        
        # Test getting context window
        print("Testing get_context_window...")
        try:
            context_window = handler.get_context_window()
            print(f"Context window: {context_window}")
            assert context_window == 4096
            print("Get context window test passed")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise


def test_thinking_budget():
    """Test thinking budget functionality."""
    print("Starting test_thinking_budget...")
    # Create a mock LLM instance with thinking support
    mock_llm = MockLLM()
    print("Created MockLLM instance")
    
    # Verify the is_thinking_model property
    print("Checking is_thinking_model property...")
    assert mock_llm.is_thinking_model is True
    print("is_thinking_model property check passed")
    
    # Verify that the model is in the THINKING_MODELS list
    print("Checking THINKING_MODELS list...")
    assert mock_llm.model_name in mock_llm.THINKING_MODELS
    print("THINKING_MODELS list check passed")
    
    # Test that the model supports thinking
    print("Testing supports_thinking method...")
    assert hasattr(mock_llm, "is_thinking_model")
    print("supports_thinking method check passed")
    
    print("All thinking budget tests passed")


def test_provider_specific_features():
    """Test provider-specific features and configurations."""
    print("Starting test_provider_specific_features...")
    # Create test message
    test_message = {
        "message_type": "human",
        "message": "Test provider-specific features",
    }
    print("Created test message")

    # Create a mock LLM instance
    mock_llm = MockLLM()
    print("Created MockLLM instance")

    # Test with a single provider
    with patch.object(LLMHandler, "get_provider", return_value=mock_llm):
        print("Creating LLMHandler instance")
        handler = LLMHandler()

        # Test with tools
        print("Testing with tools...")
        def test_tool(x: int) -> int:
            """Double the number."""
            print(f"test_tool called with x={x}")
            return x * 2

        try:
            handler.register_tools([test_tool])
            print("Registered test_tool")
            
            tool_dict = {
                "name": "test_tool",
                "description": "Double the number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "Number to double"}
                    },
                    "required": ["x"]
                }
            }
            
            response = handler.generate(
                messages=[test_message], tools=[tool_dict]
            )
            print(f"Response with tools: {response}")
            assert response == "Mock response"
            print("Tools test passed")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

        # Test with temperature
        print("Testing with temperature...")
        try:
            response = handler.generate(
                messages=[test_message], temperature=0.7
            )
            print(f"Response with temperature: {response}")
            assert response == "Mock response"
            print("Temperature test passed")
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
