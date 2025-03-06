"""
Unit tests for the base LLM class.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from lluminary.models.base import LLM


class MockLLM(LLM):
    """Mock implementation of the LLM base class for testing."""

    CONTEXT_WINDOW = {"test-model": 4096}

    COST_PER_MODEL = {
        "test-model": {"read_token": 0.001, "write_token": 0.002, "image_cost": 0.01}
    }

    SUPPORTED_MODELS = ["test-model"]
    THINKING_MODELS = ["test-model"]
    EMBEDDING_MODELS = []
    RERANKING_MODELS = []

    def __init__(self, model_name: str = "test-model", **kwargs):
        """Initialize the MockLLM with default test model."""
        super().__init__(model_name, **kwargs)

    def _format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Mock implementation of message formatting."""
        return messages

    def auth(self) -> None:
        """Mock implementation of authentication."""
        pass

    def _raw_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        top_k: int = 200,
        tools: List[Dict[str, Any]] = None,
        thinking_budget: int = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Mock implementation of raw generation."""
        usage = {
            "read_tokens": 10,
            "write_tokens": 5,
            "images": 0,
            "total_tokens": 15,
            "read_cost": 0.01,
            "write_cost": 0.01,
            "image_cost": 0,
            "total_cost": 0.02,
        }
        return "test response", usage


@pytest.fixture
def mock_llm():
    """Fixture to create a MockLLM instance."""
    return MockLLM("test-model")


def test_llm_initialization(mock_llm):
    """Test LLM initialization."""
    assert mock_llm.model_name == "test-model"
    assert ["test-model"] == mock_llm.SUPPORTED_MODELS
    assert ["test-model"] == mock_llm.THINKING_MODELS
    assert "test-model" in mock_llm.CONTEXT_WINDOW
    assert "test-model" in mock_llm.COST_PER_MODEL


def test_validate_model(mock_llm):
    """Test model validation."""
    # Test valid model
    assert mock_llm.validate_model("test-model") is True

    # Test invalid model
    assert mock_llm.validate_model("invalid-model") is False


def test_get_context_window(mock_llm):
    """Test context window retrieval."""
    # Test getting context window
    assert mock_llm.get_context_window() == 4096

    # Test with invalid model
    with patch.object(mock_llm, "model_name", "invalid-model"):
        with pytest.raises(Exception):
            mock_llm.get_context_window()


def test_get_model_costs(mock_llm):
    """Test model cost retrieval."""
    # Test basic cost estimation
    cost = mock_llm.get_model_costs()
    assert cost["read_token"] == 0.001
    assert cost["write_token"] == 0.002
    assert cost["image_cost"] == 0.01

    # Test with invalid model
    with patch.object(mock_llm, "model_name", "invalid-model"):
        with pytest.raises(Exception):
            mock_llm.get_model_costs()


def test_message_formatting(mock_llm):
    """Test message formatting."""
    # Test basic message
    messages = [{"message_type": "human", "message": "test"}]
    formatted = mock_llm._format_messages_for_model(messages)
    assert formatted == messages


def test_convert_function_to_tool(mock_llm):
    """Test converting a function to a tool."""

    def test_func(param1: str, param2: int = 0) -> str:
        """Test function docstring"""
        return f"{param1} {param2}"

    tool = mock_llm._convert_function_to_tool(test_func)

    assert tool["name"] == "test_func"
    assert "description" in tool
    assert "input_schema" in tool
    assert "properties" in tool["input_schema"]
    assert "param1" in tool["input_schema"]["properties"]
    assert "param2" in tool["input_schema"]["properties"]
    assert tool["input_schema"]["properties"]["param1"]["type"] == "string"
    assert tool["input_schema"]["properties"]["param2"]["type"] == "integer"
    assert "required" in tool["input_schema"]
    assert "param1" in tool["input_schema"]["required"]


def test_convert_functions_to_tools(mock_llm):
    """Test converting multiple functions to tools."""

    def func1(a: str) -> str:
        """Func1 doc"""
        return a

    def func2(b: int) -> int:
        """Func2 doc"""
        return b

    tools = mock_llm._convert_functions_to_tools([func1, func2])

    assert len(tools) == 2
    assert tools[0]["name"] == "func1"
    assert tools[1]["name"] == "func2"


def test_generate(mock_llm):
    """Test the generate method."""
    # Mock the _raw_generate method to avoid actual API calls
    with patch.object(mock_llm, "_raw_generate") as mock_raw_generate:
        mock_raw_generate.return_value = (
            "Test response",
            {
                "read_tokens": 10,
                "write_tokens": 5,
                "images": 0,
                "total_tokens": 15,
                "read_cost": 0.01,
                "write_cost": 0.01,
                "image_cost": 0,
                "total_cost": 0.02,
            },
        )

        # Test basic generation
        result, usage, _ = mock_llm.generate(
            event_id="test_event",
            system_prompt="test prompt",
            messages=[{"message_type": "human", "message": "hello"}],
            max_tokens=100,
        )

        assert result == "Test response"
        assert "read_tokens" in usage
        assert "write_tokens" in usage
        assert "total_cost" in usage


def test_estimate_tokens(mock_llm):
    """Test token estimation."""
    # Test basic token estimation
    tokens = mock_llm.estimate_tokens("test message")
    assert isinstance(tokens, int)
    assert tokens > 0


def test_check_context_fit(mock_llm):
    """Test context fit checking."""
    # Test basic context fit
    fits, message = mock_llm.check_context_fit("test prompt", 100)
    assert fits is True
    assert isinstance(message, str)

    # Test large prompt
    large_prompt = "test " * 2000
    fits, message = mock_llm.check_context_fit(large_prompt, 1000)
    # This will either fit or not depending on the estimate, but should return a valid result
    assert isinstance(fits, bool)
    assert isinstance(message, str)


def test_estimate_cost(mock_llm):
    """Test cost estimation."""
    total_cost, breakdown = mock_llm.estimate_cost("test prompt", 100)
    assert total_cost > 0
    assert "prompt_cost" in breakdown
    assert "response_cost" in breakdown
    assert "image_cost" in breakdown


def test_supports_image_input(mock_llm):
    """Test image input support check."""
    assert isinstance(mock_llm.supports_image_input(), bool)


def test_get_supported_models(mock_llm):
    """Test retrieving supported models."""
    models = mock_llm.get_supported_models()
    assert isinstance(models, list)
    assert "test-model" in models


def test_embed_not_implemented(mock_llm):
    """Test that embed raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        mock_llm.embed(texts=["test text"])


def test_supports_embeddings(mock_llm):
    """Test supports_embeddings method."""
    assert mock_llm.supports_embeddings() is False


def test_supports_reranking(mock_llm):
    """Test supports_reranking method."""
    assert mock_llm.supports_reranking() is False


def test_rerank_not_implemented(mock_llm):
    """Test rerank raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        mock_llm.rerank(query="test", documents=["doc1"])


def test_stream_generate_not_implemented(mock_llm):
    """Test stream_generate raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        next(
            mock_llm.stream_generate(
                event_id="test",
                system_prompt="system",
                messages=[{"message_type": "human", "message": "hello"}],
            )
        )


class MockLLMWithEmbeddings(MockLLM):
    """Mock LLM with embedding support."""

    EMBEDDING_MODELS = ["test-model"]

    def embed(
        self, texts: List[str], model: Optional[str] = None, batch_size: int = 100
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """Mock implementation of embed."""
        embeddings = [[0.1, 0.2, 0.3] for _ in texts]
        usage = {
            "tokens": len(" ".join(texts)) // 4,
            "cost": 0.0001 * len(" ".join(texts)) // 4,
        }
        return embeddings, usage


def test_mock_with_embeddings():
    """Test the MockLLMWithEmbeddings class."""
    llm = MockLLMWithEmbeddings("test-model")

    # Test embedding support
    assert llm.supports_embeddings() is True

    # Test embed method
    embeddings, usage = llm.embed(texts=["test1", "test2"])
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert "tokens" in usage
    assert "cost" in usage


class MockLLMWithStreaming(MockLLM):
    """Mock LLM with streaming support."""

    def stream_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        functions: List[Callable] = None,
        callback: Callable[[str, Dict[str, Any]], None] = None,
    ):
        """Mock implementation of stream_generate."""
        chunks = ["This ", "is ", "a ", "test ", "response."]

        for i, chunk in enumerate(chunks):
            usage = {
                "read_tokens": 10,
                "write_tokens": i + 1,
                "total_tokens": 10 + i + 1,
                "is_complete": False,
            }

            if callback:
                callback(chunk, usage)

            yield chunk, usage

        # Final usage
        final_usage = {
            "read_tokens": 10,
            "write_tokens": len(chunks),
            "total_tokens": 10 + len(chunks),
            "is_complete": True,
            "read_cost": 0.01,
            "write_cost": 0.01,
            "total_cost": 0.02,
        }

        if callback:
            callback("", final_usage)

        yield "", final_usage


def test_mock_with_streaming():
    """Test the MockLLMWithStreaming class."""
    llm = MockLLMWithStreaming("test-model")

    # Test with callback
    chunks = []
    usages = []

    def callback(chunk, usage):
        chunks.append(chunk)
        usages.append(usage)

    # Collect chunks using the generator
    collected_chunks = []
    collected_usages = []

    for chunk, usage in llm.stream_generate(
        event_id="test",
        system_prompt="system",
        messages=[{"message_type": "human", "message": "hello"}],
        callback=callback,
    ):
        collected_chunks.append(chunk)
        collected_usages.append(usage)

    # Verify callback results
    assert len(chunks) == 6  # 5 content chunks + 1 empty final chunk
    assert chunks[-1] == ""  # Last chunk should be empty
    assert usages[-1]["is_complete"] is True
    assert "total_cost" in usages[-1]

    # Verify generator results
    assert len(collected_chunks) == 6
    assert collected_chunks[-1] == ""
    assert collected_usages[-1]["is_complete"] is True
