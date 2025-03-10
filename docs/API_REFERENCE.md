# LLUMINARY API REFERENCE

## Overview

This document provides detailed API references for the LLuMinary library, including core interfaces, provider implementations, and advanced features.

## Table of Contents

- [Core Interfaces](#core-interfaces)
  - [LLM Base Class](#llm-base-class)
  - [Provider Registration](#provider-registration)
  - [Configuration](#configuration)
- [Provider-Specific Implementations](#provider-specific-implementations)
  - [OpenAI Provider](#openai-provider)
  - [Anthropic Provider](#anthropic-provider)
  - [Google Provider](#google-provider)
  - [Cohere Provider](#cohere-provider)
- [Advanced Features](#advanced-features)
  - [Embeddings](#embeddings)
  - [Document Reranking](#document-reranking)
  - [Streaming Responses](#streaming-responses)
  - [Function Calling](#function-calling)
  - [Tool Use](#tool-use)
- [Error Handling](#error-handling)
  - [Exceptions](#exceptions)
  - [Retry Logic](#retry-logic)

## Core Interfaces

### LLM Base Class

The `LLM` base class defines the common interface for all LLM provider implementations.

```python
class LLM:
    """Base class for LLM providers."""

    def generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        functions: List[Callable] = None,
        retry_limit: int = 3
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate a response from the LLM.

        Args:
            event_id (str): Unique identifier for this generation event
            system_prompt (str): System-level instructions for the model
            messages (List[Dict[str, Any]]): List of messages in the standard format
            max_tokens (int): Maximum number of tokens to generate
            temp (float): Temperature for generation
            functions (List[Callable]): List of functions the LLM can use
            retry_limit (int): Maximum number of retry attempts

        Returns:
            Tuple containing:
            - Generated response text
            - Usage statistics including tokens and costs
            - Updated messages list including the response

        Raises:
            NotImplementedError: If the provider doesn't implement generation
        """

    def stream_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        functions: List[Callable] = None,
        callback: Callable[[str, Dict[str, Any]], None] = None
    ):
        """
        Stream a response from the LLM, yielding chunks as they become available.

        Args:
            event_id (str): Unique identifier for this generation event
            system_prompt (str): System-level instructions for the model
            messages (List[Dict[str, Any]]): List of messages in the standard format
            max_tokens (int): Maximum number of tokens to generate
            temp (float): Temperature for generation
            functions (List[Callable]): List of functions the LLM can use
            callback (Callable): Optional callback function for each chunk

        Yields:
            Tuple[str, Dict[str, Any]]: Tuples of (text_chunk, partial_usage_data)

        Raises:
            NotImplementedError: If the provider doesn't implement streaming
        """

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 100,
        **kwargs
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts (List[str]): List of texts to embed
            model (Optional[str]): Specific embedding model to use, defaults to a provider-specific model
            batch_size (int): Number of texts to process in each batch
            **kwargs: Additional provider-specific parameters

        Returns:
            Tuple[List[List[float]], Dict[str, Any]]:
                - List of embedding vectors (one per input text)
                - Usage information with token counts and costs

        Raises:
            NotImplementedError: If the provider doesn't implement embedding
            ValueError: If embedding fails
        """

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = None,
        return_scores: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Rerank a list of documents based on their relevance to a query.

        Args:
            query (str): The search query to use for ranking
            documents (List[str]): List of document texts to rerank
            top_n (int, optional): Number of top documents to return
            return_scores (bool): Whether to include relevance scores
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'ranked_documents' (List[str]): List of reranked document texts
                - 'indices' (List[int]): Original indices of the reranked documents
                - 'scores' (List[float], optional): Relevance scores if return_scores=True
                - 'usage': Usage information with token counts and costs

        Raises:
            NotImplementedError: If the model doesn't support reranking
        """

    def classify(
        self,
        messages: List[Dict[str, Any]],
        categories: Dict[str, str],
        examples: Optional[List[Dict[str, Any]]] = None,
        max_options: int = 1,
        system_prompt: Optional[str] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Classify messages into predefined categories.

        Args:
            messages (List[Dict[str, Any]]): Messages to classify
            categories (Dict[str, str]): Dictionary mapping category names to descriptions
            examples (Optional[List[Dict[str, Any]]]): Optional example classifications
            max_options (int): Maximum number of categories to select
            system_prompt (Optional[str]): Optional system prompt override

        Returns:
            Tuple[List[str], Dict[str, Any]]:
                - List of selected category names
                - Usage information with token counts and costs
        """

    def supports_embeddings(self) -> bool:
        """
        Check if this provider and model supports embeddings.

        Returns:
            bool: True if embeddings are supported, False otherwise
        """

    def supports_reranking(self) -> bool:
        """
        Check if this provider and model supports document reranking.

        Returns:
            bool: True if reranking is supported, False otherwise
        """

    def supports_image_input(self) -> bool:
        """
        Check if this provider and model supports image inputs.

        Returns:
            bool: True if image inputs are supported, False otherwise
        """

    def get_model_costs(self) -> Dict[str, float]:
        """
        Get the cost information for the current model.

        Returns:
            Dict[str, float]: Dictionary with cost per token information
        """
```

### Provider Registration

The provider registry pattern enables dynamic registration of LLM providers.

```python
def register_provider(provider_name: str, provider_class: Type[LLM]) -> None:
    """
    Register a new LLM provider class.

    Args:
        provider_name (str): Name of the provider (e.g., 'openai', 'anthropic')
        provider_class (Type[LLM]): The provider class, must extend LLM base class

    Raises:
        TypeError: If provider_class is not a subclass of LLM
    """

def get_llm_from_model(model_name: str, **kwargs) -> LLM:
    """
    Get the appropriate LLM instance based on the friendly model name.

    Args:
        model_name (str): Friendly name of the model
        **kwargs: Additional configuration parameters for the LLM

    Returns:
        LLM: An instance of the appropriate LLM provider

    Raises:
        ValueError: If the model name is not recognized or provider isn't registered
    """

def register_model(
    friendly_name: str,
    provider_name: str,
    model_id: str
) -> None:
    """
    Register a new model with the system.

    Args:
        friendly_name (str): User-friendly name for the model
        provider_name (str): Name of the provider (must be registered)
        model_id (str): Provider-specific model identifier

    Raises:
        ValueError: If the provider is not registered
    """

def list_available_models() -> Dict[str, list[str]]:
    """
    Get a dictionary of all available models grouped by provider.

    Returns:
        Dict[str, list[str]]: Dictionary mapping provider names to lists of their friendly model names
    """
```

### Configuration

```python
class LLMHandler:
    """
    Main class for handling LLM operations.
    Provides a unified interface for working with different LLM providers
    and managing message generation, tool usage, and error handling.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM handler with configuration.

        Args:
            config: Configuration dictionary containing:
                - default_provider: Name of the default LLM provider
                - providers: Dictionary of provider-specific configurations
                - logging: Logging configuration
        """

    def get_provider(self, provider_name: Optional[str] = None) -> LLM:
        """
        Get an LLM provider instance.

        Args:
            provider_name: Optional name of the provider to use.
                         If not specified, uses the default provider.

        Returns:
            LLM: The requested LLM provider instance

        Raises:
            ValueError: If the provider is not found or not initialized
        """

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        provider: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        retry_limit: int = 3
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response using the specified or default provider.

        Args:
            messages: List of messages in the standard format
            system_prompt: System-level instructions for the model
            provider: Optional provider name to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            tools: Optional list of tools/functions to use
            retry_limit: Maximum number of retry attempts

        Returns:
            Tuple containing:
            - Generated response text
            - Usage statistics including tokens and costs

        Raises:
            Exception: If generation fails after retries
        """
```

## Provider-Specific Implementations

### OpenAI Provider

```python
class OpenAILLM(LLM):
    """
    Implementation of the LLM interface for the OpenAI API.
    Supports OpenAI's chat models, embeddings, and reranking.
    """

    # Supported models
    SUPPORTED_MODELS = [
        "gpt-4.5-preview", "gpt-4o", "gpt-4o-mini", "o1", "o3-mini",
        # ...and more
    ]

    # Models that support embeddings
    EMBEDDING_MODELS = [
        "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"
    ]

    # Models that support reranking
    RERANKING_MODELS = [
        "text-embedding-3-small", "text-embedding-3-large"
    ]

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the OpenAI LLM.

        Args:
            model_name (str): The name of the OpenAI model to use
            **kwargs: Additional provider-specific configuration
                - api_key (str): OpenAI API key
                - api_base (str): Optional API base URL
        """
```

### Anthropic Provider

```python
class AnthropicLLM(LLM):
    """
    Implementation of the LLM interface for the Anthropic API.
    Supports Anthropic's Claude AI models.
    """

    # Supported models
    SUPPORTED_MODELS = [
        "claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219",
        # ...and more
    ]

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Anthropic LLM.

        Args:
            model_name (str): The name of the Anthropic model to use
            **kwargs: Additional provider-specific configuration
                - api_key (str): Anthropic API key
        """
```

### Google Provider

```python
class GoogleLLM(LLM):
    """
    Implementation of the LLM interface for the Google API.
    Supports Google's Gemini AI models.
    """

    # Supported models
    SUPPORTED_MODELS = [
        "gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-2.0-pro-exp-02-05",
        # ...and more
    ]

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Google LLM.

        Args:
            model_name (str): The name of the Google model to use
            **kwargs: Additional provider-specific configuration
                - api_key (str): Google API key
        """
```

### Cohere Provider

```python
class CohereLLM(LLM):
    """
    Implementation of the LLM interface for the Cohere API.
    Supports Cohere's command and chat models.
    """

    # Supported models
    SUPPORTED_MODELS = [
        "command", "command-light", "command-r", "command-r-plus",
        # ...and more
    ]

    # Models that support embeddings
    EMBEDDING_MODELS = [
        "embed-english-v3.0", "embed-multilingual-v3.0",
        "embed-english-light-v3.0", "embed-multilingual-light-v3.0"
    ]

    # Models that support reranking
    RERANKING_MODELS = [
        "rerank-english-v3.0", "rerank-multilingual-v3.0"
    ]

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Cohere LLM.

        Args:
            model_name (str): The name of the Cohere model to use
            **kwargs: Additional provider-specific configuration
                - api_key (str): Cohere API key
        """
```

## Advanced Features

### Embeddings

```python
# Initialize an LLM that supports embeddings
llm = get_llm_from_model("text-embedding-3-small")

# Check if the model supports embeddings
if llm.supports_embeddings():
    # Generate embeddings
    texts = ["This is a sample text", "Another example"]
    embeddings, usage = llm.embed(texts=texts)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding dimensions: {len(embeddings[0])}")
    print(f"Tokens used: {usage['total_tokens']}")
    print(f"Cost: ${usage['total_cost']}")
```

### Document Reranking

```python
# Initialize an LLM that supports reranking
llm = get_llm_from_model("text-embedding-3-small")  # OpenAI
# or
llm = get_llm_from_model("rerank-english-v3.0")  # Cohere

# Check if the model supports reranking
if llm.supports_reranking():
    # Sample documents
    documents = [
        "Python is a popular programming language.",
        "The Eiffel Tower is in Paris, France.",
        "Machine learning is a subset of AI.",
        "Cats are common pets worldwide."
    ]

    # Rerank documents based on a query
    query = "Which programming language is popular?"
    results = llm.rerank(
        query=query,
        documents=documents,
        top_n=2,  # Optional: limit to top N results
        return_scores=True  # Optional: include relevance scores
    )

    # Display ranked results
    for i, (doc, score) in enumerate(zip(
        results["ranked_documents"],
        results["scores"]
    )):
        print(f"{i+1}. [{score:.4f}] {doc}")

    # Usage information
    print(f"Tokens used: {results['usage']['total_tokens']}")
    print(f"Cost: ${results['usage']['total_cost']}")
```

### Streaming Responses

```python
# Initialize an LLM that supports streaming
llm = get_llm_from_model("gpt-4o")  # OpenAI
# or
llm = get_llm_from_model("claude-haiku-3.5")  # Anthropic
# or
llm = get_llm_from_model("gemini-2.0-flash")  # Google

# Define a callback function to process chunks
def process_chunk(chunk, usage_data):
    if chunk:  # Empty chunk signals completion
        print(chunk, end="", flush=True)
    else:
        print("\nStream completed")
        print(f"Total tokens: {usage_data['total_tokens']}")
        print(f"Cost: ${usage_data.get('total_cost', 0)}")

# Stream a response
for chunk, usage in llm.stream_generate(
    event_id="my_stream",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "Explain quantum computing briefly."}
    ],
    max_tokens=300,
    temp=0.7,
    callback=process_chunk  # Optional: Process chunks as they arrive
):
    # You can also process chunks here if you prefer
    # This loop will yield each chunk as it's received
    pass  # The callback handles the chunks, so nothing to do here
```

### Function Calling

```python
# Define a function that the LLM can use
def get_weather(location: str, unit: str = "celsius"):
    """
    Get the current weather for a location.

    Args:
        location (str): The city and state, e.g. "San Francisco, CA"
        unit (str): The temperature unit, either "celsius" or "fahrenheit"

    Returns:
        Dict[str, Any]: The current weather
    """
    # In a real implementation, you would call a weather API here
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny"
    }

# Initialize an LLM
llm = get_llm_from_model("gpt-4o")

# Generate a response with function calling
response, usage, updated_messages = llm.generate(
    event_id="function_call_example",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "What's the weather like in San Francisco?"}
    ],
    functions=[get_weather]
)

print(response)
print(f"Total tokens: {usage['total_tokens']}")
print(f"Cost: ${usage['total_cost']}")
```

## Error Handling

### Exceptions

```python
from lluminary.exceptions import LLMMistake

try:
    response, usage, updated_messages = llm.generate(
        event_id="error_handling_example",
        system_prompt="You are a helpful assistant.",
        messages=[
            {"message_type": "human", "message": "What is 2 + 2?"}
        ]
    )
except LLMMistake as e:
    # Handle specific LLM errors
    print(f"LLM made a mistake: {str(e)}")
except Exception as e:
    # Handle other errors
    print(f"Error: {str(e)}")
```

### Retry Logic

```python
# The retry logic is built into the generate method
response, usage, updated_messages = llm.generate(
    event_id="retry_example",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "What is the capital of France?"}
    ],
    retry_limit=3  # Will retry up to 3 times if the generation fails
)
```

## Related Documentation

- [ARCHITECTURE](./ARCHITECTURE.md) - Detailed architecture of the LLuMinary library
- [TUTORIALS](./TUTORIALS.md) - Step-by-step tutorials for common use cases
- [TEST_COVERAGE](./TEST_COVERAGE.md) - Current test coverage status
- [ERROR_HANDLING](./development/ERROR_HANDLING.md) - Comprehensive error handling guidelines
