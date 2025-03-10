# LLUMINARY TUTORIALS

## Overview

This document provides a comprehensive series of tutorials to help you get started with the LLuMinary library and explore its features. Each tutorial includes sample code and explanations to demonstrate key capabilities of the library.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
- [Working with Different Providers](#working-with-different-providers)
  - [OpenAI Integration](#openai-integration)
  - [Anthropic Integration](#anthropic-integration)
  - [Google Integration](#google-integration)
  - [Cohere Integration](#cohere-integration)
- [Advanced Features](#advanced-features)
  - [Embedding Tutorial](#embedding-tutorial)
  - [Document Reranking Tutorial](#document-reranking-tutorial)
  - [Streaming Responses Tutorial](#streaming-responses-tutorial)
  - [Function Calling Tutorial](#function-calling-tutorial)
- [Building Applications](#building-applications)
  - [Streamlit Chat Application](#streamlit-chat-application)
- [Building Custom Integrations](#building-custom-integrations)
  - [Creating a Custom Provider](#creating-a-custom-provider)
  - [Registering Custom Models](#registering-custom-models)

## Getting Started

### Installation

You can install the LLM Handler library using pip:

```bash
pip install llm-handler
```

For development or to include optional dependencies:

```bash
# Install with all optional dependencies
pip install llm-handler[all]

# Install with specific provider dependencies
pip install llm-handler[openai,anthropic]
```

### Basic Usage

Here's a simple example of how to use the LLM Handler library:

```python
from lluminary import get_llm_from_model

# Initialize an LLM with your API key
llm = get_llm_from_model("gpt-4o", api_key="your-openai-api-key")

# Generate a response
response, usage, _ = llm.generate(
    event_id="getting-started",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "What is the capital of France?"}
    ]
)

# Print the response and usage information
print(f"Response: {response}")
print(f"Tokens used: {usage['total_tokens']}")
print(f"Cost: ${usage['total_cost']}")
```

## Working with Different Providers

### OpenAI Integration

To use OpenAI models:

```python
from lluminary import get_llm_from_model

# Initialize an OpenAI LLM
llm = get_llm_from_model("gpt-4o", api_key="your-openai-api-key")

# Or use environment variables
# export OPENAI_API_KEY=your-key
llm = get_llm_from_model("gpt-4o")  # Will use OPENAI_API_KEY from environment

# Generate a response
response, usage, _ = llm.generate(
    event_id="openai-example",
    system_prompt="You are a knowledgeable assistant.",
    messages=[
        {"message_type": "human", "message": "Explain quantum computing briefly."}
    ],
    max_tokens=500,
    temp=0.7
)

print(response)
```

### Anthropic Integration

To use Anthropic's Claude models:

```python
from lluminary import get_llm_from_model

# Initialize an Anthropic LLM
llm = get_llm_from_model("claude-haiku-3.5", api_key="your-anthropic-api-key")

# Generate a response
response, usage, _ = llm.generate(
    event_id="anthropic-example",
    system_prompt="You are Claude, a friendly and concise assistant.",
    messages=[
        {"message_type": "human", "message": "Write a short poem about AI."}
    ]
)

print(response)
```

### Google Integration

To use Google's Gemini models:

```python
from lluminary import get_llm_from_model

# Initialize a Google LLM
llm = get_llm_from_model("gemini-2.0-flash", api_key="your-google-api-key")

# Generate a response
response, usage, _ = llm.generate(
    event_id="google-example",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "Explain the benefits of renewable energy."}
    ]
)

print(response)
```

### Cohere Integration

To use Cohere models:

```python
from lluminary import get_llm_from_model

# Initialize a Cohere LLM
llm = get_llm_from_model("cohere-command", api_key="your-cohere-api-key")

# Generate a response
response, usage, _ = llm.generate(
    event_id="cohere-example",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "Summarize the importance of clean water."}
    ]
)

print(response)
```

## Advanced Features

### Embedding Tutorial

Embeddings convert text into numerical vectors that capture semantic meaning. Here's how to generate embeddings:

```python
from lluminary import get_llm_from_model

# Initialize an LLM that supports embeddings
llm = get_llm_from_model("text-embedding-3-small", api_key="your-openai-api-key")

# Check if the model supports embeddings
if llm.supports_embeddings():
    # Prepare texts to embed
    texts = [
        "This is a sample text about artificial intelligence.",
        "Embeddings are useful for semantic search and similarity matching.",
        "Vector representations help machines understand text meaning."
    ]

    # Generate embeddings
    embeddings, usage = llm.embed(texts=texts)

    # Print information about the embeddings
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0])} dimensions")
    print(f"Tokens used: {usage['total_tokens']}")
    print(f"Cost: ${usage['total_cost']}")

    # Use embeddings for similarity comparison
    from numpy import dot
    from numpy.linalg import norm

    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    # Compare the first and second embeddings
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between first and second text: {similarity:.4f}")
```

### Document Reranking Tutorial

Document reranking helps sort a list of documents by their relevance to a query:

```python
from lluminary import get_llm_from_model

# Initialize an LLM that supports reranking
llm = get_llm_from_model("text-embedding-3-small", api_key="your-openai-api-key")
# Or for Cohere's dedicated reranking model:
# llm = get_llm_from_model("rerank-english-v3.0", api_key="your-cohere-api-key")

# Check if the model supports reranking
if llm.supports_reranking():
    # Sample documents
    documents = [
        "Python is a popular programming language known for its readability and simplicity.",
        "The Eiffel Tower is located in Paris, France, and is a famous landmark.",
        "Machine learning algorithms learn patterns from data to make predictions.",
        "Renewable energy sources include solar, wind, and hydroelectric power.",
        "The human brain contains approximately 86 billion neurons.",
        "Python was created by Guido van Rossum in the late 1980s.",
        "Neural networks are inspired by the structure of the human brain.",
        "Paris is known as the City of Light and is famous for its art and culture."
    ]

    # Query for reranking
    query = "Tell me about programming languages"

    # Rerank the documents
    results = llm.rerank(
        query=query,
        documents=documents,
        top_n=3,  # Return top 3 results
        return_scores=True  # Include relevance scores
    )

    # Display ranked results
    print(f"Query: {query}\n")
    print("Top 3 most relevant documents:")
    for i, (doc, score) in enumerate(zip(
        results["ranked_documents"],
        results["scores"]
    )):
        print(f"{i+1}. [{score:.4f}] {doc}")

    # Usage information
    print(f"\nTokens used: {results['usage']['total_tokens']}")
    print(f"Cost: ${results['usage']['total_cost']}")

    # Try a different query
    query2 = "What is Paris famous for?"
    results2 = llm.rerank(query=query2, documents=documents, top_n=3)

    print(f"\nQuery: {query2}\n")
    print("Top 3 most relevant documents:")
    for i, (doc, score) in enumerate(zip(
        results2["ranked_documents"],
        results2["scores"]
    )):
        print(f"{i+1}. [{score:.4f}] {doc}")
```

### Streaming Responses Tutorial

Streaming allows you to receive responses in real-time, as they're generated:

```python
import time
from lluminary import get_llm_from_model

# Initialize an LLM that supports streaming
llm = get_llm_from_model("gpt-4o", api_key="your-openai-api-key")
# Alternative providers:
# llm = get_llm_from_model("claude-haiku-3.5", api_key="your-anthropic-api-key")
# llm = get_llm_from_model("gemini-2.0-flash", api_key="your-google-api-key")

# Define a callback function to process chunks
def process_chunk(chunk, usage_data):
    if chunk:  # Non-empty chunk
        print(chunk, end="", flush=True)
        time.sleep(0.01)  # Slight delay to simulate typing
    else:  # Empty chunk signals completion
        print("\n\n--- Stream Completed ---")
        print(f"Total tokens: {usage_data['total_tokens']}")
        print(f"Total cost: ${usage_data['total_cost']:.6f}")

# Stream a response
print("Streaming response:\n")
start_time = time.time()

for chunk, usage in llm.stream_generate(
    event_id="streaming-example",
    system_prompt="You are a helpful assistant who responds concisely.",
    messages=[
        {"message_type": "human", "message": "Explain how streaming works in LLM APIs"}
    ],
    max_tokens=300,
    temp=0.7,
    callback=process_chunk  # Process chunks as they arrive
):
    # The callback handles the output, we don't need to do anything here
    pass

# Calculate and print the elapsed time
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Example of using streaming without a callback
print("\nSimple streaming (without callback):\n")

poem_messages = [
    {"message_type": "human", "message": "Write a four-line poem about technology"}
]

for chunk, _ in llm.stream_generate(
    event_id="simple-streaming",
    system_prompt="You are a creative poet.",
    messages=poem_messages,
    max_tokens=100
):
    if chunk:
        print(chunk, end="", flush=True)
```

### Function Calling Tutorial

Function calling allows the LLM to use tools and external functions:

```python
from lluminary import get_llm_from_model
import json

# Define some functions that the LLM can use
def get_weather(location: str, unit: str = "celsius"):
    """
    Get the current weather for a location.

    Args:
        location (str): The city and state, e.g. "San Francisco, CA"
        unit (str): The temperature unit, either "celsius" or "fahrenheit"

    Returns:
        Dict[str, Any]: The current weather
    """
    # This is a mock implementation
    weather_data = {
        "San Francisco, CA": {"temp": 18, "condition": "Foggy"},
        "New York, NY": {"temp": 22, "condition": "Sunny"},
        "London, UK": {"temp": 15, "condition": "Rainy"},
    }

    data = weather_data.get(location, {"temp": 20, "condition": "Clear"})
    temp = data["temp"]
    if unit == "fahrenheit":
        temp = (temp * 9/5) + 32

    return {
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": data["condition"]
    }

def calculate_mortgage(principal: float, interest_rate: float, years: int):
    """
    Calculate monthly mortgage payment.

    Args:
        principal (float): Loan amount
        interest_rate (float): Annual interest rate (as a decimal, e.g., 0.05 for 5%)
        years (int): Loan term in years

    Returns:
        Dict[str, Any]: Monthly payment information
    """
    monthly_rate = interest_rate / 12
    n_payments = years * 12

    if monthly_rate == 0:
        monthly_payment = principal / n_payments
    else:
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)

    total_paid = monthly_payment * n_payments
    total_interest = total_paid - principal

    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_paid": round(total_paid, 2),
        "total_interest": round(total_interest, 2)
    }

# Initialize an LLM
llm = get_llm_from_model("gpt-4o", api_key="your-openai-api-key")

# Generate a response with function calling
messages = [
    {"message_type": "human", "message": "What's the weather like in San Francisco and what's the monthly payment on a $500,000 mortgage at 4.5% interest for 30 years?"}
]

response, usage, updated_messages = llm.generate(
    event_id="function-calling-example",
    system_prompt="You are a helpful assistant that can access external tools.",
    messages=messages,
    functions=[get_weather, calculate_mortgage],
    max_tokens=500,
    temp=0.2
)

print("Response:")
print(response)
print(f"\nTokens used: {usage['total_tokens']}")
print(f"Cost: ${usage['total_cost']}")

# Examine the tool calls made
print("\nFunction calls made:")
for message in updated_messages:
    if message.get("message_type") == "ai" and "tool_use" in message:
        tool_use = message["tool_use"]
        print(f"- Function: {tool_use['name']}")
        print(f"  Arguments: {tool_use['arguments']}")
        print(f"  Response: {message.get('tool_result', {}).get('result')}")
```

## Building Applications

### Streamlit Chat Application

This tutorial demonstrates how to build a chat application using Streamlit and the llm-handler library. The application allows users to interact with various LLM providers through a unified interface.

#### Prerequisites

- Python 3.10+
- Streamlit
- llm-handler

```bash
pip install streamlit llm-handler
```

#### Complete Application Code

```python
import streamlit as st
import os
from typing import List, Dict, Any, Optional
from lluminary import get_llm_from_model

# Set page configuration
st.set_page_config(
    page_title="LLM Chat with llm-handler",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state.llm = None

# App title and description
st.title("🤖 LLM Chat Application")
st.markdown("""
This application demonstrates how to use the llm-handler library to create a chat interface
with various LLM providers including OpenAI, Anthropic, Google, and more.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Model selection
    model_options = {
        "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "Anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-haiku-3.5"],
        "Google": ["gemini-2.0-pro", "gemini-2.0-flash"],
        "Cohere": ["command", "command-light"]
    }

    provider = st.selectbox("Provider", list(model_options.keys()))
    model = st.selectbox("Model", model_options[provider])

    # API key input
    api_key = st.text_input("API Key", type="password")

    # System prompt
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, friendly AI assistant. Provide clear and concise responses.",
        height=150
    )

    # Temperature slider
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    # Max tokens slider
    max_tokens = st.slider("Max Tokens", min_value=100, max_value=4000, value=1000, step=100)

    # Initialize or update LLM when settings change
    if st.button("Apply Settings") or st.session_state.llm is None:
        if api_key:
            with st.spinner("Initializing LLM..."):
                try:
                    st.session_state.llm = get_llm_from_model(model, api_key=api_key)
                    st.success(f"Successfully initialized {model}")
                except Exception as e:
                    st.error(f"Error initializing LLM: {str(e)}")
        else:
            st.warning("Please enter an API key")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if LLM is initialized
    if st.session_state.llm is None:
        with st.chat_message("assistant"):
            st.error("Please configure and initialize the LLM in the sidebar first.")
    else:
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Format messages for the LLM
            formatted_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    formatted_messages.append({
                        "message_type": "human",
                        "message": msg["content"],
                        "image_paths": [],
                        "image_urls": []
                    })
                elif msg["role"] == "assistant":
                    formatted_messages.append({
                        "message_type": "ai",
                        "message": msg["content"]
                    })

            # Stream the response
            try:
                # Define callback for streaming
                def process_chunk(chunk, usage_data):
                    nonlocal full_response
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

                # Stream generate
                for _, _ in st.session_state.llm.stream_generate(
                    event_id="streamlit_chat",
                    system_prompt=system_prompt,
                    messages=formatted_messages,
                    max_tokens=max_tokens,
                    temp=temperature,
                    callback=process_chunk
                ):
                    pass

                # Display final response
                message_placeholder.markdown(full_response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Add a button to clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.experimental_rerun()

# Display usage information in the sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses the [llm-handler](https://github.com/yourusername/llm-handler) library
    to provide a unified interface for interacting with various LLM providers.

    Features:
    - Support for multiple providers (OpenAI, Anthropic, Google, Cohere)
    - Streaming responses
    - Configurable parameters (temperature, max tokens)
    - Custom system prompts
    """)
```

#### How to Run the Application

Save the code to a file named `app.py` and run it with:

```bash
streamlit run app.py
```

#### Key Components Explained

1. **Configuration in Sidebar**:
   - Provider and model selection
   - API key input
   - System prompt customization
   - Parameter adjustment (temperature, max tokens)

2. **Message Formatting**:
   - Converts Streamlit's message format to llm-handler's expected format
   - Handles both user and assistant messages

3. **Streaming Implementation**:
   - Uses `stream_generate()` with a callback function
   - Updates the UI in real-time as chunks are received
   - Shows a typing indicator (▌) during generation

4. **Session State Management**:
   - Maintains chat history between interactions
   - Stores the LLM instance for reuse

5. **Error Handling**:
   - Gracefully handles initialization errors
   - Provides clear error messages during generation

This application demonstrates how to leverage the llm-handler library's unified interface to create a flexible chat application that works with multiple LLM providers through a consistent API.

## Building Custom Integrations

### Creating a Custom Provider

You can extend the LLM Handler library with your own custom providers:

```python
from typing import Any, Dict, List, Optional, Tuple
from lluminary.models.base import LLM
from lluminary.models.router import register_provider

class CustomLLM(LLM):
    """
    Custom LLM provider implementation.
    """

    # Define supported models
    SUPPORTED_MODELS = ["custom-model-1", "custom-model-2"]

    # Define context window sizes
    CONTEXT_WINDOW = {
        "custom-model-1": 8192,
        "custom-model-2": 16384
    }

    # Define token costs
    COST_PER_MODEL = {
        "custom-model-1": {
            "read_token": 0.0000005,  # $0.50 per million tokens
            "write_token": 0.0000015   # $1.50 per million tokens
        },
        "custom-model-2": {
            "read_token": 0.0000010,  # $1.00 per million tokens
            "write_token": 0.0000020   # $2.00 per million tokens
        }
    }

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the Custom LLM.

        Args:
            model_name (str): The name of the model to use
            **kwargs: Additional configuration parameters
                - api_key (str): API key for the custom service
                - api_url (str, optional): API URL override
        """
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get("api_key")
        self.api_url = kwargs.get("api_url", "https://api.custom-llm-provider.com")

    def generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        functions: List = None,
        retry_limit: int = 3
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """Generate a response from the custom LLM provider."""
        # Implement your custom generation logic here
        # ...

        # Mock implementation for example purposes
        response = "This is a response from the custom LLM provider."
        usage = {
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "total_cost": 0.0001
        }

        updated_messages = messages + [{
            "message_type": "ai",
            "message": response
        }]

        return response, usage, updated_messages

# Register the custom provider
register_provider("custom", CustomLLM)

# Now you can use your custom provider
llm = get_llm_from_model("custom-model-1", api_key="your-custom-api-key")
response, usage, _ = llm.generate(
    event_id="custom-example",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "Hello, custom LLM!"}
    ]
)
```

### Registering Custom Models

You can register custom models with existing providers:

```python
from lluminary.models.router import register_model

# Register a custom OpenAI model
register_model(
    friendly_name="my-gpt4-turbo",
    provider_name="openai",
    model_id="gpt-4-turbo-2024-04-09"  # The actual model ID used by the provider
)

# Register a custom Anthropic model
register_model(
    friendly_name="my-claude-3",
    provider_name="anthropic",
    model_id="claude-3-opus-20240229"
)

# Now you can use these custom models
llm = get_llm_from_model("my-gpt4-turbo", api_key="your-openai-api-key")
```

## Related Documentation

- [API_REFERENCE](./API_REFERENCE.md) - Complete API reference for all components
- [ARCHITECTURE](./ARCHITECTURE.md) - System architecture and component relationships
- [TEST_COVERAGE](./TEST_COVERAGE.md) - Current test coverage and implementation guides
- [MODELS](./development/MODELS.md) - Comprehensive list of supported models
- [ERROR_HANDLING](./development/ERROR_HANDLING.md) - Error handling implementation details
