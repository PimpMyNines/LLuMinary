Anthropic Provider
=================

The Anthropic provider in LLuMinary allows you to interact with Anthropic's Claude models, which excel at understanding and following complex instructions, reasoning through problems, and handling nuanced conversations.

Setup and Authentication
-----------------------

To use the Anthropic provider, you need an API key from Anthropic. Here's how to set it up:

**Method 1: Direct initialization with API key**

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="anthropic",
        model="claude-3-5-sonnet-20240620-v1:0",
        api_key="your-anthropic-api-key"
    )

**Method 2: Using environment variables**

Set the ``ANTHROPIC_API_KEY`` environment variable:

.. code-block:: bash

    export ANTHROPIC_API_KEY=your-anthropic-api-key

Then initialize without explicitly providing the key:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="anthropic",
        model="claude-3-5-sonnet-20240620-v1:0"
    )

Supported Models
--------------

The Anthropic provider supports these Claude models:

* **claude-3-5-sonnet-20240620-v1:0** - High-performance model balancing intelligence and speed
* **claude-3-haiku-20240307-v1:0** - Fast, cost-effective model for simpler tasks
* **claude-3-opus-20240229-v1:0** - Most powerful model with advanced reasoning
* **claude-3-sonnet-20240229-v1:0** - Earlier version of the sonnet model
* **claude-3-7-sonnet-preview-20240626-v1:0** - Preview model with thinking/reasoning capabilities

All models support a 200,000 token context window, image inputs, and tool use.

For the complete list with capabilities and pricing, see the :ref:`models_reference`.

Basic Usage
----------

**Text Generation**

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="anthropic", model="claude-3-5-sonnet-20240620-v1:0")

    # Simple completion
    response = llm.generate("Explain quantum computing in simple terms")
    print(response.content)

    # Chat completion with messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the capital of France?"}
    ]
    response = llm.generate(messages)
    print(response.content)  # "The capital of France is Paris."

**Streaming**

.. code-block:: python

    # Stream the response
    for chunk in llm.stream("Write a short poem about AI"):
        print(chunk.content, end="", flush=True)

Advanced Features
---------------

**Image Input**

Process images as part of your prompt:

.. code-block:: python

    from lluminary import LLuMinary
    from pathlib import Path

    llm = LLuMinary(provider="anthropic", model="claude-3-5-sonnet-20240620-v1:0")

    # From a file path
    image_path = Path("path/to/image.jpg")
    response = llm.generate([
        {"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image_url": {"url": str(image_path)}}
        ]}
    ])

    # From a URL
    image_url = "https://example.com/image.jpg"
    response = llm.generate([
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image in detail."},
            {"type": "image", "image_url": {"url": image_url}}
        ]}
    ])

**Tool Calling**

Define and use tools that the model can call:

.. code-block:: python

    from lluminary import LLuMinary
    import json

    # Define tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    ]

    llm = LLuMinary(provider="anthropic", model="claude-3-5-sonnet-20240620-v1:0")

    # First message with tool definition
    response = llm.generate(
        "What's the weather like in San Francisco?",
        tools=tools
    )

    # Handle tool calls
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_input = tool_call["input"]

        # Simulate function execution
        if tool_name == "get_weather":
            # Parse input and generate result
            location = tool_input.get("location")
            weather_data = {"temperature": 72, "condition": "sunny"}

            # Send function result back
            messages = [
                {"role": "user", "content": "What's the weather like in San Francisco?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": tool_call["id"], "name": tool_name, "input": tool_input}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tool_call["id"], "content": json.dumps(weather_data)}
                ]}
            ]
            final_response = llm.generate(messages)
            print(final_response.content)

**Embeddings**

Generate embeddings for text:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="anthropic")

    # Generate embeddings for a single text
    embedding = llm.embed("The quick brown fox jumps over the lazy dog")
    print(f"Embedding dimension: {len(embedding)}")

    # Process batch of texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Natural language processing is powerful"
    ]
    embeddings = llm.embed_batch(texts)
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Dimension of each embedding: {len(embeddings[0])}")

**Thinking Budget**

Newer Claude models support a thinking budget feature that allows them to spend time thinking before responding:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="anthropic",
        model="claude-3-7-sonnet-preview-20240626-v1:0",
        thinking_budget=0.7  # Allocate 70% of tokens to thinking before responding
    )

    # Generate response with thinking
    response = llm.generate("Solve this complex problem: What is the largest prime factor of 600851475143?")
    print(response.content)

    # Access thinking token usage
    usage = response.usage
    print(f"Thinking tokens: {usage.get('thinking_tokens', 0)}")
    print(f"Response tokens: {usage.get('completion_tokens', 0)}")

Provider-Specific Parameters
--------------------------

The Anthropic provider supports these additional parameters:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="anthropic",
        model="claude-3-5-sonnet-20240620-v1:0",
        # Anthropic-specific parameters
        temperature=0.7,          # Controls randomness (0.0 to 1.0)
        top_p=0.9,                # Controls diversity via nucleus sampling
        max_tokens=1000,          # Maximum tokens to generate
        thinking_budget=0.5,      # Thinking budget ratio (for supported models)
        system="You are a helpful assistant who speaks like a pirate.",  # System prompt
        anthropic_version="2023-06-01",  # API version
        stop_sequences=["STOP"],  # Custom sequences that stop generation
        timeout=60                # Request timeout in seconds
    )

Error Handling
------------

LLuMinary implements comprehensive error handling for Anthropic:

.. code-block:: python

    from lluminary import LLuMinary
    from lluminary.exceptions import (
        AnthropicAuthenticationError,
        AnthropicRateLimitError,
        AnthropicAPIError,
        AnthropicTimeoutError
    )

    try:
        llm = LLuMinary(provider="anthropic", model="claude-3-5-sonnet-20240620-v1:0")
        response = llm.generate("Hello, world!")
    except AnthropicAuthenticationError as e:
        print(f"Authentication error: {e}")
    except AnthropicRateLimitError as e:
        print(f"Rate limit exceeded: {e}")
    except AnthropicTimeoutError as e:
        print(f"Request timed out: {e}")
    except AnthropicAPIError as e:
        print(f"API error: {e}")

Models Reference
--------------

For detailed information about Anthropic models, their capabilities, and pricing, see the :doc:`/models_reference` page.
