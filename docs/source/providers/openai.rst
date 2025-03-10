OpenAI Provider
==============

The OpenAI provider in LLuMinary allows you to interact with OpenAI's models, including GPT-4.5, GPT-4o, GPT-4o-mini and newer models with advanced capabilities like tool calling, image generation, and embeddings.

Setup and Authentication
-----------------------

To use the OpenAI provider, you need an API key from OpenAI. There are several ways to provide this:

**Method 1: Direct initialization with API key**

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="openai",
        model="gpt-4o",
        api_key="your-openai-api-key"
    )

**Method 2: Using environment variables**

Set the ``OPENAI_API_KEY`` environment variable:

.. code-block:: bash

    export OPENAI_API_KEY=your-openai-api-key

Then initialize without explicitly providing the key:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="openai",
        model="gpt-4o"
    )

**Method 3: Organization ID (optional)**

If you belong to multiple organizations, you can specify which one to use:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="openai",
        model="gpt-4o",
        api_key="your-openai-api-key",
        organization="org-..."
    )

Supported Models
--------------

The OpenAI provider supports these model families:

* **gpt-4.5-preview** - Latest GPT-4.5 preview with 128k context
* **gpt-4o** - Standard GPT-4o with 128k context
* **gpt-4o-mini** - Smaller, faster GPT-4o with 128k context
* **o1** - Advanced model with reasoning capabilities and 200k context
* **o3-mini** - Compact version with reasoning capabilities and 200k context

For the complete list with capabilities and pricing, see the :ref:`models_reference`.

Basic Usage
----------

**Text Generation**

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="openai", model="gpt-4o")

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

    llm = LLuMinary(provider="openai", model="gpt-4o")

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

**Tool Calling / Function Calling**

Define and use tools (functions) that the model can call:

.. code-block:: python

    from lluminary import LLuMinary

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
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
        }
    ]

    llm = LLuMinary(provider="openai", model="gpt-4o")

    # First message with tool definition
    response = llm.generate(
        "What's the weather like in San Francisco?",
        tools=tools
    )

    # Handle tool calls
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        function_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]

        # Simulate function execution
        if function_name == "get_weather":
            import json
            args = json.loads(arguments)
            weather_data = {"temperature": 72, "condition": "sunny"}

            # Send function result back
            messages = [
                {"role": "user", "content": "What's the weather like in San Francisco?"},
                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": json.dumps(weather_data)
                }
            ]
            final_response = llm.generate(messages)
            print(final_response.content)

**Image Generation**

Generate images using DALL-E models:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="openai")

    # Generate an image
    result = llm.generate_image(
        "A serene landscape with mountains and a lake at sunset",
        model="dall-e-3",  # or "dall-e-2"
        size="1024x1024",  # Options: "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
        quality="standard",  # Options: "standard", "hd"
        style="natural",  # Options: "natural", "vivid"
        response_format="url"  # Options: "url", "b64_json"
    )

    # Print the URL to the generated image
    print(result.data[0].url)

**Embeddings**

Generate embeddings for text:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="openai")

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

**Reranking**

Rerank documents based on relevance to a query:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="openai")

    query = "quantum computing applications"
    documents = [
        "Quantum computing is used in cryptography and security systems.",
        "Machine learning can be enhanced by quantum algorithms.",
        "Cloud computing services are widely available today.",
        "Quantum supremacy was demonstrated in 2019.",
        "Mobile applications use cloud computing infrastructure."
    ]

    # Rerank documents
    results = llm.rerank(query, documents)

    # Print ranked results
    for result in results:
        print(f"Score: {result.score:.4f} - {result.document}")

Provider-Specific Parameters
--------------------------

The OpenAI provider supports these additional parameters:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="openai",
        model="gpt-4o",
        # OpenAI-specific parameters
        temperature=0.7,  # Controls randomness (0.0 to 2.0)
        top_p=1.0,        # Controls diversity of options considered
        max_tokens=100,   # Maximum number of tokens to generate
        organization="org-...",  # Organization ID
        response_format={"type": "json_object"},  # Force JSON output
        seed=123,         # For reproducible outputs
        frequency_penalty=0.0,  # Penalizes repetition
        presence_penalty=0.0,   # Penalizes repetitive tokens
    )

Error Handling
------------

LLuMinary implements comprehensive error handling for OpenAI:

.. code-block:: python

    from lluminary import LLuMinary
    from lluminary.exceptions import (
        OpenAIAuthenticationError,
        OpenAIRateLimitError,
        OpenAIAPIError,
        OpenAITimeoutError
    )

    try:
        llm = LLuMinary(provider="openai", model="gpt-4o")
        response = llm.generate("Hello, world!")
    except OpenAIAuthenticationError as e:
        print(f"Authentication error: {e}")
    except OpenAIRateLimitError as e:
        print(f"Rate limit exceeded: {e}")
    except OpenAITimeoutError as e:
        print(f"Request timed out: {e}")
    except OpenAIAPIError as e:
        print(f"API error: {e}")

.. _models_reference:

Models Reference
--------------

For detailed information about OpenAI models, their capabilities, and pricing, see the :doc:`/models_reference` page.
