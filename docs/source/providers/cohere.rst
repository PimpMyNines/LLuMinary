Cohere Provider
==============

The Cohere provider in LLuMinary allows you to interact with Cohere's state-of-the-art language models, which excel at enterprise tasks, semantic search, and reranking.

Setup and Authentication
-----------------------

To use the Cohere provider, you need an API key from Cohere. Here's how to set it up:

**Method 1: Direct initialization with API key**

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="cohere",
        model="command-r",
        api_key="your-cohere-api-key"
    )

**Method 2: Using environment variables**

Set the ``COHERE_API_KEY`` environment variable:

.. code-block:: bash

    export COHERE_API_KEY=your-cohere-api-key

Then initialize without explicitly providing the key:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="cohere",
        model="command-r"
    )

Supported Models
--------------

The Cohere provider supports these model families:

* **command-r** - Latest generation Command model for powerful reasoning and instruction following
* **command-r-plus** - Enhanced Command model with extended capabilities
* **command-light** - Lightweight model for simpler tasks with faster inference
* **embed-english-v3.0** - Specialized embedding model for English text
* **embed-multilingual-v3.0** - Multilingual embedding model supporting 100+ languages
* **rerank-english-v3.0** - Specialized reranking model for search results

For the complete list with capabilities and pricing, see the :ref:`models_reference`.

Basic Usage
----------

**Text Generation**

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="cohere", model="command-r")

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
            "parameter_definitions": {
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

    llm = LLuMinary(provider="cohere", model="command-r")

    # First message with tool definition
    response = llm.generate(
        "What's the weather like in San Francisco?",
        tools=tools
    )

    # Handle tool calls
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        parameters = tool_call["parameters"]

        # Simulate function execution
        if tool_name == "get_weather":
            location = parameters.get("location")
            weather_data = {"temperature": 72, "condition": "sunny"}

            # Send function result back
            messages = [
                {"role": "user", "content": "What's the weather like in San Francisco?"},
                {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(weather_data)
                }
            ]
            final_response = llm.generate(messages)
            print(final_response.content)

**Embeddings**

Generate embeddings for text:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="cohere")

    # Generate embeddings for a single text
    embedding = llm.embed(
        "The quick brown fox jumps over the lazy dog",
        model="embed-english-v3.0"  # Specify Cohere's embedding model
    )
    print(f"Embedding dimension: {len(embedding)}")

    # Process batch of texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Natural language processing is powerful"
    ]
    embeddings = llm.embed_batch(texts, model="embed-english-v3.0")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Dimension of each embedding: {len(embeddings[0])}")

**Reranking**

Cohere excels at reranking, which helps improve search results by reordering documents based on their relevance to a query:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="cohere")

    query = "quantum computing applications"
    documents = [
        "Quantum computing is used in cryptography and security systems.",
        "Machine learning can be enhanced by quantum algorithms.",
        "Cloud computing services are widely available today.",
        "Quantum supremacy was demonstrated in 2019.",
        "Mobile applications use cloud computing infrastructure."
    ]

    # Rerank documents using Cohere's reranking model
    results = llm.rerank(
        query,
        documents,
        model="rerank-english-v3.0",
        top_n=3  # Return only top 3 results
    )

    # Print ranked results
    for result in results:
        print(f"Score: {result.score:.4f} - {result.document}")

Provider-Specific Parameters
--------------------------

The Cohere provider supports these additional parameters:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="cohere",
        model="command-r",
        # Cohere-specific parameters
        temperature=0.7,        # Controls randomness (0.0 to 2.0)
        p=0.9,                  # Controls diversity via nucleus sampling (like top_p)
        k=0,                    # Limits vocabulary options per token
        max_tokens=1000,        # Maximum tokens to generate
        frequency_penalty=0.0,  # Penalizes repetition
        presence_penalty=0.0,   # Penalizes repetitive tokens
        preamble="You are a helpful AI assistant.",  # Custom preamble for the model
        stop_sequences=["STOP"],  # Custom sequences that stop generation
        return_prompt=False,      # Whether to include prompt in the response
        logit_bias={},            # Token biasing for controlled generation
        timeout=60               # Request timeout in seconds
    )

Error Handling
------------

LLuMinary implements comprehensive error handling for Cohere:

.. code-block:: python

    from lluminary import LLuMinary
    from lluminary.exceptions import (
        CohereAuthenticationError,
        CohereRateLimitError,
        CohereAPIError,
        CohereTimeoutError
    )

    try:
        llm = LLuMinary(provider="cohere", model="command-r")
        response = llm.generate("Hello, world!")
    except CohereAuthenticationError as e:
        print(f"Authentication error: {e}")
    except CohereRateLimitError as e:
        print(f"Rate limit exceeded: {e}")
    except CohereTimeoutError as e:
        print(f"Request timed out: {e}")
    except CohereAPIError as e:
        print(f"API error: {e}")

Models Reference
--------------

For detailed information about Cohere models, their capabilities, and pricing, see the :doc:`/models_reference` page.
