Google Provider
==============

The Google provider in LLuMinary allows you to interact with Google's Gemini models, which offer strong multimodal capabilities, reasoning, and tool use functionality.

Setup and Authentication
-----------------------

To use the Google provider, you need an API key from Google AI. Here's how to set it up:

**Method 1: Direct initialization with API key**

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="google",
        model="gemini-2.0-pro-exp-02-05",
        api_key="your-google-api-key"
    )

**Method 2: Using environment variables**

Set the ``GOOGLE_API_KEY`` environment variable:

.. code-block:: bash

    export GOOGLE_API_KEY=your-google-api-key

Then initialize without explicitly providing the key:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="google",
        model="gemini-2.0-pro-exp-02-05"
    )

**Method 3: Using Application Default Credentials (ADC)**

For Google Cloud users, you can use Application Default Credentials:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="google",
        model="gemini-2.0-pro-exp-02-05",
        use_app_default_credentials=True,
        project_id="your-gcp-project-id"  # Optional if set in environment
    )

Supported Models
--------------

The Google provider supports these Gemini models:

* **gemini-2.0-flash** - Fast, cost-effective model with multimodal capabilities
* **gemini-2.0-flash-lite-preview-02-05** - Even lighter, faster model for basic tasks
* **gemini-2.0-pro-exp-02-05** - More capable model for complex tasks
* **gemini-2.0-flash-thinking-exp-01-21** - Model with thinking/reasoning capabilities

All models support a 128,000 token context window, image inputs, and tool use.

For the complete list with capabilities and pricing, see the :ref:`models_reference`.

Basic Usage
----------

**Text Generation**

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="google", model="gemini-2.0-flash")

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

    llm = LLuMinary(provider="google", model="gemini-2.0-flash")

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
    ]

    llm = LLuMinary(provider="google", model="gemini-2.0-flash")

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
            args = json.loads(arguments)
            weather_data = {"temperature": 72, "condition": "sunny"}

            # Send function result back
            messages = [
                {"role": "user", "content": "What's the weather like in San Francisco?"},
                {"role": "model", "content": None, "tool_calls": [tool_call]},
                {
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(weather_data)
                }
            ]
            final_response = llm.generate(messages)
            print(final_response.content)

**Embeddings**

Generate embeddings for text:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="google")

    # Generate embeddings for a single text
    embedding = llm.embed(
        "The quick brown fox jumps over the lazy dog",
        model="text-embedding-004"  # Use Google's embedding model
    )
    print(f"Embedding dimension: {len(embedding)}")

    # Process batch of texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Natural language processing is powerful"
    ]
    embeddings = llm.embed_batch(texts, model="text-embedding-004")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Dimension of each embedding: {len(embeddings[0])}")

**Image Generation**

Generate images using Imagen models:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="google")

    # Generate an image
    result = llm.generate_image(
        "A serene landscape with mountains and a lake at sunset",
        model="imagen-3.0",
        size="1024x1024"  # Options: "256x256", "512x512", "1024x1024", etc.
    )

    # Print the URL to the generated image
    print(result.data[0].url)

**Thinking/Reasoning**

With the thinking-enabled models, you can leverage advanced reasoning capabilities:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="google",
        model="gemini-2.0-flash-thinking-exp-01-21"
    )

    # Generate response with thinking
    response = llm.generate(
        "Solve this complex problem: What is the largest prime factor of 600851475143?",
        reasoning_effort=0.8  # Control the effort spent on reasoning
    )
    print(response.content)

Provider-Specific Parameters
--------------------------

The Google provider supports these additional parameters:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="google",
        model="gemini-2.0-flash",
        # Google-specific parameters
        temperature=0.7,            # Controls randomness (0.0 to 1.0)
        top_p=0.9,                  # Controls diversity via nucleus sampling
        top_k=40,                   # Limits vocabulary options per token
        max_output_tokens=1000,     # Maximum tokens to generate
        safety_settings=[           # Content safety thresholds
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ],
        reasoning_effort=0.5,       # For thinking models
        candidate_count=1,          # Number of responses to generate
        stop_sequences=["STOP"],    # Custom sequences that stop generation
        timeout=60                  # Request timeout in seconds
    )

Error Handling
------------

LLuMinary implements comprehensive error handling for Google:

.. code-block:: python

    from lluminary import LLuMinary
    from lluminary.exceptions import (
        GoogleAuthenticationError,
        GoogleRateLimitError,
        GoogleAPIError,
        GoogleTimeoutError,
        GoogleSafetyError
    )

    try:
        llm = LLuMinary(provider="google", model="gemini-2.0-flash")
        response = llm.generate("Hello, world!")
    except GoogleAuthenticationError as e:
        print(f"Authentication error: {e}")
    except GoogleRateLimitError as e:
        print(f"Rate limit exceeded: {e}")
    except GoogleTimeoutError as e:
        print(f"Request timed out: {e}")
    except GoogleSafetyError as e:
        print(f"Content safety error: {e}")
    except GoogleAPIError as e:
        print(f"API error: {e}")

Models Reference
--------------

For detailed information about Google models, their capabilities, and pricing, see the :doc:`/models_reference` page.
