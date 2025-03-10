AWS Bedrock Provider
=================

The AWS Bedrock provider in LLuMinary allows you to interact with a variety of foundation models through Amazon's Bedrock service, including models from Anthropic, AI21, Cohere, Meta, and Amazon's own Titan models.

Setup and Authentication
-----------------------

To use the AWS Bedrock provider, you need AWS credentials with appropriate permissions to access Bedrock services. Here's how to set it up:

**Method 1: Using AWS credentials**

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="bedrock",
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_access_key_id="your-aws-access-key",
        aws_secret_access_key="your-aws-secret-key",
        aws_region="us-west-2"  # Specify the AWS region where Bedrock is deployed
    )

**Method 2: Using environment variables**

Set the standard AWS environment variables:

.. code-block:: bash

    export AWS_ACCESS_KEY_ID=your-aws-access-key
    export AWS_SECRET_ACCESS_KEY=your-aws-secret-key
    export AWS_REGION=us-west-2

Then initialize without explicitly providing credentials:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="bedrock",
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )

**Method 3: Using AWS Profiles**

If you have configured profiles with the AWS CLI, you can use them:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="bedrock",
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_profile="your-profile-name"
    )

Supported Models
--------------

The AWS Bedrock provider supports models from multiple AI providers:

**Anthropic Claude Models**:
* **us.anthropic.claude-3-5-haiku-20241022-v1:0** - Fast, cost-effective Claude model
* **us.anthropic.claude-3-5-sonnet-20240620-v1:0** - Balanced Claude model
* **us.anthropic.claude-3-5-sonnet-20241022-v2:0** - Updated sonnet model
* **us.anthropic.claude-3-7-sonnet-20250219-v1:0** - Latest Claude model with thinking capabilities

**Amazon Titan Models**:
* **amazon.titan-text-express-v1** - Amazon's general-purpose text model
* **amazon.titan-text-premier-v1** - Amazon's high-performance text model
* **amazon.titan-embed-text-v1** - Text embedding model

**Other Provider Models**:
* **meta.llama3-70b-instruct-v1** - Meta's Llama 3 70B model
* **cohere.command-r-plus-v1** - Cohere's Command R Plus model
* **ai21.j2-ultra-v1** - AI21's Jurassic-2 Ultra model

For the complete list with capabilities and pricing, see the :ref:`models_reference`.

Basic Usage
----------

**Text Generation**

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="bedrock", model="us.anthropic.claude-3-5-sonnet-20240620-v1:0")

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

**Image Input (with supported models)**

Process images as part of your prompt with models that support image input, like Claude:

.. code-block:: python

    from lluminary import LLuMinary
    from pathlib import Path

    llm = LLuMinary(provider="bedrock", model="us.anthropic.claude-3-5-sonnet-20240620-v1:0")

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

**Tool Calling (with supported models)**

Define and use tools that the model can call:

.. code-block:: python

    from lluminary import LLuMinary
    import json

    # Define tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "input_schema": {  # For Claude models
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

    llm = LLuMinary(provider="bedrock", model="us.anthropic.claude-3-5-sonnet-20240620-v1:0")

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

Generate embeddings for text using Bedrock embedding models:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(provider="bedrock")

    # Generate embeddings for a single text
    embedding = llm.embed(
        "The quick brown fox jumps over the lazy dog",
        model="amazon.titan-embed-text-v1"  # Specify Bedrock's embedding model
    )
    print(f"Embedding dimension: {len(embedding)}")

    # Process batch of texts
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Natural language processing is powerful"
    ]
    embeddings = llm.embed_batch(texts, model="amazon.titan-embed-text-v1")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Dimension of each embedding: {len(embeddings[0])}")

**Thinking Budget (with Claude 3.7 models)**

For Claude 3.7 models via Bedrock, you can use the thinking budget feature:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="bedrock",
        model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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

The Bedrock provider supports various parameters, which are model-specific and vary by the underlying model provider:

.. code-block:: python

    from lluminary import LLuMinary

    # For Anthropic Claude models on Bedrock
    llm = LLuMinary(
        provider="bedrock",
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        # Claude-specific parameters
        temperature=0.7,          # Controls randomness (0.0 to 1.0)
        top_p=0.9,                # Controls diversity via nucleus sampling
        max_tokens=1000,          # Maximum tokens to generate
        thinking_budget=0.5,      # Thinking budget ratio (for Claude 3.7 models)
        stop_sequences=["STOP"],  # Custom sequences that stop generation

        # AWS-specific parameters
        aws_region="us-west-2",            # AWS region
        aws_profile="default",             # AWS CLI profile
        aws_access_key_id="your-key-id",   # AWS access key
        aws_secret_access_key="your-secret-key",  # AWS secret key
        max_retries=3,                     # Maximum retry attempts
        retry_mode="standard",             # AWS retry mode
        timeout=60                         # Request timeout in seconds
    )

    # For Amazon Titan models
    llm = LLuMinary(
        provider="bedrock",
        model="amazon.titan-text-express-v1",
        # Titan-specific parameters
        temperature=0.7,      # Controls randomness
        top_p=0.9,            # Controls diversity
        max_tokens=1000,      # Maximum tokens to generate
        stop_sequences=["STOP"]  # Custom sequences that stop generation
    )

Error Handling
------------

LLuMinary implements comprehensive error handling for AWS Bedrock:

.. code-block:: python

    from lluminary import LLuMinary
    from lluminary.exceptions import (
        BedrockAuthenticationError,
        BedrockRateLimitError,
        BedrockAPIError,
        BedrockTimeoutError,
        BedrockResourceNotFoundError
    )

    try:
        llm = LLuMinary(provider="bedrock", model="us.anthropic.claude-3-5-sonnet-20240620-v1:0")
        response = llm.generate("Hello, world!")
    except BedrockAuthenticationError as e:
        print(f"Authentication error: {e}")
    except BedrockRateLimitError as e:
        print(f"Rate limit exceeded: {e}")
    except BedrockTimeoutError as e:
        print(f"Request timed out: {e}")
    except BedrockResourceNotFoundError as e:
        print(f"Resource not found: {e}")
    except BedrockAPIError as e:
        print(f"API error: {e}")

AWS Specific Features
------------------

**Automatic Retries**

The Bedrock provider automatically implements AWS SDK retry behavior:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="bedrock",
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_retries=5,                 # Maximum retry attempts
        retry_mode="adaptive",         # Options: "legacy", "standard", "adaptive"
        connect_timeout=5,             # Connection timeout in seconds
        read_timeout=60                # Read timeout in seconds
    )

**Regional Endpoint Configuration**

Specify the AWS region for Bedrock:

.. code-block:: python

    from lluminary import LLuMinary

    llm = LLuMinary(
        provider="bedrock",
        model="us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        aws_region="us-east-1"  # Specify the AWS region where Bedrock is available
    )

Models Reference
--------------

For detailed information about AWS Bedrock models, their capabilities, and pricing, see the :doc:`/models_reference` page.
