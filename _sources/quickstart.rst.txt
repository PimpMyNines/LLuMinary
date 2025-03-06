Quickstart
==========

This guide will help you get started with LLuMinary, a versatile Python library for working with various LLM providers.

Basic Usage
----------

Here's a simple example of using LLuMinary with OpenAI:

.. code-block:: python

   from lluminary import get_llm_from_model

   # Initialize model (automatically selects the right provider)
   llm = get_llm_from_model("gpt-4o", api_key="your-api-key")

   # Generate a response
   response, usage, _ = llm.generate(
       event_id="quick-start",
       system_prompt="You are a helpful assistant.",
       messages=[
           {"message_type": "human", "message": "What is machine learning?"}
       ]
   )

   print(response)
   print(f"Tokens used: {usage['total_tokens']}")
   print(f"Cost: ${usage['total_cost']}")

Using Different Providers
------------------------

LLuMinary supports multiple providers through the same interface:

.. code-block:: python

   # OpenAI
   llm = get_llm_from_model("gpt-4o")

   # Anthropic
   llm = get_llm_from_model("claude-haiku-3.5")

   # Google
   llm = get_llm_from_model("gemini-2.0-flash")

   # Cohere
   llm = get_llm_from_model("cohere-command")

Streaming Responses
-----------------

Stream responses from LLMs for real-time output:

.. code-block:: python

   from lluminary import get_llm_from_model

   # Initialize a streaming-capable model
   llm = get_llm_from_model("gpt-4o")

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
       pass  # The callback handles the chunks

Embeddings
---------

Generate embeddings for text with supported models:

.. code-block:: python

   from lluminary import get_llm_from_model

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

Using the Client Class
--------------------

For more advanced usage, you can use the LLuMinary client class:

.. code-block:: python

   from lluminary import LLuMinary

   # Initialize with configuration
   client = LLuMinary(config={
       "default_provider": "openai",
       "providers": {
           "openai": {
               "api_key": "your-openai-key",
               "default_model": "gpt-4o"
           },
           "anthropic": {
               "api_key": "your-anthropic-key",
               "default_model": "claude-haiku-3.5"
           }
       }
   })

   # Generate responses
   response, usage = client.generate_with_usage(
       messages=[
           {"message_type": "human", "message": "What is machine learning?"}
       ],
       system_prompt="You are a helpful assistant."
   )

   print(response)
   print(f"Cost: ${usage['total_cost']}")

See the API documentation for more details on available methods and options.
