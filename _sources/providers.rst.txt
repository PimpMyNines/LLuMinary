Provider Documentation
====================

LLuMinary supports multiple LLM providers through a unified interface. This section provides detailed documentation for each supported provider, including setup, authentication, supported models, and provider-specific features.

.. toctree::
   :maxdepth: 2
   :caption: Providers:

   providers/openai
   providers/anthropic
   providers/google
   providers/cohere
   providers/bedrock

Provider Comparison
------------------

The table below provides a quick comparison of the features supported by each provider:

+------------------+--------+-----------+--------+--------+---------+
| Feature          | OpenAI | Anthropic | Google | Cohere | Bedrock |
+==================+========+===========+========+========+=========+
| Text Generation  | ✓      | ✓         | ✓      | ✓      | ✓       |
+------------------+--------+-----------+--------+--------+---------+
| Embeddings       | ✓      | ✓         | ✓      | ✓      | ✓       |
+------------------+--------+-----------+--------+--------+---------+
| Streaming        | ✓      | ✓         | ✓      | ✓      | ✓       |
+------------------+--------+-----------+--------+--------+---------+
| Tool Calling     | ✓      | ✓         | ✓      | ✓      | ✓       |
+------------------+--------+-----------+--------+--------+---------+
| Image Input      | ✓      | ✓         | ✓      | ×      | ✓       |
+------------------+--------+-----------+--------+--------+---------+
| Image Generation | ✓      | ×         | ✓      | ×      | ✓       |
+------------------+--------+-----------+--------+--------+---------+
| Reranking        | ✓      | ×         | ✓      | ✓      | ×       |
+------------------+--------+-----------+--------+--------+---------+

Authentication Strategy
---------------------

LLuMinary follows a consistent authentication approach across providers:

1. API keys passed directly to constructor
2. Environment variables
3. Configuration files

For detailed authentication information for each provider, refer to their respective documentation pages.

Switching Between Providers
-------------------------

One of the key advantages of LLuMinary is the ability to easily switch between providers:

.. code-block:: python

    from lluminary import LLuMinary

    # Using OpenAI
    llm = LLuMinary(provider="openai", model="gpt-4o")

    # Switch to Anthropic
    llm = LLuMinary(provider="anthropic", model="claude-3-haiku")

    # Switch to Google
    llm = LLuMinary(provider="google", model="gemini-1.5-pro")

For provider-specific parameters, see individual provider documentation.
