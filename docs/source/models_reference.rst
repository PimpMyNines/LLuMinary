.. _models_reference:

Models Reference
==============

This page provides a comprehensive reference of all models supported by LLuMinary, including their capabilities, context windows, and pricing information.

Model Support Matrix
------------------

The table below provides a quick overview of the features supported by each provider model:

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
| Thinking/Reasoning | ✓    | ✓         | ✓      | ×      | ✓       |
+------------------+--------+-----------+--------+--------+---------+

OpenAI Models
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15 15 20 20 20

   * - Model
     - Context Window
     - Image Support
     - Tool Calling
     - Reasoning
     - Input Cost
     - Output Cost
     - Image Cost
   * - gpt-4.5-preview
     - 128,000
     - ✓
     - ✓
     - ×
     - $0.0000750
     - $0.00015
     - Variable
   * - gpt-4o
     - 128,000
     - ✓
     - ✓
     - ×
     - $0.0000025
     - $0.00001
     - Variable
   * - gpt-4o-mini
     - 128,000
     - ✓
     - ✓
     - ×
     - $0.00000015
     - $0.0000006
     - Variable
   * - o1
     - 200,000
     - ✓
     - ✓
     - ✓
     - $0.000015
     - $0.00006
     - Variable
   * - o3-mini
     - 200,000
     - ✓
     - ✓
     - ✓
     - $0.0000011
     - $0.0000044
     - Variable

OpenAI's image costs depend on resolution and detail level. Low detail: 85 tokens, High detail: varies by size.

Anthropic Models
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15 15 20 20 20

   * - Model
     - Context Window
     - Image Support
     - Tool Calling
     - Reasoning
     - Input Cost
     - Output Cost
     - Image Cost
   * - claude-3-5-sonnet-20240620-v1:0
     - 200,000
     - ✓
     - ✓
     - ×
     - $0.000003
     - $0.000015
     - $0.024/image
   * - claude-3-haiku-20240307-v1:0
     - 200,000
     - ✓
     - ✓
     - ×
     - $0.00000025
     - $0.00000125
     - $0.024/image
   * - claude-3-opus-20240229-v1:0
     - 200,000
     - ✓
     - ✓
     - ×
     - $0.000015
     - $0.000075
     - $0.024/image
   * - claude-3-sonnet-20240229-v1:0
     - 200,000
     - ✓
     - ✓
     - ×
     - $0.000003
     - $0.000015
     - $0.024/image
   * - claude-3-7-sonnet-preview-20240626-v1:0
     - 200,000
     - ✓
     - ✓
     - ✓
     - $0.000003
     - $0.000015
     - $0.024/image

Google (Gemini) Models
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15 15 20 20 20

   * - Model
     - Context Window
     - Image Support
     - Tool Calling
     - Reasoning
     - Input Cost
     - Output Cost
     - Image Cost
   * - gemini-2.0-flash
     - 128,000
     - ✓
     - ✓
     - ×
     - $0.0000025
     - $0.00001
     - $0.001/image
   * - gemini-2.0-flash-lite-preview-02-05
     - 128,000
     - ✓
     - ✓
     - ×
     - $0.000001
     - $0.000004
     - $0.0005/image
   * - gemini-2.0-pro-exp-02-05
     - 128,000
     - ✓
     - ✓
     - ×
     - $0.000003
     - $0.000012
     - $0.002/image
   * - gemini-2.0-flash-thinking-exp-01-21
     - 128,000
     - ✓
     - ✓
     - ✓
     - $0.000004
     - $0.000016
     - $0.002/image

Cohere Models
----------

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Model
     - Type
     - Context Window
     - Tool Calling
     - Pricing
   * - command-r
     - Text Generation
     - 128,000
     - ✓
     - $1.00/M tokens (input), $3.00/M tokens (output)
   * - command-r-plus
     - Text Generation
     - 128,000
     - ✓
     - $3.00/M tokens (input), $15.00/M tokens (output)
   * - command-light
     - Text Generation
     - 128,000
     - ✓
     - $0.30/M tokens (input), $0.60/M tokens (output)
   * - embed-english-v3.0
     - Embedding
     - N/A
     - N/A
     - $0.10/M tokens
   * - embed-multilingual-v3.0
     - Embedding
     - N/A
     - N/A
     - $0.10/M tokens
   * - rerank-english-v3.0
     - Reranking
     - N/A
     - N/A
     - $0.10/M tokens

AWS Bedrock Models
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15 15 20 20 20

   * - Model
     - Context Window
     - Image Support
     - Tool Calling
     - Reasoning
     - Input Cost
     - Output Cost
     - Image Cost
   * - us.anthropic.claude-3-5-haiku-20241022-v1:0
     - 200,000
     - ✓
     - ✓
     - ×
     - $0.000001
     - $0.000005
     - $0.024/image
   * - us.anthropic.claude-3-5-sonnet-20240620-v1:0
     - 200,000
     - ✓
     - ✓
     - ×
     - $0.000003
     - $0.000015
     - $0.024/image
   * - us.anthropic.claude-3-7-sonnet-20250219-v1:0
     - 200,000
     - ✓
     - ✓
     - ✓
     - $0.000003
     - $0.000015
     - $0.024/image
   * - amazon.titan-text-express-v1
     - 8,000
     - ×
     - ×
     - ×
     - $0.0000008
     - $0.0000008
     - N/A
   * - amazon.titan-text-premier-v1
     - 32,000
     - ×
     - ×
     - ×
     - $0.000009
     - $0.000009
     - N/A
   * - amazon.titan-embed-text-v1
     - N/A
     - N/A
     - N/A
     - N/A
     - $0.0000002/token
     - N/A
     - N/A
   * - meta.llama3-70b-instruct-v1
     - 8,000
     - ×
     - ×
     - ×
     - $0.00000075
     - $0.000001
     - N/A

Feature Support Details
--------------------

Image Support
~~~~~~~~~~~~

Models with image support can process images as part of the input:

- **OpenAI**: Processes images as JPEG with base64 encoding
- **Anthropic**: Supports multiple image formats with various sizing
- **Google**: Supports multiple image formats
- **AWS Bedrock**: Uses PNG format with preserved transparency

Tool/Function Calling
~~~~~~~~~~~~~~~~~~

Models supporting tool/function calling can:

- Parse and understand function schemas
- Choose appropriate functions to call
- Format arguments correctly for function execution
- Process function results and incorporate them into responses

Each provider has a unique format for tool definitions and responses:

- **OpenAI**: Uses a function calling format with "arguments" as a JSON string
- **Anthropic**: Uses a structured content format with toolUse and toolResult objects
- **Google**: Uses Part.from_function_call and Part.from_function_response
- **Cohere**: Uses tools with parameter_definitions format
- **AWS Bedrock**: Uses provider-specific formats depending on underlying model

Reasoning/Thinking
~~~~~~~~~~~~~~~

Models with reasoning/thinking capabilities can:

- Generate detailed step-by-step reasoning
- Provide internal thought process (visible or invisible to user)
- Solve complex problems more systematically

Available on:
- **OpenAI**: o1, o3-mini
- **Anthropic**: claude-3-7-sonnet-preview models
- **Google**: gemini-2.0-flash-thinking-exp-01-21
- **AWS Bedrock**: claude-3-7-sonnet models

Usage Notes
---------

1. **Context Window**: Represents the maximum number of tokens the model can process in a single conversation (input + output).

2. **Token Calculation**: Different providers calculate tokens differently:
   - Text tokens: ~4 characters per token (English, varies by language)
   - Image tokens: Different calculation methods per provider

3. **Costs**: Prices are in USD and subject to change. Always check the provider's pricing page for current rates.

4. **Model Availability**: Some models may be experimental or in preview. Availability might change.

5. **Rate Limits**: Each provider implements different rate limiting policies. Check provider documentation for details.

Provider-Specific Considerations
-----------------------------

For detailed information about each provider, including authentication, specific parameters, and error handling, see the individual provider documentation pages:

- :doc:`/providers/openai`
- :doc:`/providers/anthropic`
- :doc:`/providers/google`
- :doc:`/providers/cohere`
- :doc:`/providers/bedrock`
