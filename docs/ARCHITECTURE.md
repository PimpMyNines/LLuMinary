# LLUMINARY ARCHITECTURE

## Overview

This document provides a comprehensive visual overview of the LLuMinary package's architecture, including component relationships, data flow, and key design patterns. Use this as a reference for understanding the high-level system design and component interactions.

## Table of Contents

- [Overview](#overview)
- [High-Level Architecture](#high-level-architecture)
- [Directory Structure](#directory-structure)
- [Provider Registry Pattern](#provider-registry-pattern)
- [Message Flow](#message-flow)
- [Component Relationships](#component-relationships)
- [Embedding Architecture](#embedding-architecture)
- [Reranking Architecture](#reranking-architecture)
- [Streaming Architecture](#streaming-architecture)
- [Tool Handling Architecture](#tool-handling-architecture)
- [Error Handling Architecture](#error-handling-architecture)
- [Classification Architecture](#classification-architecture)

## Overview

The LLuMinary package provides a unified interface for interacting with multiple LLM providers, including OpenAI, Anthropic, Google, Cohere, and AWS Bedrock. It abstracts away provider-specific implementation details while exposing a consistent API for common operations like text generation, streaming, embeddings, function calling, and document reranking.

## High-Level Architecture

This diagram illustrates the high-level architecture of the LLuMinary package, showing the main components and their relationships:

```mermaid
flowchart TD
    User([User Application]) --> LLuMinary[LLuMinary Handler]

    subgraph Core Components
        LLuMinary --> Router[Provider Router]
        LLuMinary --> Tools[Tool Registry]
        LLuMinary --> ClassificationSystem[Classification System]
        LLuMinary --> ErrorHandling[Error Handling]
    end

    subgraph Provider Layer
        Router --> BaseLLM[Base LLM Abstract Class]
        BaseLLM --> OpenAI[OpenAI Provider]
        BaseLLM --> Anthropic[Anthropic Provider]
        BaseLLM --> Google[Google Provider]
        BaseLLM --> Bedrock[AWS Bedrock Provider]
        BaseLLM --> Cohere[Cohere Provider]
    end

    subgraph External Services
        OpenAI --> OpenAIAPI[OpenAI API]
        Anthropic --> AnthropicAPI[Anthropic API]
        Google --> GoogleAPI[Google Gemini API]
        Bedrock --> BedrockAPI[AWS Bedrock API]
        Cohere --> CohereAPI[Cohere API]
    end

    Tools -.-> OpenAI
    Tools -.-> Anthropic
    Tools -.-> Google
    Tools -.-> Bedrock

    ClassificationSystem -.-> BaseLLM
    ErrorHandling -.-> BaseLLM

    style LLuMinary fill:#d0e0ff,stroke:#0066cc,stroke-width:2px,color:#000000
    style Router fill:#ffe6cc,stroke:#ff9933,stroke-width:2px,color:#000000
    style BaseLLM fill:#f8cecc,stroke:#b85450,stroke-width:2px,color:#000000
    style OpenAI fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style Anthropic fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style Google fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style Bedrock fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style Cohere fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style Tools fill:#e1d5e7,stroke:#9673a6,stroke-width:2px,color:#000000
    style ClassificationSystem fill:#e1d5e7,stroke:#9673a6,stroke-width:2px,color:#000000
    style ErrorHandling fill:#e1d5e7,stroke:#9673a6,stroke-width:2px,color:#000000

    classDef api fill:#f5f5f5,stroke:#666666,stroke-width:1px,color:#000000
    class OpenAIAPI,AnthropicAPI,GoogleAPI,BedrockAPI,CohereAPI api
```

## Directory Structure

The LLuMinary package follows this directory structure:

```
src/lluminary/
├── cli/                   # Command-line interface tools
│   └── classify.py        # Classification command-line tools
├── exceptions.py          # Centralized error hierarchy
├── handler.py             # Main LLuMinary interface class
├── models/                # Core LLM model implementations
│   ├── base.py            # Abstract base class for all LLM implementations
│   ├── classification/    # Classification system components
│   │   ├── classifier.py  # Main classification implementation
│   │   ├── config.py      # Classification configuration management
│   │   └── validators.py  # XML response validators
│   ├── providers/         # LLM provider-specific implementations
│   │   ├── anthropic.py   # Anthropic Claude implementation
│   │   ├── openai.py      # OpenAI implementation
│   │   ├── google.py      # Google Gemini implementation
│   │   ├── bedrock.py     # AWS Bedrock implementation
│   │   ├── cohere.py      # Cohere implementation
│   │   └── provider_template.py # Template for new providers
│   └── router.py          # Model routing and registry
├── prompts/               # System prompts and templates
│   └── classification/    # Classification-specific prompts
│       └── base.yaml      # Base classification prompt template
├── py.typed               # Marker file for PEP 561 typing support
├── tools/                 # Function calling and tool implementations
│   ├── registry.py        # Tool registration system
│   └── validators.py      # Tool input/output validators
├── utils/                 # Shared utility functions
│   └── aws.py             # AWS utilities (authentication, etc.)
└── version.py             # Package version information
```

This structure follows modern Python package organization with clear separation of concerns:

```mermaid
graph TB
    subgraph Package Structure
        Handler[handler.py] --> Models[models/]
        Handler --> Tools[tools/]
        Handler --> Utils[utils/]
        Handler --> Exceptions[exceptions.py]

        Models --> Base[base.py]
        Models --> Router[router.py]
        Models --> Providers[providers/]
        Models --> Classification[classification/]

        Providers --> OpenAI[openai.py]
        Providers --> Anthropic[anthropic.py]
        Providers --> Google[google.py]
        Providers --> Bedrock[bedrock.py]
        Providers --> Cohere[cohere.py]
        Providers --> Template[provider_template.py]

        Tools --> Registry[registry.py]
        Tools --> Validators[validators.py]

        CLI[cli/] --> ClassifyCLI[classify.py]

        Classification --> Classifier[classifier.py]
        Classification --> Config[config.py]
        Classification --> ClassValidators[validators.py]

        Prompts[prompts/] --> ClassificationPrompts[classification/]
    end

    style Handler fill:#d0e0ff,stroke:#0066cc,stroke-width:2px,color:#000000
    style Models fill:#ffe6cc,stroke:#ff9933,stroke-width:2px,color:#000000
    style Providers fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style Tools fill:#e1d5e7,stroke:#9673a6,stroke-width:2px,color:#000000
    style Classification fill:#fff2cc,stroke:#d6b656,stroke-width:2px,color:#000000
    style Exceptions fill:#f8cecc,stroke:#b85450,stroke-width:2px,color:#000000
```

Key design principles in this structure:
1. Clear separation of concerns between components
2. Provider-agnostic base interfaces
3. Shared utilities at the package level
4. Modular organization for extensibility

```mermaid
graph TD
    Client[Client Application] --> LLMHandler[LLM Handler]
    LLMHandler --> Registry[Provider Registry]
    Registry --> OpenAI[OpenAI Provider]
    Registry --> Anthropic[Anthropic Provider]
    Registry --> Google[Google Provider]
    Registry --> Cohere[Cohere Provider]
    Registry --> Custom[Custom Providers]

    OpenAI --> OpenAIAPI[OpenAI API]
    Anthropic --> AnthropicAPI[Anthropic API]
    Google --> GoogleAPI[Google API]
    Cohere --> CohereAPI[Cohere API]
    Custom --> CustomAPI[Custom APIs]

    style LLMHandler fill:#d0e0ff,stroke:#0066cc,stroke-width:2px,color:#000000
    style Registry fill:#ffe6cc,stroke:#ff9933,stroke-width:2px,color:#000000
    style OpenAI fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style Anthropic fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style Google fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style Cohere fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style Custom fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
```

## Provider Registry Pattern

The library uses a registry pattern to dynamically register and manage LLM providers. This enables extensibility by allowing new providers to be added without modifying existing code.

```mermaid
graph TD
    Client[Client Code] -->|"get_llm_from_model('gpt-4o')"| Registry[Provider Registry]
    Registry -->|1. Look up provider| ModelMap[Model Mappings]
    ModelMap -->|2. Return provider info| Registry
    Registry -->|3. Instantiate provider| Provider[Provider Instance]
    Provider -->|4. Return to client| Client

    Developer[Developer] -->|"register_provider('custom', CustomLLM)"| Registry
    Developer -->|"register_model('my-model', 'custom', 'actual-model-id')"| ModelMap

    style Registry fill:#ffe6cc,stroke:#ff9933,stroke-width:2px,color:#000000
    style ModelMap fill:#e1d5e7,stroke:#9673a6,stroke-width:2px,color:#000000
    style Provider fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
```

## Message Flow

This diagram illustrates how messages flow through the system, from user input to provider-specific formats and back.

```mermaid
sequenceDiagram
    participant Client
    participant LLMHandler
    participant ProviderRegistry
    participant ProviderInstance
    participant ExternalAPI

    Client->>LLMHandler: generate(messages, system_prompt)
    LLMHandler->>ProviderRegistry: get_provider()
    ProviderRegistry->>LLMHandler: provider_instance
    LLMHandler->>ProviderInstance: generate(messages, system_prompt)
    ProviderInstance->>ProviderInstance: format_messages_for_model(messages)
    ProviderInstance->>ExternalAPI: API Request
    ExternalAPI->>ProviderInstance: API Response
    ProviderInstance->>ProviderInstance: format_response()
    ProviderInstance->>ProviderInstance: calculate_usage()
    ProviderInstance->>LLMHandler: response, usage, updated_messages
    LLMHandler->>Client: response, usage
```

## Component Relationships

The following diagram shows the relationships between major components of the library.

```mermaid
classDiagram
    class LLM {
        <<abstract>>
        +generate(event_id, system_prompt, messages, max_tokens, temp, functions, retry_limit)
        +stream_generate(event_id, system_prompt, messages, max_tokens, temp, functions, callback)
        +embed(texts, model, batch_size)
        +rerank(query, documents, top_n, return_scores)
        +supports_embeddings()
        +supports_reranking()
        +supports_image_input()
        +get_model_costs()
    }

    class OpenAILLM {
        +SUPPORTED_MODELS
        +EMBEDDING_MODELS
        +RERANKING_MODELS
        +DEFAULT_EMBEDDING_MODEL
        +DEFAULT_RERANKING_MODEL
        +generate()
        +stream_generate()
        +embed()
        +rerank()
    }

    class AnthropicLLM {
        +SUPPORTED_MODELS
        +generate()
        +stream_generate()
    }

    class GoogleLLM {
        +SUPPORTED_MODELS
        +generate()
        +stream_generate()
    }

    class CohereLLM {
        +SUPPORTED_MODELS
        +EMBEDDING_MODELS
        +RERANKING_MODELS
        +generate()
        +stream_generate()
        +embed()
        +rerank()
    }

    class LLMHandler {
        -providers
        -default_provider
        +__init__(config)
        +get_provider(provider_name)
        +generate(messages, system_prompt, provider, max_tokens, temperature, tools, retry_limit)
    }

    class ProviderRegistry {
        +register_provider(provider_name, provider_class)
        +get_llm_from_model(model_name)
        +register_model(friendly_name, provider_name, model_id)
        +list_available_models()
    }

    LLM <|-- OpenAILLM
    LLM <|-- AnthropicLLM
    LLM <|-- GoogleLLM
    LLM <|-- CohereLLM

    LLMHandler o-- LLM : uses
    ProviderRegistry o-- LLM : registers
    LLMHandler --> ProviderRegistry : uses
```

## Embedding Architecture

This diagram illustrates the embedding functionality workflow.

```mermaid
sequenceDiagram
    participant Client
    participant LLM
    participant EmbeddingProvider

    Client->>LLM: supports_embeddings()
    LLM->>Client: true/false

    Client->>LLM: embed(texts=["text1", "text2"])

    alt OpenAI Provider
        LLM->>EmbeddingProvider: embeddings.create(model=embedding_model, input=texts)
        EmbeddingProvider->>LLM: embeddings_response
    else Cohere Provider
        LLM->>EmbeddingProvider: embed(texts=texts, model=embedding_model)
        EmbeddingProvider->>LLM: embeddings_response
    end

    LLM->>LLM: calculate_usage()
    LLM->>Client: embeddings, usage
```

## Reranking Architecture

This diagram shows the document reranking workflow.

```mermaid
sequenceDiagram
    participant Client
    participant LLM
    participant RerankingProvider

    Client->>LLM: supports_reranking()
    LLM->>Client: true/false

    Client->>LLM: rerank(query="...", documents=["doc1", "doc2"], top_n=2)

    alt OpenAI Approach
        LLM->>RerankingProvider: embeddings.create(model=rerank_model, input=documents)
        RerankingProvider->>LLM: document_embeddings
        LLM->>RerankingProvider: embeddings.create(model=rerank_model, input=[query])
        RerankingProvider->>LLM: query_embedding
        LLM->>LLM: calculate_similarities()
        LLM->>LLM: sort_by_relevance()
    else Cohere Approach
        LLM->>RerankingProvider: rerank(query=query, documents=documents, top_n=top_n)
        RerankingProvider->>LLM: ranked_results
    end

    LLM->>LLM: calculate_usage()
    LLM->>Client: {"ranked_documents": [...], "indices": [...], "scores": [...], "usage": {...}}
```

## Streaming Architecture

This diagram illustrates the streaming response workflow.

```mermaid
sequenceDiagram
    participant Client
    participant LLM
    participant StreamingProvider

    Client->>LLM: stream_generate(messages, system_prompt, callback=process_chunk)

    LLM->>StreamingProvider: create_streaming_request()
    StreamingProvider-->>LLM: stream_iterator

    loop For each chunk
        StreamingProvider-->>LLM: next_chunk
        LLM->>LLM: process_chunk()
        LLM->>LLM: calculate_partial_usage()

        alt Callback provided
            LLM->>Client: callback(chunk, partial_usage)
        end

        LLM-->>Client: yield chunk, partial_usage
    end

    LLM->>LLM: calculate_final_usage()
    LLM-->>Client: yield "", final_usage
```

## Tool Handling Architecture

The tool handling system enables function calling and tool use with LLMs. This architecture supports a consistent interface across providers with different native function calling capabilities.

```mermaid
graph TD
    Client[Client Application] -->|"function_call(tools=[...])"| LLuMinary

    subgraph LLuMinary Package
        LLuMinary[LLuMinary Handler] -->|verify_tools| Registry[Tool Registry]
        Registry -->|validate_schema| Validators[Tool Validators]

        LLuMinary -->|"generate(..., functions=tools)"| Provider[Provider Instance]

        subgraph "Provider-Specific Handling"
            Provider -->|supports_native_tools?| Decision{Native Tools Support?}

            Decision -->|Yes| NativeFormat[Format for Native Function Calling]
            Decision -->|No| PromptFormat[Format as Prompt Instructions]

            NativeFormat --> APICall[Provider API with Functions]
            PromptFormat --> APICall2[Provider API without Functions]
        end

        APICall --> ResponseParser[Parse Function Call Response]
        APICall2 --> TextParser[Parse Text for Function Calls]

        ResponseParser --> Client
        TextParser --> Client
    end

    style LLuMinary fill:#d0e0ff,stroke:#0066cc,stroke-width:2px,color:#000000
    style Registry fill:#e1d5e7,stroke:#9673a6,stroke-width:2px,color:#000000
    style Validators fill:#e1d5e7,stroke:#9673a6,stroke-width:2px,color:#000000
    style Decision fill:#ffe6cc,stroke:#ff9933,stroke-width:2px,color:#000000
    style Provider fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style NativeFormat,PromptFormat fill:#fff2cc,stroke:#d6b656,stroke-width:1px,color:#000000
```

### Tool Registration Flow

```mermaid
sequenceDiagram
    participant Client
    participant ToolRegistry
    participant Validator
    participant LLM
    participant ProviderAPI

    Client->>ToolRegistry: register_tool(name, schema, description)
    ToolRegistry->>Validator: validate_schema(schema)
    Validator-->>ToolRegistry: validation_result

    alt Schema Valid
        ToolRegistry-->>Client: Success
    else Schema Invalid
        ToolRegistry-->>Client: ValidationError
    end

    Client->>LLM: generate(messages, functions=tools)
    LLM->>ToolRegistry: get_tools_for_provider(provider_type)
    ToolRegistry-->>LLM: formatted_tools

    LLM->>ProviderAPI: api_call(messages, formatted_tools)
    ProviderAPI-->>LLM: response_with_tool_calls

    LLM->>LLM: parse_tool_calls(response)
    LLM-->>Client: response, tool_calls
```

## Error Handling Architecture

The LLuMinary package implements a robust error handling system with a unified exception hierarchy and provider-specific error mapping.

```mermaid
graph TD
    subgraph Error Hierarchy
        BaseError[LLMError] --> RateLimit[LLMRateLimitError]
        BaseError --> Timeout[LLMTimeoutError]
        BaseError --> Connection[LLMConnectionError]
        BaseError --> Auth[LLMAuthenticationError]
        BaseError --> Validation[LLMValidationError]
        BaseError --> ModelError[LLMMistake]
    end

    subgraph Provider Specific
        OpenAIError[OpenAI Exceptions] --> ErrorMapper1[Error Mapper]
        AnthropicError[Anthropic Exceptions] --> ErrorMapper2[Error Mapper]
        GoogleError[Google Exceptions] --> ErrorMapper3[Error Mapper]
        CohereError[Cohere Exceptions] --> ErrorMapper4[Error Mapper]
        BedrockError[AWS Exceptions] --> ErrorMapper5[Error Mapper]
    end

    ErrorMapper1 --> BaseError
    ErrorMapper2 --> BaseError
    ErrorMapper3 --> BaseError
    ErrorMapper4 --> BaseError
    ErrorMapper5 --> BaseError

    Application[User Application] --> TryCatch[Try/Catch Block]
    TryCatch --> BaseError

    style BaseError fill:#f8cecc,stroke:#b85450,stroke-width:2px,color:#000000
    style RateLimit,Timeout,Connection,Auth,Validation,ModelError fill:#f8cecc,stroke:#b85450,stroke-width:1px,color:#000000
    style OpenAIError,AnthropicError,GoogleError,CohereError,BedrockError fill:#d5e8d4,stroke:#82b366,stroke-width:1px,color:#000000
    style ErrorMapper1,ErrorMapper2,ErrorMapper3,ErrorMapper4,ErrorMapper5 fill:#fff2cc,stroke:#d6b656,stroke-width:1px,color:#000000
```

### Error Handling Flow

```mermaid
sequenceDiagram
    participant User
    participant LLuMinary
    participant Provider
    participant ErrorMapper

    User->>LLuMinary: generate(messages)
    LLuMinary->>Provider: api_call(formatted_messages)

    alt API Success
        Provider-->>LLuMinary: successful_response
        LLuMinary-->>User: formatted_response, usage
    else API Error
        Provider-->>LLuMinary: provider_specific_error
        LLuMinary->>ErrorMapper: map_error(provider_error)
        ErrorMapper-->>LLuMinary: standard_error

        alt Retryable Error
            Note over LLuMinary: Add error to messages
            LLuMinary->>Provider: retry_api_call(updated_messages)
            Provider-->>LLuMinary: successful_response
            LLuMinary-->>User: formatted_response, usage
        else Non-Retryable Error
            LLuMinary-->>User: LLMError exception
        end
    end
```

## Classification Architecture

The classification system provides a consistent way to categorize text across all providers.

```mermaid
graph TD
    subgraph Classification System
        Client[Client Application] -->|classify| LLM[LLM Provider]
        Client -->|classify_from_file| ConfigLoader[Config Loader]

        ConfigLoader -->|load config| Config[Classification Config]
        Config --> LLM

        LLM -->|format prompt| Prompt[XML Classification Prompt]
        Prompt --> ProviderAPI[Provider API]
        ProviderAPI --> XMLResponse[XML Response]
        XMLResponse --> Parser[XML Parser]
        Parser --> Validator[Response Validator]
        Validator --> Results[Classification Results]
        Results --> Client
    end

    subgraph Configuration
        JSONConfig[JSON Config File] --> ConfigLoader
        Library[Classification Library] --> ConfigLoader
    end

    style Client fill:#d0e0ff,stroke:#0066cc,stroke-width:2px,color:#000000
    style LLM fill:#d5e8d4,stroke:#82b366,stroke-width:2px,color:#000000
    style ConfigLoader fill:#ffe6cc,stroke:#ff9933,stroke-width:2px,color:#000000
    style Config fill:#ffe6cc,stroke:#ff9933,stroke-width:2px,color:#000000
    style Prompt fill:#e1d5e7,stroke:#9673a6,stroke-width:2px,color:#000000
    style Parser fill:#fff2cc,stroke:#d6b656,stroke-width:1px,color:#000000
    style Validator fill:#fff2cc,stroke:#d6b656,stroke-width:1px,color:#000000
```

### Classification Flow

```mermaid
sequenceDiagram
    participant Client
    participant ClassificationConfig
    participant LLM
    participant ProviderAPI
    participant XMLParser

    alt Direct Classification
        Client->>LLM: classify(messages, categories, examples)
    else From Config File
        Client->>ClassificationConfig: load_from_file(filepath)
        ClassificationConfig-->>Client: config
        Client->>LLM: classify_from_file(filepath, messages)
    end

    LLM->>LLM: format_classification_prompt(messages, categories, examples)
    LLM->>ProviderAPI: api_call(formatted_prompt)
    ProviderAPI-->>LLM: text_response

    LLM->>XMLParser: parse_xml_response(text_response)
    XMLParser-->>LLM: parsed_categories

    LLM->>LLM: validate_categories(parsed_categories)
    LLM->>LLM: calculate_usage()

    LLM-->>Client: selected_categories, usage
```

## Related Documentation

- [API_REFERENCE](./API_REFERENCE.md) - Detailed API reference for all components
- [TUTORIALS](./TUTORIALS.md) - Step-by-step guides for common use cases
- [ERROR_HANDLING](./development/ERROR_HANDLING.md) - Details on the error handling system
- [PROVIDER_TESTING](./development/PROVIDER_TESTING.md) - Information on provider implementation testing
- [MODELS](./development/MODELS.md) - Comprehensive list of supported models
