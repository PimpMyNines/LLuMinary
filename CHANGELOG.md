# Changelog

All notable changes to the LLuMinary project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-03-10

### Initial Release

LLuMinary 1.0.0 introduces a versatile Python library for interacting with multiple LLM providers through a unified interface.

#### Core Features

- 🤝 **Unified API** across multiple providers (OpenAI, Anthropic, Google, Cohere)
- 🔄 **Provider Registry Pattern** enabling easy extension with custom providers
- 📝 **Text Generation** with consistent interface and error handling
- 🌊 **Streaming Responses** from all major providers
- 🔍 **Embeddings** generation for semantic search and similarity matching
- 📚 **Document Reranking** for improved search relevance
- 🔧 **Function Calling** for tool use and external API integration
- 📊 **Token Counting & Cost Tracking** for budget management
- 🖼️ **Image Input Support** for multimodal models
- 🏷️ **Classification** for categorizing text inputs
- 🧪 **Enhanced Type Validation** for complex nested types and robust error handling
- 🖥️ **CLI Tools** for classification management and testing

#### Provider Support

- **OpenAI**
  - Support for GPT-4o, GPT-4, GPT-3.5-Turbo models
  - Text embedding models (text-embedding-3-small, text-embedding-3-large)
  - Image input support for multimodal models
  - Function calling and tool use

- **Anthropic**
  - Support for Claude 3 models (Opus, Sonnet, Haiku)
  - Claude 2 support
  - Streaming response capabilities
  - Image input support

- **Google**
  - Support for Gemini models (Pro, Flash)
  - Streaming response capabilities
  - Image input support for multimodal models

- **Cohere**
  - Support for Command models
  - Embedding models
  - Reranking capabilities

#### Architecture & Design

- Robust error handling system with unified exception hierarchy
- Provider-specific error mapping to standardized exceptions
- Automatic retry logic for transient errors
- Comprehensive type hints throughout the codebase
- Modular architecture for extensibility
- Consistent message formatting across providers

#### Documentation

- Comprehensive API reference documentation
- Architecture diagrams and component relationships
- Step-by-step tutorials for common use cases
- Working code examples for all features
- Test coverage reports and implementation guides

#### Developer Tools

- Classification CLI for managing and testing classification configurations
- Interactive testing notebook for exploring capabilities
- Type validation decorators for function inputs/outputs
- Diagnostic utilities for environment setup

#### Utilities

- Token counting and cost estimation
- Secure API key handling
- Batch processing for embeddings
- Customizable logging
