# Provider Template Tests

This directory contains comprehensive tests for the `provider_template.py` module, which serves as a template for implementing new LLM providers in the system.

## Overview

These tests cover all aspects of the provider template implementation:

- **Core Provider Attributes**: Tests for model list, context window sizes, cost parameters, etc.
- **Initialization**: Tests for proper object initialization with various parameters
- **Authentication**: Tests for API key handling from environment variables and AWS Secrets Manager
- **Message Formatting**: Tests for converting standard message format to provider-specific format
- **Text Generation**: Tests for the raw_generate method including error handling
- **Image Handling**: Tests for processing image inputs (currently skipped)
- **Tool Support**: Tests for function/tool handling
- **Registration**: Tests for provider registration with the router (currently skipped)
- **Streaming**: Tests for streaming functionality (currently skipped)

## Test Files

- **test_provider_template.py**: Core tests for the provider class including attributes, initialization, authentication, message formatting, and generation
- **test_provider_template_utils.py**: Tests for utility methods like image processing and tool handling
- **test_provider_template_streaming.py**: Tests for streaming functionality (all skipped)
- **test_provider_template_registration.py**: Tests for provider registration with the router (all skipped)

## Test Status

Currently:
- 19 tests passing
- 10 tests skipped

The skipped tests cover functionality that is either left as stub methods in the template (`pass` methods) or requires functionality that would be implemented in the actual provider. These tests include:

1. AWS Secrets Manager authentication
2. Image processing (URL and file)
3. Streaming functionality
4. Provider registration with the router

## Notes for Implementation

When implementing a real provider based on the template:

1. The skipped tests should be unskipped and implemented to ensure full coverage
2. The streaming implementation should be added and tested if the provider supports it
3. The image processing methods should be implemented if the provider supports multimodal inputs
4. The provider registration with the router should be tested thoroughly

## Running the Tests

To run these tests:

```bash
python -m pytest tests/unit/provider_template/ -v
```

To check coverage:

```bash
python -m pytest tests/unit/provider_template/ --cov=src/llmhandler/models/providers/provider_template.py
```
