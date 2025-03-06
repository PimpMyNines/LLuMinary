# Integration Tests for LLMHandler

This document provides an overview of the integration tests created for the LLMHandler library.

## Test Categories

Integration tests make actual API calls to LLM providers and are organized by feature:

### Basic Generation Tests
- `test_generate.py`: Tests the core text generation functionality across all providers
  - Tests system prompt influence
  - Tests token limits
  - Tests basic generation across all providers

### Advanced Features Tests
- `test_advanced_features.py`: Tests advanced features like:
  - Response processing
  - Retry mechanisms
  - Function calling
  - Thinking budget

- `test_advanced_features_extended.py`: Tests additional advanced features:
  - JSON mode output
  - Long context handling
  - Tool use
  - Parallel request handling

### Classification Tests
- `test_classification.py`: Tests text classification functionality
  - Single category classification
  - Classification with examples
  - Multi-message classification

### Image Input Tests
- `test_images.py`: Tests image input functionality
  - Single image processing from URL
  - Multiple images processing

### Image Generation Tests
- `test_image_generation.py`: Tests image generation functionality
  - Basic image generation across providers
  - Image generation with different parameters
  - Image generation with style parameters

### Embedding and Reranking Tests
- `test_embeddings.py`: Tests embedding generation and usage
  - Single text embedding (for supported providers)
  - Batch embeddings (for supported providers)
  - Similarity calculation with embeddings
  - Testing provider-specific embedding models
  - Testing missing embedding support in certain providers

- `test_reranking.py`: Tests document reranking functionality
  - OpenAI reranking
  - Cohere reranking
  - Cross-provider reranking comparison
  - Reranking with empty documents
  - Reranking with long documents

### Streaming Tests
- `test_streaming_integration.py`: Tests streaming functionality
  - Provider-specific streaming tests (OpenAI, Anthropic, Google, Bedrock)
  - Cross-provider streaming comparison
  - Streaming cancellation

### Cost Tracking Tests
- `test_cost_tracking.py`: Tests cost tracking functionality
  - Provider-specific cost tracking
  - Cost scaling with response length
  - Cross-provider cost comparison
  - Function calling cost impact

### CLI Integration Tests
- `test_cli_integration.py`: Tests CLI functionality
  - Testing the validate command
  - Testing the list_configs command
  - Testing the classification test command

### Tool Registry Integration Tests
- `test_tool_registry_integration.py`: Tests tool registry functionality
  - Tool registration and usage
  - Function calling integration
  - Error handling for tool execution

### Cross-Provider Integration Tests
- `test_cross_provider_integration.py`: Comprehensive cross-provider tests
  - Simple prompts across all providers
  - Error handling consistency
  - Model switching
  - Parallel requests to different providers

## Running the Tests

All integration tests gracefully handle API authentication errors, allowing them to skip individual tests when credentials are not available.

To run all integration tests:
```bash
python -m pytest tests/integration/ -v
```

To run a specific integration test category:
```bash
python -m pytest tests/integration/test_embeddings.py -v
```

To run tests with a specific marker:
```bash
python -m pytest tests/integration/ -m "tools" -v
```

## Adding New Integration Tests

When adding new integration tests, follow these guidelines:

1. Use the `@pytest.mark.integration` and appropriate feature-specific markers
2. Always use try/except blocks with `pytest.skip()` to handle API failures gracefully
3. Keep provider-specific tests parametrized when possible to avoid duplication
4. Include meaningful assertions that validate the expected behavior
5. Implement fallback mechanisms to try multiple providers when possible

Integration tests should focus on testing the full functionality with real API calls, while unit tests (with mocks) should test the internal logic and error handling.

## Available Test Markers

- `@pytest.mark.integration` - Applied to all integration tests
- `@pytest.mark.classification` - Tests for classification functionality
- `@pytest.mark.image` - Tests for image input functionality
- `@pytest.mark.image_generation` - Tests for image generation functionality
- `@pytest.mark.tools` - Tests for tool registry functionality
- `@pytest.mark.cli` - Tests for CLI functionality
- `@pytest.mark.cross_provider` - Tests that verify behavior across providers
- `@pytest.mark.streaming` - Tests for streaming functionality
