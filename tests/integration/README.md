# LLMHandler Integration Tests

This directory contains integration tests for the LLMHandler library. Integration tests make actual API calls to LLM providers to verify the functionality of the library in real-world scenarios.

## Test Structure

Integration tests are organized by feature, with each file focusing on a specific area of functionality:

| Test File | Description |
|-----------|-------------|
| `test_advanced_features.py` | Tests for response processing, retry mechanisms, function calling, and thinking budget |
| `test_advanced_features_extended.py` | Tests for JSON mode, long context handling, tool use, and parallel processing |
| `test_classification.py` | Tests for single and multi-category classification functionality |
| `test_cli_integration.py` | Tests for classification CLI commands (validate, list_configs, test) |
| `test_cost_tracking.py` | Tests for cost tracking across providers and with different parameters |
| `test_cross_provider_integration.py` | Comprehensive cross-provider tests for consistent behavior |
| `test_embeddings.py` | Tests for embedding generation and similarity calculations |
| `test_generate.py` | Basic text generation tests across all providers |
| `test_image_generation.py` | Tests for image generation with different parameters |
| `test_images.py` | Tests for image input processing |
| `test_reranking.py` | Tests for document reranking functionality |
| `test_streaming_integration.py` | Tests for streaming functionality across providers |
| `test_tool_registry_integration.py` | Tests for tool registration, validation, and function calling |

## Running the Tests

### Prerequisites

To run these tests, you need to have API keys for the providers you want to test. The tests will automatically skip providers for which no credentials are available.

### Running All Integration Tests

```bash
python -m pytest tests/integration/ -v
```

### Running Tests for a Specific Feature

```bash
python -m pytest tests/integration/test_embeddings.py -v
```

### Running Tests for a Specific Provider

```bash
python -m pytest tests/integration/ -k "openai" -v
```

### Running Tests with a Specific Marker

```bash
python -m pytest tests/integration/ -m "tools" -v
```

## Available Test Markers

- `integration` - Applied to all integration tests
- `classification` - Tests for classification functionality
- `image` - Tests for image input functionality
- `image_generation` - Tests for image generation functionality
- `tools` - Tests for tool registry functionality
- `cli` - Tests for CLI functionality
- `cross_provider` - Tests that verify behavior across providers
- `streaming` - Tests for streaming functionality

## Notes on Test Implementation

1. **Graceful Failure**: All tests use try/except blocks with `pytest.skip()` to handle API authentication failures gracefully.
2. **Provider Fallbacks**: Many tests attempt to run with multiple providers, falling back if one fails.
3. **Cross-provider Comparison**: Where relevant, tests compare results across different providers.
4. **Cost Awareness**: Tests use minimal tokens and the smallest models possible to minimize costs during testing.
5. **Parallel Testing**: Some tests validate parallel request handling to ensure concurrent usage works correctly.

## See Also

For more detailed information about the integration tests, see [INTEGRATION_TESTS.md](INTEGRATION_TESTS.md).
