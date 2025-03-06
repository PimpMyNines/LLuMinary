# LLuMinary Test Suite

This directory contains tests for the LLuMinary package. The tests are organized to provide comprehensive coverage of the package's functionality while keeping test execution efficient.

## Test Structure

- **Unit Tests**: `tests/unit/` - Tests individual components without making API calls
- **Integration Tests**: `tests/integration/` - Tests that make actual API calls to LLM providers
- **Test Data**: `tests/test_data/` - Sample data and example tests

## Running Tests

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Run only unit tests (no API calls)
python -m pytest tests/unit/ -v

# Run only integration tests (makes API calls)
python -m pytest tests/integration/ -v

# Run only image-related tests
python -m pytest tests/ -m "image" -v

# Run tests for a specific provider
python -m pytest tests/ -k "openai" -v
```

### Run Tests in Parallel

```bash
# Run tests using all available CPU cores
python -m pytest tests/ -n auto -v
```

### Run Tests with Coverage

```bash
python -m pytest tests/ --cov=src/lluminary --cov-report=term
```

## Test Categories (Markers)

The tests use pytest markers for organization:

- **unit**: Unit tests that don't make API calls
- **integration**: Tests that make actual API calls
- **image**: Tests that involve image processing
- **image_generation**: Tests for image generation functionality
- **classification**: Tests for classification functionality
- **tools**: Tests for function calling and tools functionality
- **cli**: Tests for CLI functionality
- **cross_provider**: Tests that verify behavior across providers
- **streaming**: Tests for streaming functionality
- **cost**: Tests for cost tracking
- **rate_limiting**: Tests for rate limiting
- **provider_errors**: Tests for provider error handling
- **dynamic_model_selection**: Tests for model selection
- **reranking**: Tests for reranking functionality
- **embedding**: Tests for embedding functionality
- **api**: Tests that make external API calls
- **slow**: Tests that are known to be slow

## Test Organization

Unit tests are organized by provider and feature:
- Provider-specific tests:
  - `test_openai_*.py`: Tests for OpenAI provider functionality
  - `test_anthropic_*.py`: Tests for Anthropic provider functionality
  - `test_google_*.py`: Tests for Google provider functionality
  - `test_bedrock_*.py`: Tests for AWS Bedrock provider functionality
  - `test_cohere_*.py`: Tests for Cohere provider functionality
- Feature-specific tests:
  - `test_classification_*.py`: Tests for classification functionality
  - `test_tool_registry.py`: Tests for function calling and tools
  - `test_embedding.py`: Tests for embedding functionality
  - `test_reranking.py`: Tests for reranking functionality
  - `test_streaming.py`: Tests for streaming functionality

## Example Test

See `tests/test_data/example_test.py` for a simple example of how to create your own tests with LLuMinary.

## Writing New Tests

When adding new tests, please follow these guidelines:

1. Use appropriate markers to categorize your tests
2. For integration tests, always use try/except blocks with `pytest.skip()` to handle API failures gracefully
3. Keep provider-specific tests parametrized when possible to avoid duplication
4. Use mock_model_clients fixture for unit tests to avoid making actual API calls
5. Include both success and error paths in your tests
6. Test with multiple providers when testing core functionality

## Cost Considerations

Integration tests make real API calls that incur costs. To minimize costs:

1. Use the smallest, least expensive models when possible (e.g., claude-haiku-3.5 instead of claude-sonnet-3.5)
2. Keep max_tokens low (50-100 tokens is sufficient for most tests)
3. Run only the specific tests you need during development
4. Use the test markers to run only relevant test categories
