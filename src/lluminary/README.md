# LLuMinary Package Structure

This directory contains the core implementation of the LLuMinary package. Here's an overview of the package structure:

## Package Components

- `__init__.py`: Main package exports and version information
- `handler.py`: Main LLuMinary class interface
- `exceptions.py`: Error hierarchy for all LLM-related exceptions
- `version.py`: Version information using single-source versioning
- `py.typed`: Marker file for PEP 561 typing support

### Subdirectories

- `models/`: Core LLM model implementations
  - `base.py`: Base classes for all LLM implementations
  - `router.py`: Provider registry and model routing
  - `providers/`: Provider-specific implementations
    - `openai.py`: OpenAI provider implementation
    - `anthropic.py`: Anthropic provider implementation
    - `google.py`: Google provider implementation
    - `bedrock.py`: AWS Bedrock provider implementation
    - `cohere.py`: Cohere provider implementation
    - `provider_template.py`: Template for adding new providers
  - `classification/`: Text classification functionality
    - `classifier.py`: Classifier implementation
    - `config.py`: Classification configuration
    - `validators.py`: Input validation for classification
  - `utils/`: Provider-specific utility functions

- `cli/`: Command-line interface components
  - `classify.py`: CLI for text classification

- `prompts/`: YAML prompt templates
  - `classification/`: Classification prompt templates

- `tools/`: Function calling and tools support
  - `registry.py`: Tool registry for function calling
  - `validators.py`: Validators for tool definitions

- `utils/`: Shared utility functions
  - `aws.py`: AWS-specific utilities

## Implementation Guide

When extending the package:

1. Follow the provider template when adding new providers
2. Maintain consistent error handling across all components
3. Ensure all public APIs are properly typed
4. Add proper docstrings to all public functions and classes
5. Include unit tests for all new functionality

## Error Handling

All provider-specific errors are mapped to the standard error hierarchy defined in `exceptions.py`. This ensures
that applications using LLuMinary can handle errors consistently regardless of the underlying provider.

## Testing

Each component should have corresponding unit tests in the `tests/unit/` directory. Integration tests
should be added to `tests/integration/` to verify cross-component functionality.
