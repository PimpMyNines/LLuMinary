# UPDATED COMPONENTS DOCUMENTATION

## Overview

This document details recent updates to key components in the LLuMinary library. These enhancements improve the library's functionality, type safety, and test coverage, providing a comprehensive reference of newly implemented features and improvements.

## Table of Contents

- [Enhanced Type Validation](#enhanced-type-validation)
  - [Complex Nested Type Support](#complex-nested-type-support)
  - [Union and Optional Type Handling](#union-and-optional-type-handling)
  - [JSON Serialization Validation](#json-serialization-validation)
  - [Usage Examples](#usage-examples)
- [Classification Components](#classification-components)
  - [Core Components](#core-components)
  - [Comprehensive Testing](#comprehensive-testing)
  - [Usage Examples](#classification-usage-examples)
- [CLI Components](#cli-components)
  - [Classification Management Commands](#classification-management-commands)
  - [Command Usage](#command-usage)
  - [Testing and Validation](#testing-and-validation)
- [Provider Implementation Improvements](#provider-implementation-improvements)
  - [Type Safety Enhancements](#type-safety-enhancements)
  - [Error Handling Improvements](#error-handling-improvements)
  - [Consistent Interface Implementation](#consistent-interface-implementation)

## Enhanced Type Validation

The type validation system in `tools/validators.py` has been significantly enhanced to support complex nested types and provide more robust validation.

### Complex Nested Type Support

The validator can now handle deeply nested type structures including:

- Nested dictionaries with specified key and value types
- Lists of complex objects
- Sets and tuples with element type checking
- Arbitrary nesting levels

```python
from typing import Dict, List, Any, Union
from lluminary.tools.validators import validate_return_type

@validate_return_type(Dict[str, List[Dict[str, Union[str, int]]]])
def get_users() -> Dict[str, List[Dict[str, Union[str, int]]]]:
    """Function that returns a nested data structure."""
    return {
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
    }
```

### Union and Optional Type Handling

The validator properly handles Union and Optional types including:

- Proper validation of `Union[Type1, Type2]` with correct type checking
- Support for `Optional[Type]` equivalent to `Union[Type, None]`
- Nested unions within complex data structures

```python
from typing import Optional, Union, List, Dict
from lluminary.tools.validators import validate_params

@validate_params(Optional[str], data=Dict[str, Union[str, int, List[str]]])
def process_data(query: Optional[str], data: Dict[str, Union[str, int, List[str]]]):
    """Function with Union and Optional parameter types."""
    # Function implementation
    pass
```

### JSON Serialization Validation

Enhanced validation for JSON serialization ensures that:

- All objects are properly serializable in nested structures
- Non-serializable objects are detected early with clear error messages
- All dictionary keys are strings (JSON requirement)

```python
from lluminary.tools.validators import json_serializable

@json_serializable
def get_config() -> Dict[str, Any]:
    """Return configuration that must be JSON serializable."""
    return {
        "name": "config1",
        "values": [1, 2, 3],
        "settings": {
            "enabled": True,
            "parameters": {
                "timeout": 30,
                "retries": 3
            }
        }
    }
```

### Usage Examples

#### Validating Complex Return Types

```python
from typing import Dict, List, Union, Any
from lluminary.tools.validators import validate_return_type, validate_tool

@validate_tool
@validate_return_type(Dict[str, List[Dict[str, Any]]])
def search_documents(query: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search documents and return structured results.

    Args:
        query: Search query string

    Returns:
        Dict containing search results
    """
    # Implementation
    return {
        "results": [
            {"title": "Document 1", "score": 0.95, "content": "..."},
            {"title": "Document 2", "score": 0.83, "content": "..."}
        ]
    }
```

#### Validating Parameter Types

```python
from typing import Dict, List, Optional, Union
from lluminary.tools.validators import validate_params

@validate_params(str, Optional[List[str]], config=Dict[str, Union[str, int, bool]])
def process_query(
    query: str,
    filters: Optional[List[str]] = None,
    config: Dict[str, Union[str, int, bool]] = None
) -> Dict[str, Any]:
    """Process a search query with optional filters and configuration."""
    # Implementation
    pass
```

## Classification Components

The classification system provides a flexible framework for categorizing messages using LLMs. The system consists of three main components that work together to provide comprehensive classification capabilities.

### Core Components

1. **ClassificationConfig** (`models/classification/config.py`)
   - Manages classification configuration including categories, examples, and metadata
   - Handles serialization/deserialization of configurations to/from JSON
   - Provides validation for configuration integrity
   - Supports configuration library management

2. **Classifier** (`models/classification/classifier.py`)
   - Implements the classification logic using LLM providers
   - Loads system prompts from YAML files
   - Processes classification responses
   - Converts numeric selections to category names

3. **Validators** (`models/classification/validators.py`)
   - Validates LLM classification responses
   - Ensures selections are within valid range
   - Handles multi-category selection validation
   - Provides informative error messages for invalid responses

### Comprehensive Testing

All classification components now have comprehensive test coverage (90%+):

- **Config Tests** cover:
  - Configuration initialization and validation
  - Category and example validation
  - Configuration serialization and deserialization
  - File operations (loading/saving)
  - Classification library management

- **Validator Tests** cover:
  - Response validation
  - Error handling for invalid responses
  - Multi-category selection validation
  - Format validation

- **Classifier Tests** cover:
  - Basic classification functionality
  - Classification with examples
  - Multiple category selection
  - Custom system prompt support
  - Error handling
  - Result processing
  - Category name conversion

All tests use the `@pytest.mark.classification` marker for better organization and selective test execution.

### Classification Usage Examples

#### Basic Classification

```python
from lluminary.handler import LLMHandler
from lluminary.models.classification.config import ClassificationConfig

# Create a classification configuration
config = ClassificationConfig(
    name="sentiment",
    description="Sentiment analysis configuration",
    categories={
        "positive": "Content that expresses positive sentiment",
        "neutral": "Content that is factual or neutral in tone",
        "negative": "Content that expresses negative sentiment"
    },
    examples=[
        {
            "user_input": "I love this product!",
            "doc_str": "This expresses clear positive sentiment",
            "selection": "positive"
        },
        {
            "user_input": "This product is terrible.",
            "doc_str": "This expresses clear negative sentiment",
            "selection": "negative"
        }
    ],
    max_options=1
)

# Initialize handler
handler = LLMHandler(provider="anthropic", model="claude-3-sonnet-20240229")

# Messages to classify
messages = [
    {"role": "user", "content": "The weather today is quite nice."}
]

# Perform classification
categories, usage = handler.llm.classify(
    messages=messages,
    categories=config.categories,
    examples=config.examples,
    max_options=config.max_options
)

print(f"Classification: {categories}")
print(f"Token usage: {usage}")
```

#### Multi-Category Classification

```python
from lluminary.handler import LLMHandler
from lluminary.models.classification.config import ClassificationConfig

# Create a multi-category classification config
topic_config = ClassificationConfig(
    name="topics",
    description="Message topic classification",
    categories={
        "technology": "Content about technology, computers, software",
        "science": "Content about scientific discoveries, research",
        "business": "Content about business, finance, economics",
        "health": "Content about health, medicine, wellness",
        "politics": "Content about politics, government, policy"
    },
    max_options=3  # Allow up to 3 topics to be selected
)

handler = LLMHandler(provider="anthropic", model="claude-3-sonnet-20240229")

# Classify a message that might belong to multiple categories
message = [
    {"role": "user", "content": "The new AI startup announced a healthcare platform that uses machine learning to identify potential drug interactions, which could revolutionize personalized medicine and attract significant investment."}
]

topics, usage = handler.llm.classify(
    messages=message,
    categories=topic_config.categories,
    max_options=topic_config.max_options
)

print(f"Topics identified: {topics}")
print(f"Token usage: {usage}")
```

## CLI Components

The CLI components in `cli/classify.py` provide a command-line interface for classification management.

### Classification Management Commands

The CLI includes the following commands:

- `list_configs`: List available classification configurations
- `validate`: Validate a classification configuration file
- `test`: Test a classification configuration with a message
- `create`: Create a new classification configuration interactively

### Command Usage

#### Listing Configurations

```bash
python -m lluminary.cli.classify list_configs /path/to/config/directory
```

This command displays all available classification configurations in the specified directory, including their names, descriptions, categories, and metadata.

#### Validating a Configuration

```bash
python -m lluminary.cli.classify validate /path/to/config.json
```

This command validates a classification configuration file, checking for required fields, proper structure, and internal consistency. It displays a summary of the configuration if valid.

#### Testing a Classification

```bash
python -m lluminary.cli.classify test /path/to/config.json "Test message" --model claude-haiku-3.5
```

This command tests a classification configuration against a message, returning the classified categories and usage statistics.

Available options:
- `--model`: The model to use for classification (default: "claude-sonnet-3.5")
- `--system-prompt`: Optional system prompt to override the default

#### Creating a New Configuration

```bash
python -m lluminary.cli.classify create /path/to/output.json
```

This command guides you through creating a new classification configuration interactively, prompting for:
- Configuration name and description
- Categories with descriptions
- Example classifications (optional)
- Maximum number of categories to select
- Metadata (author, version, creation date, tags)

### Testing and Validation

The CLI components now have comprehensive unit tests ensuring:

- Proper handling of valid configurations
- Appropriate error handling for invalid configurations
- Correct interaction with the classification system
- Proper metadata management

All CLI commands are tested with both valid and invalid inputs, ensuring robust behavior in all scenarios.

## Provider Implementation Improvements

The provider implementations have been significantly enhanced with improved type safety, error handling, and consistent interfaces.

### Type Safety Enhancements

All provider implementation files have received comprehensive type safety improvements:

- **Fixed Collection Type Annotations**: Replaced generic `list` and `dict` references with properly typed `List[T]` and `Dict[K, V]`.
- **Added Optional Type Annotations**: Parameters with default `None` values now use `Optional[T]` for proper type hinting.
- **Fixed Method Signatures**: Ensured all methods match their parent class contracts with compatible signatures.
- **Proper Return Types**: Added correct return type annotations, especially for generators and streaming methods.
- **Enhanced Error Type Handling**: Improved error handling with properly typed exceptions.

**Example improvements:**
```python
# Before
def stream_generate(self, event_id, system_prompt, messages, max_tokens=1000,
                   temp=0.0, tools=None):
    # Implementation...

# After
def stream_generate(
    self,
    event_id: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1000,
    temp: float = 0.0,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    # Implementation...
```

### Error Handling Improvements

Error handling has been significantly improved across provider implementations:

- **Standardized Error Mapping**: Each provider now has a comprehensive `_map_error` method that converts provider-specific errors to standard LLMHandler errors.
- **Exception Chaining**: Proper `from` clauses now preserve exception context for better debugging.
- **Improved Error Details**: Error messages now include more context and specific details about the failure.
- **Robust Type Checking**: Type-safe code prevents many potential runtime errors.

### Consistent Interface Implementation

All providers now implement a consistent interface with standardized methods:

- **Fixed Method Parameter Lists**: All overridden methods now have parameter lists compatible with the parent class.
- **Standardized Return Types**: Consistent return types across all providers for interchangeability.
- **Cost Calculation**: Improved null-safety in cost calculations with proper type handling.
- **Image Processing**: Enhanced image processing with proper PIL types and error handling.

The providers with completed type safety improvements include:
- `AnthropicLLM` - All mypy errors resolved
- `ProviderNameLLM` (template) - All mypy errors resolved
- `GoogleLLM` - All mypy errors resolved
- `BedrockLLM` - All mypy errors resolved
- `OpenAILLM` - All mypy errors resolved

## Related Documentation

- [API_REFERENCE](./API_REFERENCE.md) - Complete API reference for all components
- [ARCHITECTURE](./ARCHITECTURE.md) - System architecture and component relationships
- [TEST_COVERAGE](./TEST_COVERAGE.md) - Current test coverage reports and implementation guides
- [TUTORIALS](./TUTORIALS.md) - Step-by-step guides for using these components
- [ERROR_HANDLING](./development/ERROR_HANDLING.md) - Error handling integration details
