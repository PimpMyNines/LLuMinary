# Type Safety and Code Quality Guidelines

This document provides guidelines for maintaining type safety and code quality in the LLMHandler codebase.

## Type Safety Practices

### 1. Type Annotations

- Always use explicit type annotations for:
  - Function parameters
  - Function return types
  - Class variables
  - Complex variables within functions

```python
def process_data(messages: List[Dict[str, Any]], max_tokens: int = 1000) -> Tuple[str, Dict[str, Any]]:
    result: Dict[str, Any] = {}
    # Implementation...
    return response_text, result
```

### 2. Collection Type Handling

- Use properly parameterized collection types:
  - `List[T]` instead of `list`
  - `Dict[K, V]` instead of `dict`
  - `Optional[T]` for parameters that can be None

```python
def get_user_data(user_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
    # Implementation...
```

### 3. None Safety

- Always handle potential None values in math operations:

```python
# Good
value = some_dict.get("key", 0) or 0
result = int_value * float(value)

# Good - alternative approach
if value is not None:
    result = int_value * float(value)
else:
    result = 0
```

### 4. Provider Consistency

- Maintain parameter consistency across all provider implementations
- Follow the interface defined in the base LLM class
- Use the same field names for compatibility (e.g., "input" not "arguments")

## Code Review Plan

### 1. GitHub Actions Verification

- Ensure all GitHub Action workflows pass successfully:
  - Run the type checking workflow
  - Validate linting and formatting
  - Verify test success across all providers

```bash
# Local pre-check before pushing
python -m mypy src/llmhandler/
python -m pytest tests/unit/
```

### 2. Dependency Consistency

- Confirm all dependencies are properly declared in:
  - pyproject.toml
  - requirements.txt
  - GitHub Action workflows

### 3. Provider Interface Compliance

- Verify each provider consistently implements:
  - stream_generate - with proper tools parameter
  - _raw_generate - with proper typing
  - Tool handling - consistent field names
  - Error mapping - proper exception types

### 4. Documentation Updates

For any changes to provider interfaces:
1. Update API_REFERENCE.md with new parameter descriptions
2. Update examples to demonstrate proper usage
3. Ensure TEST_COVERAGE.md reflects current status

## Testing Requirements

All changes must include appropriate tests:

1. **Unit Tests**: For isolated component functionality
2. **Integration Tests**: For provider interactions
3. **Edge Cases**: Especially for None-value handling

```bash
# Run specific provider tests
python -m pytest tests/unit/test_openai*.py -v
python -m pytest tests/unit/test_anthropic*.py -v
```

## Type Checking Configuration

The mypy.ini file has been set up to handle third-party library issues:

```ini
# Provider-specific ignores
[mypy.src.llmhandler.models.providers.*]
ignore_errors = True

# Third-party libraries
[mypy.requests.*]
ignore_missing_imports = True

[mypy.PIL.*]
ignore_missing_imports = True

[mypy.openai.*]
ignore_missing_imports = True

[mypy.google.*]
ignore_missing_imports = True

[mypy.anthropic.*]
ignore_missing_imports = True

[mypy.boto3.*]
ignore_missing_imports = True
```

Any changes to mypy.ini should be carefully reviewed to maintain the balance between type safety and practicality with third-party libraries.