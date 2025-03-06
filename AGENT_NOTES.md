# GitHub Actions Debugging Guide

This document identifies the current issues with GitHub Actions workflows and provides a detailed plan to fix them.

## Current Issues

Based on our analysis of the codebase, we've identified several categories of issues causing GitHub Actions to fail:

### 1. Type Checking (mypy) Errors

Running mypy on the base LLM and handler classes reveals multiple errors:

#### LLM Base Class Errors
- **Unreachable code**: Several code blocks are detected as unreachable
- **Undefined variables**: Many variables are accessed but not defined in scope
- **Type hints**: Some function return types are incorrect

#### Handler Class Errors
- **Constructor return value**: Missing return statement in the class constructor
- **LLM Class errors**: Attempts to access methods/attributes that aren't defined
- **Error initialization**: Incorrect parameters to LLMError constructor
- **Set.add() errors**: Incorrectly using the return value of set.add()

### 2. Style and Formatting Issues (ruff)

The ruff check reveals numerous style issues:

- **Line length**: Many lines exceed the 88 character limit
- **Error handling**: Need to use `raise ... from error` pattern
- **Unused variables**: Some defined variables are never used
- **Code readability**: Several blocks could be simplified

## Action Plan

### 1. Fix Type Checking Errors

#### Base LLM Class Fixes:
- Fix variable scope issues in methods
- Correct return type annotations
- Address unreachable code segments
- Fix function parameters

```python
# Example fix for unreachable code
def method_with_unreachable_code(self):
    # Original:
    if condition:
        return result
    # Unreachable code after return
    process_something()

    # Fixed:
    if condition:
        return result
    else:
        process_something()
```

#### Handler Class Fixes:
- Fix initialization pattern for LLMError-derived exceptions
- Add missing return statement to constructor
- Fix set.add() usage in duplicate removal

```python
# Example fix for set.add() issue
# Original:
seen = set()
providers_to_try = [p for p in providers_to_try if not (p in seen or seen.add(p))]

# Fixed:
seen = set()
providers_to_try = []
for p in initial_providers:
    if p not in seen:
        seen.add(p)
        providers_to_try.append(p)
```

### 2. Fix Style and Formatting Issues

- Refactor long lines to stay within 88 character limit
- Implement proper exception handling chaining
- Remove unused variable assignments
- Improve code readability with better formatting

```python
# Example fix for long lines:
# Original:
self.logger.info(f"Successfully initialized provider {provider} with model {model_name}")

# Fixed:
self.logger.info(
    f"Successfully initialized provider {provider} "
    f"with model {model_name}"
)
```

### 3. Update Provider Compatibility

- Ensure all providers implement the common interface methods
- Add proper attribute/method stubs to the LLM base class
- Update method signatures to match usage patterns

```python
# Example fix for missing method in base class:
# Add to LLM base class:
def supports_embeddings(self) -> bool:
    """Check if this provider supports embeddings."""
    return hasattr(self, 'embed') and callable(getattr(self, 'embed'))
```

### 4. Testing Strategy

For each fix:
1. Run mypy locally to verify type errors are resolved
2. Run ruff to verify style issues are fixed
3. Run unit tests to ensure functionality is preserved

```bash
# Commands to run for verification:
python -m mypy src/lluminary/models/base.py src/lluminary/handler.py
python -m ruff check src/lluminary/models/base.py src/lluminary/handler.py
python -m pytest tests/unit/test_base_llm.py tests/unit/test_handler.py
```

## Implementation Priority

1. **Critical Type Errors** - Fix issues preventing compilation
2. **Interface Consistency** - Ensure base classes define all required methods
3. **Exception Handling** - Update error handling patterns
4. **Style and Formatting** - Clean up code style issues

## Completion Criteria

The implementation will be considered complete when:
- All GitHub Actions workflows pass successfully
- All mypy errors are resolved
- All ruff warnings are addressed or explicitly ignored
- All unit tests pass across Python 3.8, 3.9, 3.10, and 3.11
- The package builds and installs successfully
