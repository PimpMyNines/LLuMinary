# IMPLEMENTATION NOTES

## Overview

This document provides essential implementation guidance for the LLuMinary library. The library is a provider-agnostic LLM handler that supports multiple providers (OpenAI, Anthropic, Google, Cohere, and Bedrock) with a focus on reliability, error handling, and consistent behavior.

## Table of Contents

- [Overview](#overview)
- [Key Architectural Decisions](#key-architectural-decisions)
- [Implementation Details](#implementation-details)
- [Common Pitfalls](#common-pitfalls)
- [Future Considerations](#future-considerations)
- [Critical Notes](#critical-notes)
- [Development Environment](#development-environment)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Related Documentation](#related-documentation)

## Key Architectural Decisions

### 1. Synchronous Operation
- **IMPORTANT**: All operations must be synchronous
- No async/await functionality should be added
- Previous attempts to add async have been removed
- Direct, blocking API calls are preferred

### 2. Error Handling Strategy
- Use `LLMMistake` for expected errors
- Automatic retry mechanism with feedback loop
- Deep copy messages to prevent modification
- Add failed responses to conversation history

### 3. Response Processing
- XML-based response format for classifications
- Custom validation functions for responses
- Strict output format requirements
- No response format changes without tests

### 4. Testing Requirements
- All features must have comprehensive tests
- Test both success and failure paths
- Test with all supported providers
- Run full test suite before commits

## Implementation Details

### 1. Message Format
```python
{
    "message_type": "human" | "ai",
    "message": str,
    "image_paths": List[str],
    "image_urls": List[str]
}
```

### 2. Error Correction System
```python
def generate(
    self,
    event_id: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    result_processing_function: Optional[Callable] = None,
    retry_limit: int = 3
) -> Tuple[Any, Dict[str, Any]]:
    """
    Key features:
    - Deep copy messages
    - Track cumulative usage
    - Add errors to conversation
    - Return processed response
    """
```

### 3. Provider-Specific Requirements
- **OpenAI**: Base64 JPEG with data URI
- **Anthropic**: Base64 JPEG with media type
- **Bedrock**: Raw PNG bytes with type field

## Common Pitfalls

### 1. Message Handling
- ❌ Don't modify original message lists
- ❌ Don't assume message format consistency
- ✅ Always deep copy messages
- ✅ Validate message format

### 2. Error Handling
- ❌ Don't use generic exceptions
- ❌ Don't ignore provider errors
- ✅ Use `LLMMistake` for expected errors
- ✅ Include context in error messages

### 3. Response Processing
- ❌ Don't trust raw responses
- ❌ Don't modify response formats without tests
- ✅ Use strict validation
- ✅ Handle provider-specific formats

### 4. Testing
- ❌ Don't skip provider-specific tests
- ❌ Don't ignore error paths
- ✅ Test all providers
- ✅ Include integration tests

## Future Considerations

### 1. Performance
- Consider caching mechanisms
- Optimize image processing
- Monitor token usage patterns
- Track retry statistics

### 2. Reliability
- Improve error detection
- Enhance retry strategies
- Add usage monitoring
- Implement circuit breakers

### 3. Extensibility
- Keep provider interface consistent
- Document extension points
- Maintain backward compatibility
- Consider versioning strategy

## Critical Notes
1. Always run full test suite
2. Keep operations synchronous
3. Use proper error types
4. Document changes thoroughly
5. Test all providers

## Development Environment
```bash
# Setup
source .venv/bin/activate
pip install -r requirements.txt

# Testing
python -m pytest tests/ -v
```

## Common Issues and Solutions

### 1. Response Parsing
- Use strict XML parsing
- Validate category numbers
- Handle provider variations
- Keep format consistent

### 2. Error Handling
- Use appropriate error types
- Include context in messages
- Track retry attempts
- Monitor usage across retries

### 3. Provider Differences
- Handle format variations
- Respect provider limits
- Convert image formats
- Track provider costs

Remember: This project prioritizes reliability and consistency over performance. Keep operations synchronous and maintain thorough testing.

## Related Documentation

- [API_REFERENCE](../API_REFERENCE.md) - Complete API reference for all components
- [ARCHITECTURE](../ARCHITECTURE.md) - System architecture and component relationships
- [MODELS](./MODELS.md) - Detailed information about model implementations
- [ERROR_HANDLING](./ERROR_HANDLING.md) - Error handling guidelines and implementation
- [PROVIDER_TESTING](./PROVIDER_TESTING.md) - Testing guidelines for provider implementations
- [TEST_COVERAGE](../TEST_COVERAGE.md) - Current test coverage status and goals
