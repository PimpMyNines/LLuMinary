# LLuMinary Development Notes

## Project Overview
This project is a provider-agnostic LLM handler that supports multiple providers (OpenAI, Anthropic, Google, Cohere, and Bedrock) with a focus on reliability, error handling, and consistent behavior.

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

### 1. Standard Message Format
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
- **Google**: Format varies by model/endpoint
- **Cohere**: Special handling for reranking and embeddings

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
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

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
