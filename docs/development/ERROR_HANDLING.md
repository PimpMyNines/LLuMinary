# ERROR HANDLING GUIDELINES

## Overview

This document outlines the standard approach for error handling across all LLuMinary components, with special focus on provider implementations. It serves as a reference for developers implementing error handling in new or existing components.

## Table of Contents

- [Overview](#overview)
- [Error Hierarchy](#error-hierarchy)
- [Key Error Types](#key-error-types)
- [Provider Error Handling Standard](#provider-error-handling-standard)
- [Common Error Patterns by Provider](#common-error-patterns-by-provider)
- [Implementation Checklist](#implementation-checklist)
- [Example Implementation](#example-implementation)
- [Testing Error Handling](#testing-error-handling)
- [Related Documentation](#related-documentation)

## Error Hierarchy

LLuMinary has a well-defined error hierarchy to categorize different types of errors:

```
LLuMinaryError (base class)
├── ProviderError (provider-specific issues)
└── LLMMistake (LLM response issues)
    ├── FormatError
    ├── ContentError
    ├── ToolError
    └── ThinkingError
```

## Key Error Types

1. **LLuMinaryError**: Base exception class for all errors within the library.

2. **ProviderError**: For provider-specific issues like authentication, API errors, configuration problems.
   - Used for errors where retrying with the same parameters is unlikely to succeed
   - Includes provider name and detailed error information
   - Examples: API key errors, model validation errors, service unavailability

3. **LLMMistake**: For issues with LLM responses that might be recoverable.
   - Used when a response needs correction or a retry might help
   - Includes error type and is_recoverable flag
   - Examples: Format errors, content validation failures, tool execution issues

## Provider Error Handling Standard

All provider implementations in LLuMinary must follow these standards:

### 1. Use Custom Exception Types

Always use the custom exception types from `exceptions.py` instead of generic exceptions:

```python
# INCORRECT
try:
    # API call
except Exception as e:
    raise Exception(f"Failed to call API: {str(e)}")

# CORRECT
from ..exceptions import ProviderError

try:
    # API call
except Exception as e:
    raise ProviderError(
        message=f"Failed to call API: {str(e)}",
        provider="openai",
        details={"original_error": str(e), "request_id": request_id}
    )
```

### 2. Error Categorization

Categorize errors properly based on their nature:

- **Authentication errors**: Use `ProviderError` with details about credential issues
- **Rate limiting**: Use `ProviderError` with retry information in details
- **Invalid parameters**: Use `ProviderError` with validation details
- **API errors**: Use `ProviderError` with API-specific error codes
- **Response parsing/validation**: Use `LLMMistake` with appropriate error_type

### 3. Consistent Error Format

Include consistent information in error messages:

- Provider name
- Operation being performed
- Original error message
- Relevant context (request ID, model name, etc.)

### 4. Map Provider-Specific Errors

Each provider should implement a method to map provider-specific error codes to LLuMinary exceptions:

```python
def _map_api_error(self, error):
    """Map provider API errors to appropriate LLuMinary exceptions."""
    if "invalid_api_key" in str(error):
        return ProviderError(
            message="Authentication failed: Invalid API key",
            provider="openai",
            details={"original_error": str(error)}
        )
    elif "rate_limit" in str(error):
        return ProviderError(
            message="Rate limit exceeded",
            provider="openai",
            details={
                "original_error": str(error),
                "retry_after": error.headers.get("retry-after", "60")
            }
        )
    # etc.
```

### 5. Implement Retry Logic

For recoverable errors like rate limiting, implement standardized retry logic:

```python
def _call_with_retry(self, func, *args, max_retries=3, backoff_factor=1.5, **kwargs):
    """Call API function with exponential backoff retry logic."""
    retry_count = 0
    last_exception = None

    while retry_count < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if this is a retryable error
            if "rate_limit" in str(e) or "timeout" in str(e):
                retry_count += 1
                wait_time = backoff_factor ** retry_count
                time.sleep(wait_time)
                last_exception = e
            else:
                # Non-retryable error, raise immediately
                raise self._map_api_error(e)

    # If we've exhausted retries
    raise ProviderError(
        message=f"Failed after {max_retries} retries",
        provider=self.__class__.__name__.replace("LLM", ""),
        details={"original_error": str(last_exception)}
    )
```

### 6. Recovery Mechanism

Implement proper recovery behavior for LLMMistake errors:

```python
# In base.py LLM.generate method:
except LLMMistake as e:
    attempts += 1
    cumulative_usage["retry_count"] = attempts

    # Add failed response and error to messages for context
    working_messages.extend([
        {"message_type": "ai", "message": raw_response},
        {"message_type": "human", "message": f"Error: {str(e)}. Please fix the issue."}
    ])
```

## Common Error Patterns by Provider

### OpenAI
- Invalid authentication: Use API key validation with fallbacks
- Rate limiting: Handle 429 errors with Retry-After header
- Server errors: Handle 5xx errors with appropriate backoff
- Validation errors: Handle 400 errors with detailed error messages

### Anthropic
- Invalid authentication: Use API key validation with fallbacks
- Rate limiting: Handle 429 errors with exponential backoff
- Content policy errors: Map to specific LLMMistake types

### Google
- Authentication errors: Handle credential validation with proper error messages
- Quota limits: Handle quota-exceeded errors with appropriate retry logic
- Permission errors: Map to ProviderError with details

### Bedrock
- AWS authentication errors: Map to ProviderError with AWS-specific details
- Service quotas: Handle throttling exceptions with exponential backoff
- Model access issues: Handle authorization errors with helpful messages

## Implementation Checklist

When implementing error handling in a provider:

1. ✅ Import custom exception types from exceptions.py
2. ✅ Use ProviderError for provider-specific issues
3. ✅ Use LLMMistake for response validation/parsing issues
4. ✅ Implement error mapping for provider-specific errors
5. ✅ Add retry logic with exponential backoff for rate limiting
6. ✅ Include detailed context in error messages
7. ✅ Ensure all public methods use proper error handling
8. ✅ Add tests for various error conditions

## Example Implementation

```python
from ..exceptions import LLMMistake, ProviderError

class OpenAILLM(LLM):
    # ...

    def auth(self):
        """Authenticate with OpenAI API."""
        try:
            # Authentication code
        except Exception as e:
            raise ProviderError(
                message=f"OpenAI authentication failed: {str(e)}",
                provider="openai",
                details={"original_error": str(e)}
            )

    def _raw_generate(self, event_id, system_prompt, messages, max_tokens=1000, temp=0.0,
                     tools=None, thinking_budget=None):
        """Generate text with proper error handling."""
        try:
            # Format messages
            formatted_messages = self._format_messages_for_model(messages)

            # Make API call with retry logic
            response = self._call_api_with_retry(
                formatted_messages, max_tokens, temp, tools
            )

            # Extract content and validate
            content = self._extract_content(response)
            if not content.strip():
                raise LLMMistake(
                    message="Empty response from model",
                    error_type="content",
                    provider="openai"
                )

            # Calculate usage statistics
            usage = self._calculate_usage(response, messages)

            return content, usage, messages

        except openai.APIError as e:
            # Map provider-specific error to our exception types
            mapped_error = self._map_openai_error(e)
            raise mapped_error

    def _map_openai_error(self, error):
        """Map OpenAI API errors to appropriate LLMHandler exceptions."""
        if "invalid_api_key" in str(error):
            return ProviderError(
                message="Authentication failed: Invalid API key",
                provider="openai",
                details={"original_error": str(error)}
            )
        elif "rate_limit" in str(error):
            return ProviderError(
                message="Rate limit exceeded",
                provider="openai",
                details={
                    "original_error": str(error),
                    "retry_after": getattr(error, "headers", {}).get("retry-after", "60")
                }
            )
        # General fallback
        return ProviderError(
            message=f"OpenAI API error: {str(error)}",
            provider="openai",
            details={"original_error": str(error)}
        )
```

## Testing Error Handling

All error handling should be thoroughly tested:

1. Test authentication errors
2. Test rate limiting and retry behavior
3. Test proper mapping of provider-specific errors
4. Test recovery from LLMMistake errors
5. Test proper error propagation

## Related Documentation

- [API_REFERENCE](../API_REFERENCE.md) - Complete API reference including error handling
- [ARCHITECTURE](../ARCHITECTURE.md) - System architecture including error handling architecture
- [ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION](./ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION.md) - Anthropic-specific error handling
- [OPENAI_ERROR_HANDLING_IMPLEMENTATION](./OPENAI_ERROR_HANDLING_IMPLEMENTATION.md) - OpenAI-specific error handling
- [GOOGLE_ERROR_HANDLING_IMPLEMENTATION](./GOOGLE_ERROR_HANDLING_IMPLEMENTATION.md) - Google-specific error handling
- [BEDROCK_ERROR_HANDLING_IMPLEMENTATION](./BEDROCK_ERROR_HANDLING_IMPLEMENTATION.md) - Bedrock-specific error handling
