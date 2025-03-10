# GOOGLE ERROR HANDLING IMPLEMENTATION

## Overview

This document outlines the comprehensive error handling implementation for the Google LLM provider in the LLuMinary library, detailing Google-specific error mapping strategies and implementation approaches.

## Table of Contents

- [Overview](#overview)
- [Implementation Approach](#implementation-approach)
- [Key Components](#key-components)
  - [Error Mapping Method](#1-error-mapping-method)
  - [Retry Mechanism](#2-retry-mechanism)
  - [Enhanced Authentication](#3-enhanced-authentication)
  - [Improved Image Processing](#4-improved-image-processing)
  - [Structured Exception Handling in Core Methods](#5-structured-exception-handling-in-core-methods)
- [Google-Specific Considerations](#google-specific-considerations)
  - [API Structure](#api-structure)
  - [Error Detection Patterns](#error-detection-patterns)
- [Testing Approach](#testing-approach)
- [Future Improvements](#future-improvements)
- [Integration with Handler](#integration-with-handler)
- [Related Documentation](#related-documentation)

## Implementation Approach

The Google provider error handling follows the same pattern established for OpenAI and Anthropic, with adaptations specific to Google's API behavior:

1. **Error Mapping System**: Translates Google-specific errors to our standardized error types
2. **Retry Mechanism**: Implements exponential backoff for transient errors
3. **Enhanced Authentication**: Adds environment variable fallback and better error information
4. **Improved Image Processing**: Detailed error handling for different image-related failures
5. **Structured Exception Handling**: Consistent try/except patterns throughout the code

## Key Components

### 1. Error Mapping Method

The `_map_google_error` method examines exception messages and types to determine the appropriate custom exception to raise. This ensures consistent error responses across all providers.

```python
def _map_google_error(self, error: Exception) -> Exception:
    """Map a Google API exception to an appropriate LLuMinary exception."""
    error_message = str(error).lower()
    error_type = type(error).__name__

    # Authentication errors
    if ("api key" in error_message or "credential" in error_message
        or "permission" in error_message):
        return AuthenticationError(...)

    # Rate limit errors
    if "rate limit" in error_message or "quota" in error_message:
        return RateLimitError(...)

    # Service availability errors
    if "unavailable" in error_message or "server error" in error_message:
        return ServiceUnavailableError(...)

    # etc...
```

### 2. Retry Mechanism

The `_call_with_retry` method implements an exponential backoff strategy for handling transient errors:

```python
def _call_with_retry(self, func, *args, max_retries=3, retry_delay=1, **kwargs):
    """Execute an API call with automatic retry for transient errors."""
    attempts = 0
    while attempts <= max_retries:
        try:
            return func(*args, **kwargs)
        except (RateLimitError, ServiceUnavailableError) as e:
            # Handle retry logic with exponential backoff
            # ...
```

### 3. Enhanced Authentication

Authentication now includes:
- Environment variable support (`GOOGLE_API_KEY`)
- AWS Secrets Manager fallback
- Detailed error reporting for configuration issues
- Special handling for different model versions (experimental models)

### 4. Improved Image Processing

The image processing error handling now includes:
- Detailed HTTP status code handling for URLs
- Image format validation
- Path existence checking for local files
- Content type verification
- Appropriate error classification (ContentError vs. LLMMistake)

### 5. Structured Exception Handling in Core Methods

Both `_raw_generate` and `stream_generate` methods implement a consistent structured approach to exception handling:

```python
try:
    # Main logic
except (AuthenticationError, RateLimitError, ServiceUnavailableError, LLMMistake):
    # Re-raise already mapped exceptions
    raise
except Exception as e:
    # Map unexpected errors
    mapped_error = self._map_google_error(e)
    raise mapped_error
```

## Google-Specific Considerations

### API Structure

Google's API has some unique characteristics that influenced the error handling:
- Usage metadata extraction requires careful handling as fields might be missing
- Function calls are returned differently than other providers
- Different API versions (v1alpha for experimental models)
- Safety settings and content filtering are more proactive

### Error Detection Patterns

Google's API doesn't always return clear error types, so we needed to rely on error message string detection:
- Rate limit errors can contain "quota", "rate limit", "resource exhausted"
- Authentication errors might contain "credentials", "api key", "permission"
- Service errors often include "unavailable", "timeout", "error"

## Testing Approach

The error handling implementation is tested in `test_google_error_handling.py`, which includes tests for:

1. Error mapping from various error messages to correct exception types
2. Authentication with environment variables and Secrets Manager
3. Retry mechanism with exponential backoff
4. Image processing errors (bad URLs, corrupt images, permission issues)
5. Streaming error handling

## Future Improvements

Future error handling enhancements could include:

1. Enhanced token counting during errors (for more accurate billing)
2. More granular error maps for specific Google API status codes
3. Automatic retry count adjustment based on error patterns
4. Add support for more specific Google error types as they evolve
5. Improved handling of streaming connection errors

## Integration with Handler

This implementation works with the existing error handler in the main LLuMinary handler class, enabling:
1. Automatic provider fallback when appropriate
2. Consistent error reporting across providers
3. Standardized retry strategies
4. Detailed error logging

## Related Documentation

- [ERROR_HANDLING](./ERROR_HANDLING.md) - General error handling guidelines
- [MODELS](./MODELS.md) - Model implementation details
- [PROVIDER_TESTING](./PROVIDER_TESTING.md) - Provider testing guidelines
- [API_REFERENCE](../API_REFERENCE.md) - Complete API reference
- [ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION](./ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION.md) - Anthropic error handling
- [BEDROCK_ERROR_HANDLING_IMPLEMENTATION](./BEDROCK_ERROR_HANDLING_IMPLEMENTATION.md) - Bedrock error handling
- [OPENAI_ERROR_HANDLING_IMPLEMENTATION](./OPENAI_ERROR_HANDLING_IMPLEMENTATION.md) - OpenAI error handling
