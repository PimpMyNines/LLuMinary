# Error Mapping Guide

This document explains the standardized error mapping approach used across all LLuMinary providers, helping developers implement consistent error handling in new or existing components.

## Table of Contents

1. [Introduction](#introduction)
2. [The Error Mapping Pattern](#the-error-mapping-pattern)
3. [Provider-Specific Error Mapping](#provider-specific-error-mapping)
4. [Error Categories and Detection](#error-categories-and-detection)
5. [Response Context Extraction](#response-context-extraction)
6. [Error Details Structure](#error-details-structure)
7. [Implementation Examples](#implementation-examples)
8. [Testing Error Mappers](#testing-error-mappers)
9. [Related Documentation](#related-documentation)

## Introduction

Error mapping is the process of converting provider-specific exceptions into standardized LLuMinary error types. This approach ensures:

- **Consistency**: All providers use the same error types and formats
- **Detailed Context**: Error details include provider-specific information for debugging
- **Typed Errors**: Specific error types allow for targeted handling by client code
- **Improved Debugging**: Original errors are preserved through exception chaining

## The Error Mapping Pattern

Each provider implements a dedicated error mapping method with this signature:

```python
def _map_provider_error(self, error: Exception, response: Optional[Any] = None) -> LLMError:
    """
    Map provider-specific errors to standard LLM error types.

    Args:
        error: The original exception from the provider
        response: Optional HTTP response object for additional context

    Returns:
        Standardized LLM exception with appropriate type and details
    """
```

This method is responsible for:

1. Analyzing exception details and response context
2. Categorizing the error into appropriate LLuMinary error types
3. Extracting relevant details for debugging
4. Creating a standardized error instance with consistent format
5. Preserving the original error message and context

## Provider-Specific Error Mapping

Each provider has unique error patterns that need to be recognized:

### OpenAI
- Status code (401): Authentication errors
- Status code (429): Rate limit errors
- Error types: `openai.APIError`, `openai.AuthenticationError`, etc.
- Error messages: Look for terms like "api_key", "rate_limit", etc.

### Anthropic
- Error types: `anthropic.APIError`, `anthropic.AuthenticationError`, etc.
- Error codes: "invalid_api_key", "rate_limit_exceeded", etc.
- Response error types: Look for keys like `error_type`, `status_code`

### Google
- Error types from Vertex AI
- Error codes from HTTP responses
- Model validation errors

### Bedrock
- AWS-specific error codes: "ThrottlingException", "ValidationException", etc.
- ClientError structure with specific error codes
- Region/credential-specific error formats

## Error Categories and Detection

Here's how to detect and map common error types:

### Authentication Errors
```python
# Detection patterns
auth_terms = ["api key", "auth", "unauthorized", "authentication", "permission"]
is_auth_error = any(term in error_str for term in auth_terms)

# Status code check
is_auth_error = is_auth_error or (status_code in (401, 403))

# Return appropriate error
if is_auth_error:
    return LLMAuthenticationError(
        message=f"{provider} authentication failed: {str(error)}",
        provider=provider,
        details=details
    )
```

### Rate Limit Errors
```python
# Detection patterns
rate_terms = ["rate limit", "too many requests", "quota", "exceeded"]
is_rate_limit = any(term in error_str for term in rate_terms)

# Status code check
is_rate_limit = is_rate_limit or (status_code == 429)

# Return appropriate error with retry-after if available
if is_rate_limit:
    return LLMRateLimitError(
        message=f"{provider} rate limit exceeded: {str(error)}",
        provider=provider,
        retry_after=details.get("retry_after", 60),
        details=details
    )
```

### Service Unavailable Errors
```python
# Detection patterns
service_terms = ["service unavailable", "server error", "internal error"]
is_service_error = any(term in error_str for term in service_terms)

# Status code check
is_service_error = is_service_error or (status_code and 500 <= status_code < 600)

# Return appropriate error
if is_service_error:
    return LLMServiceUnavailableError(
        message=f"{provider} service unavailable: {str(error)}",
        provider=provider,
        details=details
    )
```

### Content Policy Errors
```python
# Detection patterns
content_terms = ["content policy", "moderation", "inappropriate", "violates"]
is_content_error = any(term in error_str for term in content_terms)

# Return appropriate error
if is_content_error:
    return LLMContentError(
        message=f"{provider} content policy violation: {str(error)}",
        provider=provider,
        details=details
    )
```

## Response Context Extraction

A crucial part of error mapping is extracting useful context from response objects:

```python
def _extract_response_details(self, response: Any, details: Dict[str, Any]) -> None:
    """Extract useful details from response objects."""
    # Extract status code
    if hasattr(response, "status_code"):
        status_code = response.status_code
        if isinstance(status_code, int):
            details["status_code"] = status_code

    # Extract headers for retry-after
    if hasattr(response, "headers"):
        headers = response.headers
        if isinstance(headers, dict) or hasattr(headers, "get"):
            retry_header = headers.get("retry-after")
            if retry_header and isinstance(retry_header, str):
                try:
                    details["retry_after"] = int(retry_header)
                except ValueError:
                    details["retry_after_raw"] = retry_header

    # Extract JSON response data
    if hasattr(response, "json"):
        try:
            response_data = response.json()
            if isinstance(response_data, dict):
                details["response_data"] = response_data
        except Exception:
            # If JSON parsing fails, use text response
            if hasattr(response, "text"):
                details["response_text"] = response.text
```

## Error Details Structure

Error details should include a consistent set of fields:

```python
# Base error details
details = {
    "error": str(error),
    "error_type": type(error).__name__
}

# Add status code if available
if status_code is not None:
    details["status_code"] = status_code

# Add request ID if available
request_id = self._extract_request_id(response)
if request_id:
    details["request_id"] = request_id

# Add response data if available
if response_data:
    details["response_data"] = response_data

# Add model details if relevant
if hasattr(self, "model"):
    details["model"] = self.model
```

## Implementation Examples

### Complete Error Mapper Example

```python
def _map_provider_error(self, error: Exception, response: Optional[Any] = None) -> LLMError:
    """Map provider-specific errors to standard LLM error types."""
    provider = self.provider_name
    error_str = str(error).lower()
    details = {"error": str(error), "error_type": type(error).__name__}

    # Get status code and additional details
    status_code = None
    retry_after = None

    if response is not None:
        # Extract response details
        if hasattr(response, "status_code"):
            status_code = response.status_code
            details["status_code"] = status_code

        # Extract headers
        if hasattr(response, "headers"):
            headers = response.headers
            if headers and (isinstance(headers, dict) or hasattr(headers, "get")):
                retry_header = headers.get("retry-after")
                if retry_header and isinstance(retry_header, str):
                    try:
                        retry_after = int(retry_header)
                        details["retry_after"] = retry_after
                    except ValueError:
                        details["retry_after_raw"] = retry_header

        # Extract JSON data
        if hasattr(response, "json"):
            try:
                response_data = response.json()
                if isinstance(response_data, dict):
                    details["response_data"] = response_data
            except Exception:
                if hasattr(response, "text"):
                    details["response_text"] = response.text

    # Authentication errors
    if (
        "api key" in error_str
        or "auth" in error_str
        or "unauthorized" in error_str
        or "authentication" in error_str
        or (status_code in (401, 403))
    ):
        return LLMAuthenticationError(
            message=f"{provider} authentication failed: {str(error)}",
            provider=provider,
            details=details
        )

    # Rate limit errors
    if (
        "rate limit" in error_str
        or "too many requests" in error_str
        or "quota" in error_str
        or (status_code == 429)
    ):
        return LLMRateLimitError(
            message=f"{provider} rate limit exceeded: {str(error)}",
            provider=provider,
            retry_after=retry_after or 60,  # Default to 60 seconds
            details=details
        )

    # Service unavailable errors
    if (
        "service unavailable" in error_str
        or "server error" in error_str
        or "internal error" in error_str
        or (status_code and 500 <= status_code < 600)
    ):
        return LLMServiceUnavailableError(
            message=f"{provider} service unavailable: {str(error)}",
            provider=provider,
            details=details
        )

    # Configuration errors
    if (
        "configuration" in error_str
        or "invalid parameter" in error_str
        or "validation" in error_str
    ):
        return LLMConfigurationError(
            message=f"Invalid configuration for {provider}: {str(error)}",
            provider=provider,
            details=details
        )

    # Content policy errors
    if (
        "content policy" in error_str
        or "moderation" in error_str
        or "inappropriate" in error_str
    ):
        return LLMContentError(
            message=f"{provider} content policy violation: {str(error)}",
            provider=provider,
            details=details
        )

    # Format errors
    if (
        "format" in error_str
        or "parsing" in error_str
        or "invalid json" in error_str
    ):
        return LLMFormatError(
            message=f"Format error in {provider} response: {str(error)}",
            provider=provider,
            details=details
        )

    # Tool errors
    if (
        "tool" in error_str
        or "function call" in error_str
    ):
        return LLMToolError(
            message=f"{provider} tool error: {str(error)}",
            provider=provider,
            details=details
        )

    # Default to generic provider error
    return LLMProviderError(
        message=f"{provider} API error: {str(error)}",
        provider=provider,
        details=details
    )
```

### Using the Error Mapper

```python
def _raw_generate(self, messages, **kwargs):
    """Generate text with comprehensive error handling."""
    try:
        # Normal processing code...
        response = self._client.generate(messages=messages, **kwargs)
        return self._process_response(response)
    except requests.exceptions.RequestException as e:
        # Handle HTTP request exceptions
        response = getattr(e, "response", None)
        raise self._map_provider_error(e, response) from e
    except SomeProviderError as e:
        # Handle provider-specific errors
        raise self._map_provider_error(e) from e
    except LLMError:
        # Re-raise LLM errors directly
        raise
    except Exception as e:
        # Catch-all for unexpected errors
        raise self._map_provider_error(e) from e
```

## Testing Error Mappers

Testing error mappers thoroughly is essential:

```python
@pytest.mark.parametrize("error_text,status_code,expected_type", [
    ("Invalid API key", 401, LLMAuthenticationError),
    ("Too many requests", 429, LLMRateLimitError),
    ("Service unavailable", 503, LLMServiceUnavailableError),
    ("Invalid parameter", 400, LLMConfigurationError),
    ("Content policy violation", 400, LLMContentError),
    ("Unknown error", None, LLMProviderError),
])
def test_error_mapper(error_text, status_code, expected_type):
    """Test error mapping for different error types."""
    provider = ProviderLLM(api_key="test")

    # Create test error
    test_error = Exception(error_text)

    # Create mock response if status code is provided
    response = None
    if status_code:
        response = Mock()
        response.status_code = status_code

    # Map the error
    mapped_error = provider._map_provider_error(test_error, response)

    # Verify mapping
    assert isinstance(mapped_error, expected_type)
    assert mapped_error.provider == provider.provider_name
    assert error_text in mapped_error.message

    # Verify details
    if status_code:
        assert mapped_error.details.get("status_code") == status_code
```

## Related Documentation

- [ERROR_HANDLING.md](ERROR_HANDLING.md) - General error handling guidelines
- [ERROR_HANDLING_BEST_PRACTICES.md](ERROR_HANDLING_BEST_PRACTICES.md) - Best practices for error handling
- [OPENAI_ERROR_HANDLING_IMPLEMENTATION.md](OPENAI_ERROR_HANDLING_IMPLEMENTATION.md) - OpenAI-specific implementation
- [ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION.md](ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION.md) - Anthropic-specific implementation
- [GOOGLE_ERROR_HANDLING_IMPLEMENTATION.md](GOOGLE_ERROR_HANDLING_IMPLEMENTATION.md) - Google-specific implementation
- [BEDROCK_ERROR_HANDLING_IMPLEMENTATION.md](BEDROCK_ERROR_HANDLING_IMPLEMENTATION.md) - Bedrock-specific implementation
