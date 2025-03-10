# Error Handling Best Practices

This document provides best practices and patterns for implementing error handling in LLuMinary components, particularly for provider implementations. It includes code examples, common patterns, and testing recommendations.

## Table of Contents

1. [Core Principles](#core-principles)
2. [Error Handler Implementation](#error-handler-implementation)
3. [Response Context Extraction](#response-context-extraction)
4. [Type-Safe Error Mapping](#type-safe-error-mapping)
5. [Exception Chaining](#exception-chaining)
6. [Retry Mechanisms](#retry-mechanisms)
7. [Testing Recommendations](#testing-recommendations)
8. [Real-World Examples](#real-world-examples)
9. [Related Documentation](#related-documentation)

## Core Principles

When implementing error handling in LLuMinary, follow these core principles:

1. **Standardization**: Use standardized error types and patterns across all providers
2. **Context Preservation**: Include detailed context in error details for debugging
3. **Type Safety**: Use appropriate exception types and ensure type-safe code
4. **Chained Exceptions**: Use `raise ... from e` to preserve original tracebacks
5. **Response Context**: Include response data and status codes in error details
6. **User-Friendly Messages**: Provide clear, actionable error messages
7. **Comprehensive Error Mapping**: Map all provider-specific errors to standard types

## Error Handler Implementation

Each provider should implement a dedicated error handler function following this pattern:

```python
def _map_provider_error(self, error: Exception, response: Optional[Any] = None) -> LLMError:
    """
    Map provider-specific errors to standardized LLM error types.

    Args:
        error: The original exception from the provider API
        response: Optional HTTP response object for additional context

    Returns:
        Standardized LLM exception with appropriate type and details
    """
    provider = self.provider_name  # Use consistent provider name
    error_str = str(error).lower()  # Lowercase for case-insensitive matching
    details = {"error": str(error), "error_type": type(error).__name__}

    # Extract context from response if available
    if response is not None:
        self._extract_response_details(response, details)

    # Map to specific error types based on error content
    # Authentication errors
    if self._is_auth_error(error_str):
        return LLMAuthenticationError(
            message=f"{provider} authentication failed: {str(error)}",
            provider=provider,
            details=details
        )

    # Rate limit errors
    if self._is_rate_limit_error(error_str, response):
        return LLMRateLimitError(
            message=f"{provider} rate limit exceeded: {str(error)}",
            provider=provider,
            retry_after=details.get("retry_after", 60),
            details=details
        )

    # Add mappings for other error types...

    # Default to generic provider error when no specific mapping applies
    return LLMProviderError(
        message=f"{provider} API error: {str(error)}",
        provider=provider,
        details=details
    )

def _is_auth_error(self, error_str: str) -> bool:
    """Check if error is related to authentication."""
    auth_terms = ["api key", "auth", "unauthorized", "authentication", "permission"]
    return any(term in error_str for term in auth_terms)

def _is_rate_limit_error(self, error_str: str, response: Optional[Any]) -> bool:
    """Check if error is related to rate limiting."""
    rate_terms = ["rate limit", "too many requests", "quota", "exceeded"]
    status_code = getattr(response, "status_code", None) if response else None

    return any(term in error_str for term in rate_terms) or status_code == 429
```

## Response Context Extraction

Extract meaningful context from response objects to provide detailed error information:

```python
def _extract_response_details(self, response: Any, details: Dict[str, Any]) -> None:
    """Extract useful details from response objects for error context."""
    # Extract status code
    if hasattr(response, "status_code"):
        status_code = response.status_code
        if isinstance(status_code, int):
            details["status_code"] = status_code

    # Extract headers, focusing on retry-after for rate limits
    if hasattr(response, "headers"):
        headers = response.headers
        if headers and (isinstance(headers, dict) or hasattr(headers, "get")):
            # Extract retry-after header
            retry_header = headers.get("retry-after")
            if retry_header and isinstance(retry_header, str):
                try:
                    details["retry_after"] = int(retry_header)
                except (ValueError, TypeError):
                    # If not a valid integer, store as string
                    details["retry_after_raw"] = retry_header

    # Try to extract JSON response data
    if hasattr(response, "json"):
        try:
            response_data = response.json()
            if isinstance(response_data, dict):
                # Only include response data if it's a dictionary
                details["response_data"] = response_data
        except Exception:
            # If JSON parsing fails, store raw text if available
            if hasattr(response, "text"):
                details["response_text"] = response.text
```

## Type-Safe Error Mapping

Ensure error mapping is type-safe by following these patterns:

1. **Use explicit type annotations**:
   ```python
   def _map_provider_error(self, error: Exception, response: Optional[Any] = None) -> LLMError:
   ```

2. **Use safe attribute access**:
   ```python
   # BAD
   status_code = response.status_code  # Can raise AttributeError

   # GOOD
   status_code = getattr(response, "status_code", None)
   ```

3. **Check types before operations**:
   ```python
   # BAD
   retry_after = headers.get("retry-after")
   retry_seconds = int(retry_after)  # Can raise ValueError or TypeError

   # GOOD
   retry_after = headers.get("retry-after") if headers else None
   if retry_after and isinstance(retry_after, str):
       try:
           retry_seconds = int(retry_after)
       except (ValueError, TypeError):
           retry_seconds = 60  # Default value
   ```

4. **Handle potential None values**:
   ```python
   # BAD
   error_data = response.json()  # response might be None

   # GOOD
   error_data = response.json() if response and hasattr(response, "json") else {}
   ```

## Exception Chaining

Always use exception chaining to preserve original tracebacks:

```python
try:
    # Code that might raise an exception
    result = api_client.call_api()
except SomeException as e:
    # Map the exception to a standardized type
    mapped_error = self._map_provider_error(e)
    # Raise with exception chaining to preserve original traceback
    raise mapped_error from e
```

This ensures:
1. The original exception is preserved in the traceback
2. Debugging is easier because you can see the original error
3. The error hierarchy is maintained

## Retry Mechanisms

Implement proper retry logic with exponential backoff:

```python
def _call_with_retry(
    self,
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    jitter: float = 0.1,
    **kwargs: Any
) -> T:
    """
    Call a function with exponential backoff retry logic.

    Args:
        func: The function to call
        *args: Positional arguments for the function
        max_retries: Maximum number of retries before giving up
        backoff_factor: Base factor for calculating delay
        jitter: Random jitter factor to add to delay
        **kwargs: Keyword arguments for the function

    Returns:
        The return value of the function

    Raises:
        LLMError: If all retries are exhausted
    """
    retries = 0
    last_error = None

    while retries <= max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            response = getattr(e, "response", None)
            mapped_error = self._map_provider_error(e, response)

            # Only retry for certain error types
            if isinstance(mapped_error, (LLMRateLimitError, LLMServiceUnavailableError,
                                        LLMConnectionError, LLMTimeoutError)):
                retries += 1

                if retries > max_retries:
                    # We've exhausted retries, raise the mapped error
                    mapped_error.details["retries_attempted"] = retries
                    raise mapped_error from e

                # Calculate delay with exponential backoff and jitter
                if isinstance(mapped_error, LLMRateLimitError) and mapped_error.retry_after:
                    # Use retry-after value if available
                    delay = mapped_error.retry_after
                else:
                    # Calculate exponential backoff
                    delay = backoff_factor ** retries

                # Add jitter to prevent thundering herd
                delay += random.uniform(0, jitter * delay)

                # Log the retry attempt
                logger.warning(
                    f"Retrying after error: {mapped_error.message}. "
                    f"Attempt {retries}/{max_retries} in {delay:.2f}s"
                )

                time.sleep(delay)
                last_error = e
            else:
                # Non-retryable error, raise immediately
                raise mapped_error from e

    # This should never be reached, but just in case
    raise LLMProviderError(
        message=f"Failed after {max_retries} retries",
        provider=self.provider_name,
        details={"error": str(last_error) if last_error else "Unknown error"}
    )
```

## Testing Recommendations

Follow these testing patterns for comprehensive error handling validation:

1. **Test All Error Types**:
   ```python
   @pytest.mark.parametrize("error_message,expected_type", [
       ("Invalid API key", LLMAuthenticationError),
       ("Rate limit exceeded", LLMRateLimitError),
       ("Service temporarily unavailable", LLMServiceUnavailableError),
       ("Invalid model parameter", LLMConfigurationError),
       ("Content policy violation", LLMContentError),
   ])
   def test_error_mapping(error_message, expected_type):
       provider = ProviderLLM(api_key="test")

       # Create test error
       test_error = Exception(error_message)

       # Map the error
       mapped_error = provider._map_provider_error(test_error)

       # Verify mapping
       assert isinstance(mapped_error, expected_type)
       assert error_message in mapped_error.message
       assert mapped_error.provider == provider.provider_name
   ```

2. **Test Response Detail Extraction**:
   ```python
   def test_response_detail_extraction():
       provider = ProviderLLM(api_key="test")

       # Create mock response with details
       mock_response = Mock()
       mock_response.status_code = 429
       mock_response.headers = {"retry-after": "30"}
       mock_response.json.return_value = {"error": {"message": "Rate limited", "code": "rate_limit"}}

       # Create test error with this response
       test_error = Exception("Rate limit exceeded")

       # Map the error with response
       mapped_error = provider._map_provider_error(test_error, mock_response)

       # Verify details extraction
       assert isinstance(mapped_error, LLMRateLimitError)
       assert mapped_error.retry_after == 30
       assert mapped_error.details.get("status_code") == 429
       assert "rate_limit" in str(mapped_error.details.get("response_data", {}))
   ```

3. **Test Retry Logic**:
   ```python
   def test_retry_logic():
       provider = ProviderLLM(api_key="test")

       # Mock the API client
       with patch.object(provider, "_client") as mock_client, \
            patch("time.sleep") as mock_sleep:

           # Configure mock to fail twice, then succeed
           mock_client.generate.side_effect = [
               Exception("Rate limit exceeded"),  # First call fails
               Exception("Rate limit exceeded"),  # Second call fails
               {"choices": [{"message": {"content": "Success"}}]}  # Third call succeeds
           ]

           # Call method with retry logic
           result = provider._raw_generate(
               messages=[{"role": "user", "content": "Hello"}]
           )

           # Verify behavior
           assert mock_client.generate.call_count == 3  # Should be called 3 times
           assert mock_sleep.call_count == 2  # Sleep should be called twice
           assert "Success" in result[0]  # Final result should contain success message
   ```

4. **Test Exception Chaining**:
   ```python
   def test_exception_chaining():
       provider = ProviderLLM(api_key="test")

       # Mock client to raise exception
       with patch.object(provider, "_client") as mock_client:
           original_exception = ValueError("API Error")
           mock_client.generate.side_effect = original_exception

           # Capture the raised exception
           with pytest.raises(LLMError) as excinfo:
               provider._raw_generate(messages=[{"role": "user", "content": "test"}])

           # Verify exception chaining
           assert excinfo.value.__cause__ is original_exception
   ```

## Real-World Examples

Here are examples of error handling from implemented providers:

### Authentication Error Handling

```python
def auth(self) -> bool:
    """Authenticate with the provider API."""
    provider = "provider_name"
    if not self.api_key:
        raise LLMAuthenticationError(
            message=f"{provider} authentication failed: No API key provided",
            provider=provider,
            details={"error": "No API key provided"}
        )

    try:
        # Verify API key by making a minimal API call
        self._client.models.list(api_key=self.api_key)
        return True
    except Exception as e:
        # Map to authentication error if it's not already a known error
        auth_error = self._map_provider_error(e)
        if not isinstance(auth_error, LLMAuthenticationError):
            # Force authentication error type for auth method
            auth_error = LLMAuthenticationError(
                message=f"Failed to authenticate with {provider} API: {str(e)}",
                provider=provider,
                details=auth_error.details if hasattr(auth_error, "details") else {"error": str(e)}
            )
        raise auth_error from e
```

### Rate Limit Handling

```python
try:
    response = self._client.completions.create(
        model=model,
        messages=formatted_messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response
except Exception as e:
    response_obj = getattr(e, "response", None)
    error = self._map_provider_error(e, response_obj)

    # Check if this is a rate limit error
    if isinstance(error, LLMRateLimitError):
        # Log the rate limit with retry information
        retry_seconds = error.retry_after or 60
        logger.warning(
            f"Rate limit hit. Waiting {retry_seconds}s before retry. "
            f"Request ID: {error.details.get('request_id', 'unknown')}"
        )
        # Consider implementing quota tracking here

    raise error from e
```

### API Response Validation

```python
def _validate_response(self, response: Dict[str, Any]) -> None:
    """Validate API response structure."""
    if not response:
        raise LLMFormatError(
            message=f"{self.provider_name} returned empty response",
            provider=self.provider_name,
            details={"response": str(response)}
        )

    if "choices" not in response:
        raise LLMFormatError(
            message=f"{self.provider_name} response missing 'choices' field",
            provider=self.provider_name,
            details={"response": str(response)}
        )

    if not response["choices"]:
        raise LLMFormatError(
            message=f"{self.provider_name} returned empty choices array",
            provider=self.provider_name,
            details={"response": str(response)}
        )
```

## Related Documentation

- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Full error handling guidelines
- [OPENAI_ERROR_HANDLING_IMPLEMENTATION.md](OPENAI_ERROR_HANDLING_IMPLEMENTATION.md) - OpenAI-specific implementation
- [ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION.md](ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION.md) - Anthropic-specific implementation
- [GOOGLE_ERROR_HANDLING_IMPLEMENTATION.md](GOOGLE_ERROR_HANDLING_IMPLEMENTATION.md) - Google-specific implementation
- [BEDROCK_ERROR_HANDLING_IMPLEMENTATION.md](BEDROCK_ERROR_HANDLING_IMPLEMENTATION.md) - Bedrock-specific implementation
- [ERROR_HANDLING_SUMMARY.md](ERROR_HANDLING_SUMMARY.md) - Implementation summary across providers
