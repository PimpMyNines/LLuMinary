# BEDROCK ERROR HANDLING IMPLEMENTATION

## Overview

This document outlines the comprehensive error handling implementation for the AWS Bedrock provider in the LLuMinary library, detailing AWS-specific error mapping strategies and implementation approaches.

## Table of Contents

- [Overview](#overview)
- [Implementation Approach](#implementation-approach)
- [Key Components](#key-components)
  - [AWS Error Mapping Method](#1-aws-error-mapping-method)
  - [Retry Mechanism](#2-retry-mechanism)
  - [Enhanced Authentication](#3-enhanced-authentication)
  - [Robust Image Processing](#4-robust-image-processing)
  - [Enhanced _raw_generate](#5-enhanced-_raw_generate)
- [AWS-Specific Considerations](#aws-specific-considerations)
  - [ClientError Structure](#clienterror-structure)
  - [Error Code Patterns](#error-code-patterns)
  - [Credential Chain](#credential-chain)
- [Future Improvements](#future-improvements)
- [Related Documentation](#related-documentation)

## Implementation Approach

The Bedrock provider error handling follows the same pattern established for other providers, with specialized handling for AWS-specific error types:

1. **AWS-Specific Error Mapping**: Translates AWS ClientError exceptions to LLuMinary exception types
2. **Enhanced Authentication**: Supports multiple credential sources with detailed error reporting
3. **Retry Mechanism**: Implements exponential backoff with jitter for transient errors
4. **Image Processing Robustness**: Handles AWS-specific size and format requirements
5. **Structured Exception Flow**: Consistent try/except patterns throughout the codebase

## Key Components

### 1. AWS Error Mapping Method

The `_map_aws_error` method examines exceptions from the AWS SDK (particularly `botocore.exceptions.ClientError`) and maps them to our standardized exception types.

```python
def _map_aws_error(self, error: Exception) -> Exception:
    """Map AWS specific errors to LLMHandler custom exceptions."""

    # Special handling for AWS ClientError
    if isinstance(error, ClientError):
        error_code = error.response["Error"]["Code"]

        # Authentication errors
        if error_code in ["AccessDeniedException", "InvalidSignatureException", ...]:
            return AuthenticationError(...)

        # Rate limit errors
        elif error_code in ["ThrottlingException", "ServiceQuotaExceededException", ...]:
            return RateLimitError(...)

        # Service availability errors
        elif error_code in ["ServiceUnavailableException", "InternalServerException", ...]:
            return ServiceUnavailableError(...)

        # ... other error types ...

    # Generic error message pattern matching
    error_message = str(error).lower()
    if "credential" in error_message:
        return AuthenticationError(...)
    # ... other pattern matching ...
```

### 2. Retry Mechanism

The `_call_with_retry` method implements AWS-specific retry logic with exponential backoff and jitter:

```python
def _call_with_retry(self, func, *args, max_retries=3, retry_delay=1, **kwargs):
    """Execute an AWS API call with automatic retry for transient errors."""

    while attempts <= max_retries:
        try:
            return func(*args, **kwargs)
        except (RateLimitError, ServiceUnavailableError) as e:
            # Handle with exponential backoff
        except ClientError as e:
            # Check if it's a retryable AWS error code
            if error_code in ["ThrottlingException", "ServiceUnavailableException", ...]:
                # Use retry-after from HTTP headers if available
            else:
                # Map to appropriate error and raise
```

### 3. Enhanced Authentication

The improved `auth()` method provides:

- Support for multiple credential sources (config, env vars, AWS config, IAM roles)
- Detailed error reporting with specific failure reasons
- Verification of Bedrock access through a test API call
- Proper error classification for different AWS authentication issues

### 4. Robust Image Processing

The image processing methods have been enhanced with:

- Validation of file existence and permissions
- Image format verification
- Size limit enforcement (5MB AWS limit)
- Dimension checking and automatic resizing
- Transparent error classification (LLMMistake vs ContentError)
- Detailed error contexts for debugging

### 5. Enhanced _raw_generate

The core generation method implements structured error handling:

1. Authentication verification with error mapping
2. Message formatting in a try/except block
3. API parameter preparation with configuration error handling
4. API calls wrapped in the retry mechanism
5. Response processing with appropriate error types
6. Usage statistics extraction with graceful fallbacks
7. Hierarchical exception handling with appropriate error mapping

## AWS-Specific Considerations

### ClientError Structure

AWS errors have a specific structure that allows precise error handling:

```python
try:
    client.some_operation()
except ClientError as e:
    error_code = e.response["Error"]["Code"]  # "ThrottlingException", etc.
    error_message = e.response["Error"]["Message"]  # Human-readable message
    # Headers may contain retry-after information
    headers = e.response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
```

### Error Code Patterns

AWS uses standard error codes across services that we map to our exception hierarchy:

- Authentication: `AccessDeniedException`, `InvalidSignatureException`
- Rate Limiting: `ThrottlingException`, `TooManyRequestsException`
- Service Issues: `ServiceUnavailableException`, `InternalServerException`
- Configuration: `ValidationException`, `InvalidParameterException`
- Content: `ContentException`, `ContentModerationException`

### Credential Chain

AWS uses a credential resolution chain that we've mirrored in our error handling:

1. Explicit credentials in config (access key, secret key)
2. Environment variables (AWS_ACCESS_KEY_ID, etc.)
3. Credential files (~/.aws/credentials)
4. IAM Instance Profiles (EC2, Lambda)

## Future Improvements

1. Extract and log AWS request IDs for better troubleshooting
2. Implement region-specific retry strategies based on AWS service status
3. Add support for AWS SDK's built-in retry configuration
4. Improve STS token handling for temporary credentials
5. Add more detailed diagnostics for IAM permission issues
6. Enhance tracking of throttling patterns for adaptive retry strategies

## Related Documentation

- [ERROR_HANDLING](./ERROR_HANDLING.md) - General error handling guidelines
- [MODELS](./MODELS.md) - Model implementation details
- [PROVIDER_TESTING](./PROVIDER_TESTING.md) - Provider testing guidelines
- [API_REFERENCE](../API_REFERENCE.md) - Complete API reference
- [ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION](./ANTHROPIC_ERROR_HANDLING_IMPLEMENTATION.md) - Anthropic error handling
- [GOOGLE_ERROR_HANDLING_IMPLEMENTATION](./GOOGLE_ERROR_HANDLING_IMPLEMENTATION.md) - Google error handling
- [OPENAI_ERROR_HANDLING_IMPLEMENTATION](./OPENAI_ERROR_HANDLING_IMPLEMENTATION.md) - OpenAI error handling
