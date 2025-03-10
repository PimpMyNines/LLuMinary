"""
Custom exceptions for the LLuMinary package.
"""

from typing import Any, Dict, Optional


class LLMError(Exception):
    """Base exception class for LLM-related errors."""

    def __init__(
        self, message: str, provider: str | None = None, details: dict | None = None
    ):
        self.message = message
        self.provider = provider
        self.details = details or {}
        super().__init__(message)


class LLMProviderError(LLMError):
    """
    Exception raised when there is an issue with an LLM provider.
    Used for configuration, authentication, or availability errors.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a LLMProviderError exception.

        Args:
            message: Error description
            provider: Name of the provider that raised the error
            details: Additional error details
        """
        self.provider = provider
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary format.

        Returns:
            Dictionary containing error details
        """
        return {
            "message": str(self),
            "provider": self.provider,
            "details": self.details,
        }


class LLMAuthenticationError(LLMError):
    """
    Exception raised when authentication with a provider fails.
    This could be due to invalid, expired, or missing credentials.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)


class LLMRateLimitError(LLMError):
    """
    Exception raised when a provider's rate limit is exceeded.
    Includes information about retry delays when available.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if details is None:
            details = {}

        if retry_after is not None:
            details["retry_after"] = retry_after

        super().__init__(message, provider, details)

    @property
    def retry_after(self) -> Optional[int]:
        """Get the recommended retry delay in seconds."""
        retry_val = self.details.get("retry_after")
        return int(retry_val) if retry_val is not None else None


class LLMConfigurationError(LLMError):
    """
    Exception raised when there's an issue with provider configuration.
    This could include invalid model names, incompatible parameters, etc.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)


class LLMServiceUnavailableError(LLMError):
    """
    Exception raised when a provider service is temporarily unavailable.
    This typically indicates a temporary outage or maintenance.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)


class LLMTimeoutError(LLMError):
    """
    Exception raised when a request to a provider times out.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        timeout: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        if details is None:
            details = {}

        if timeout is not None:
            details["timeout"] = timeout

        super().__init__(message, provider, details)


class LLMConnectionError(LLMError):
    """
    Exception raised when there's a connection error with a provider.
    This could be due to network issues or provider API endpoint problems.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)


class LLMValidationError(LLMError):
    """
    Exception raised when there's an issue with input validation.
    This could include invalid parameters, incompatible types, etc.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)


class LLMMistake(LLMError):
    """
    Custom exception class for LLM-related errors.
    Used to indicate when an LLM response needs correction or retry.
    """

    def __init__(
        self,
        message: str,
        error_type: str = "general",
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an LLMMistake exception.

        Args:
            message: Error description
            error_type: Type of error (e.g., "format", "content", "tool")
            provider: Name of the provider that raised the error
            details: Additional error details
        """
        self.error_type = error_type
        self.provider = provider
        self.details = details or {}
        super().__init__(message)

    @property
    def is_recoverable(self) -> bool:
        """
        Check if this error type is potentially recoverable.

        Returns:
            bool: True if the error might be resolved by retry
        """
        # Most LLMMistake errors are recoverable by design
        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary format.

        Returns:
            Dictionary containing error details
        """
        return {
            "message": str(self),
            "error_type": self.error_type,
            "provider": self.provider,
            "details": self.details,
            "recoverable": self.is_recoverable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMistake":
        """
        Create an LLMMistake instance from a dictionary.

        Args:
            data: Dictionary containing error details

        Returns:
            LLMMistake: New instance with the specified details
        """
        return cls(
            message=data["message"],
            error_type=data.get("error_type", "general"),
            provider=data.get("provider"),
            details=data.get("details", {}),
        )


class LLMFormatError(LLMError):
    """
    Exception raised when an LLM's response has formatting issues.
    This could include invalid JSON, XML, or other expected formats.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)


class LLMContentError(LLMError):
    """
    Exception raised when an LLM's response content is problematic.
    This could include empty responses, incorrect reasoning, or hallucinations.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)


class LLMToolError(LLMError):
    """
    Exception raised when there's an issue with tool/function usage in an LLM response.
    This could include invalid parameters, incorrect tool selection, etc.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)


class LLMThinkingError(LLMError):
    """
    Exception raised when there's an issue with the LLM's thinking process.
    This applies to models with explicit thinking/reasoning steps.
    """

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, provider, details)
