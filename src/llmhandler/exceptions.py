"""
Custom exceptions for the LLMHandler package.
"""
from typing import Any, Dict, Optional


class LLMHandlerError(Exception):
    """Base exception class for all LLMHandler errors."""

    pass


class ProviderError(LLMHandlerError):
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
        Initialize a ProviderError exception.

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


class LLMMistake(LLMHandlerError):
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
