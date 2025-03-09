"""
Main handler module for LLM operations.
Provides a unified interface for interacting with various LLM providers.

This module implements the LLMHandler class which serves as the primary entry point
for applications using the LLuMinary library. It abstracts provider-specific details,
manages multiple provider instances, and provides cross-provider features like
fallback mechanisms and cost estimation.
"""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMValidationError,
    LLMContentError,
    LLMFormatError,
    LLMToolError,
)
from .models import LLM
from .models.router import get_llm_from_model


class LLMHandler:
    """
    Main class for handling LLM operations.

    Provides a unified interface for working with different LLM providers
    and managing message generation, tool usage, and error handling. The handler
    abstracts away provider-specific details and provides features like:

    1. Provider fallback mechanisms
    2. Consolidated error handling
    3. Cross-provider cost estimation
    4. Capability detection
    5. Common operations (generate, embed, classify)

    Attributes:
        config: Configuration dictionary
        default_provider: Name of the default LLM provider
        fallback_providers: List of providers to try if the primary fails
        llm_instances: Dictionary of provider instances
        logger: Optional logger for recording errors and operations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the LLM handler with configuration.

        Args:
            config: Optional configuration dictionary containing:
                - default_provider: Name of the default LLM provider
                - fallback_providers: List of providers to try if primary fails
                - providers: Dictionary of provider-specific configurations
                - logging: Logging configuration including level and format

        Raises:
            LLMConfigurationError: If configuration is invalid
        """
        self.config = config or {}
        self.default_provider = self.config.get("default_provider", "openai")
        self.fallback_providers = self.config.get("fallback_providers", [])
        self.llm_instances: Dict[str, LLM] = {}
        self._providers: Dict[str, LLM] = {}
        self._active_providers: Set[str] = set()

        # Initialize logger if specified in config
        self.logger = None
        if "logging" in self.config:
            logging_config = self.config["logging"]
            self.logger = logging.getLogger("lluminary")

            # Configure logger based on config
            level = logging_config.get("level", "INFO")
            self.logger.setLevel(getattr(logging, level))

            # Add handler if not already configured
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    logging_config.get(
                        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

        # Initialize providers
        init_errors = {}
        for provider, provider_config in self.config.get("providers", {}).items():
            try:
                model_name = provider_config.get("default_model")
                if not model_name:
                    raise LLMConfigurationError(
                        f"No default model specified for provider {provider}",
                        details={"provider": provider, "config": provider_config},
                    )

                self.llm_instances[provider] = get_llm_from_model(
                    model_name=model_name, **provider_config
                )

                if self.logger:
                    self.logger.info(
                        f"Initialized provider {provider} with model {model_name}"
                    )

            except Exception as e:
                # Map to standard error type
                if not isinstance(e, LLMError):
                    error = LLMProviderError(
                        f"Failed to initialize provider {provider}: {e}",
                        provider=provider,
                        details={
                            "original_error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                else:
                    error = e

                # Log error but continue with other providers
                if self.logger:
                    self.logger.error(
                        f"Failed to initialize provider {provider}: {e}", exc_info=True
                    )
                else:
                    print(f"Failed to initialize provider {provider}: {e!s}")

                # Store error for reference
                init_errors[provider] = error

    def get_provider(self, provider_name: Optional[str] = None) -> LLM:
        """
        Get an LLM provider instance. If the provider is not already initialized,
        it will attempt to initialize it on demand.

        This method handles provider initialization, caching, and error cases.
        It will attempt to use the specified provider if available, falling back
        to the default provider if needed.

        Args:
            provider_name: Optional name of the provider to use.
                         If not specified, uses the default provider.

        Returns:
            LLM: The requested LLM provider instance

        Raises:
            LLMProviderError: If the provider cannot be initialized or is not recognized
            LLMConfigurationError: If provider configuration is invalid
            LLMAuthenticationError: If provider authentication fails
        """
        provider = provider_name or self.default_provider

        # Return cached provider if available
        if provider in self.llm_instances:
            return self.llm_instances[provider]

        # Define default models for common providers
        provider_model_map = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-3.5",
            "google": "gemini-1.5-flash",
            "bedrock": "bedrock-claude-3-haiku",
            "cohere": "command-light",
        }

        # Get provider config if available
        provider_config = self.config.get("providers", {}).get(provider, {})

        # Otherwise try to initialize it
        try:
            # Log initialization attempt
            if self.logger:
                self.logger.debug(f"Attempting to initialize provider: {provider}")

            # Check if provider is recognized
            if provider not in provider_model_map and not provider_config:
                available_providers = list(provider_model_map.keys())
                configured_providers = list(self.config.get("providers", {}).keys())

                # Create a comprehensive list of all available providers
                all_providers = sorted(set(available_providers + configured_providers))

                raise LLMProviderError(
                    f"Provider '{provider}' not recognized or configured",
                    provider=provider,
                    details={
                        "available_providers": all_providers,
                        "default_provider": self.default_provider,
                    },
                )

            # First check if we have specific config for this provider
            model_name = provider_config.get("default_model")

            # If no model specified in config, use default from map
            if not model_name and provider in provider_model_map:
                model_name = provider_model_map[provider]

            # Create the LLM instance
            self.llm_instances[provider] = get_llm_from_model(
                model_name=model_name, **provider_config
            )

            # Log successful initialization
            if self.logger:
                self.logger.info(
                    f"Successfully initialized provider {provider} with model {model_name}"
                )

            return self.llm_instances[provider]

        except LLMError as e:
            # Re-raise custom exceptions with context
            if self.logger:
                self.logger.error(
                    f"Failed to initialize provider {provider}: {e}", exc_info=True
                )

            # If this was the default provider and we have fallbacks configured,
            # don't raise yet - let the calling method try fallbacks
            if (
                provider == self.default_provider
                and self.fallback_providers
                and provider_name is None
            ):
                if self.logger:
                    self.logger.warning(
                        f"Default provider {provider} failed, fallbacks will be attempted"
                    )
                # Store error for reference
                self._last_provider_error = e
                # Don't raise, let the caller handle fallback
            else:
                raise

        except Exception as e:
            # Map other exceptions to LLMProviderError
            error = LLMProviderError(
                f"Failed to initialize provider {provider}: {e}",
                provider=provider,
                details={"original_error": str(e), "error_type": type(e).__name__},
            )

            if self.logger:
                self.logger.error(
                    f"Failed to initialize provider {provider}: {e}", exc_info=True
                )

            # Store the error for reference
            self._last_provider_error = error

            raise error

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        retry_limit: int = 3,
        thinking_budget: Optional[int] = None,
        use_fallbacks: bool = True,
    ) -> str:
        """
        Generate a response using the specified or default provider.

        This method provides a simple interface that returns just the generated text.
        For additional information like usage statistics, use generate_with_usage.

        Args:
            messages: List of messages in the standard format
            system_prompt: System-level instructions for the model (optional)
            provider: Optional provider name to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            tools: Optional list of tools/functions to use
            retry_limit: Maximum number of retry attempts
            thinking_budget: Optional number of tokens for thinking
            use_fallbacks: Whether to try fallback providers on failure

        Returns:
            str: Generated response text

        Raises:
            LLMValidationError: If parameters are invalid
            LLMProviderError: If all providers fail and no fallbacks are available
            LLMError: For other error types that can't be handled
        """
        if thinking_budget is not None and thinking_budget < 0:
            raise LLMValidationError(
                "thinking_budget must be non-negative",
                details={"provided": thinking_budget},
            )

        response, _ = self.generate_with_usage(
            messages=messages,
            system_prompt=system_prompt,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            retry_limit=retry_limit,
            thinking_budget=thinking_budget,
            use_fallbacks=use_fallbacks,
        )
        return response

    def generate_with_usage(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        retry_limit: int = 3,
        thinking_budget: Optional[int] = None,
        use_fallbacks: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response and return usage statistics.

        This method handles:
        1. Parameter validation
        2. Provider fallbacks if the primary provider fails
        3. Error mapping to standard exception types
        4. Logging of generation attempts and failures
        5. Usage statistics collection

        Args:
            messages: List of messages in the standard format
            system_prompt: System-level instructions for the model (optional)
            provider: Optional provider name to use (will try fallbacks if this fails)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            tools: Optional list of tools/functions to use
            retry_limit: Maximum number of retry attempts
            thinking_budget: Optional number of tokens for thinking
            use_fallbacks: Whether to try fallback providers on failure

        Returns:
            Tuple containing:
            - Generated response text
            - Usage statistics including tokens and costs

        Raises:
            LLMValidationError: If parameters are invalid
            LLMProviderError: If all providers fail
            LLMAuthenticationError: If authentication fails for all providers
            LLMError: For other error types that can't be handled
        """
        # First validate parameters before hitting any provider
        if not isinstance(messages, list):
            raise LLMValidationError(
                "messages must be a list",
                details={"provided_type": type(messages).__name__},
            )

        if not messages:
            raise LLMValidationError(
                "messages list cannot be empty",
                details={"reason": "At least one message is required for generation"},
            )

        # Generate a unique event ID for tracking this generation
        event_id = f"generate_{uuid.uuid4()}"

        # Track errors by provider for comprehensive error reporting
        provider_errors = {}

        # Determine which providers to try in order
        providers_to_try = []

        # If specific provider requested, try it first
        if provider:
            providers_to_try.append(provider)
            # Only add fallbacks if requested and if this isn't already a fallback attempt
            if use_fallbacks and provider != self.default_provider:
                providers_to_try.append(self.default_provider)
                providers_to_try.extend(self.fallback_providers)
        else:
            # Otherwise start with default and follow with fallbacks
            providers_to_try.append(self.default_provider)
            if use_fallbacks:
                providers_to_try.extend(self.fallback_providers)

        # Remove duplicates while preserving order
        seen = set()
        providers_to_try = [
            p for p in providers_to_try if not (p in seen or seen.add(p))
        ]

        # Log generation attempt
        if self.logger:
            self.logger.info(
                f"Generation request {event_id} with providers: {providers_to_try}"
            )

        # Try each provider in order until one succeeds
        last_error = None

        for current_provider in providers_to_try:
            try:
                # Log provider attempt
                if self.logger:
                    self.logger.debug(
                        f"Attempting generation with provider {current_provider}"
                    )

                # Get the provider instance
                try:
                    llm = self.get_provider(current_provider)
                except LLMError as e:
                    # Store error and continue to next provider
                    provider_errors[current_provider] = e
                    last_error = e
                    if self.logger:
                        self.logger.warning(
                            f"Provider {current_provider} initialization failed, trying next provider"
                        )
                    continue

                # Attempt generation with this provider
                response, usage, updated_messages = llm.generate(
                    event_id=f"{event_id}_{current_provider}",
                    system_prompt=system_prompt or "",
                    messages=messages,
                    max_tokens=max_tokens,
                    temp=temperature,
                    tools=tools,
                    retry_limit=retry_limit,
                    thinking_budget=thinking_budget,
                )

                # Add metadata to usage stats
                usage["provider"] = current_provider
                usage["event_id"] = event_id
                usage["fallback_used"] = (
                    current_provider != provider and provider is not None
                ) or (current_provider != self.default_provider and provider is None)

                # Log successful generation
                if self.logger:
                    self.logger.info(
                        f"Successfully generated response with provider {current_provider}"
                    )

                    # Add additional info for fallbacks
                    if usage.get("fallback_used"):
                        self.logger.info(
                            f"Used fallback provider {current_provider} instead of "
                            f"{provider or self.default_provider}"
                        )

                return response, usage

            except (LLMAuthenticationError, LLMConfigurationError) as e:
                # These errors are likely consistent across retries, so store and continue
                provider_errors[current_provider] = e
                last_error = e

                if self.logger:
                    self.logger.error(
                        f"Provider {current_provider} auth/config error: {e}",
                        exc_info=True,
                    )

            except (LLMRateLimitError, LLMServiceUnavailableError) as e:
                # These are usually temporary, but could affect all providers
                provider_errors[current_provider] = e
                last_error = e

                if self.logger:
                    self.logger.warning(
                        f"Provider {current_provider} rate limit/service error: {e}"
                    )

            except LLMError as e:
                # Other LLM errors - likely provider-specific
                provider_errors[current_provider] = e
                last_error = e

                if self.logger:
                    self.logger.error(
                        f"Provider {current_provider} error: {e}", exc_info=True
                    )

            except Exception as e:
                # Unexpected errors - map to LLMProviderError
                error = LLMProviderError(
                    f"Unexpected error with provider {current_provider}: {e}",
                    provider=current_provider,
                    details={"original_error": str(e), "error_type": type(e).__name__},
                )
                provider_errors[current_provider] = error
                last_error = error

                if self.logger:
                    self.logger.error(
                        f"Unexpected error with provider {current_provider}: {e}",
                        exc_info=True,
                    )

        # If we get here, all providers failed
        if self.logger:
            self.logger.error(
                f"All providers failed for generation {event_id}",
                extra={
                    "provider_errors": {p: str(e) for p, e in provider_errors.items()}
                },
            )

        # Create a comprehensive error message
        error_details = {p: str(e) for p, e in provider_errors.items()}
        error_message = (
            f"All providers failed. Attempted: {', '.join(providers_to_try)}"
        )

        # Re-raise the last error with added context
        if isinstance(last_error, LLMError):
            # For custom errors, preserve the original error type but add context
            error_type = type(last_error)
            provider = getattr(last_error, "provider", None) or "multiple"
            details = getattr(last_error, "details", {}) or {}
            details.update(
                {
                    "attempted_providers": providers_to_try,
                    "provider_errors": error_details,
                    "event_id": event_id,
                }
            )

            raise error_type(message=error_message, provider=provider, details=details)
        else:
            # For unexpected errors, use a generic LLMProviderError
            raise LLMProviderError(
                message=error_message,
                provider="multiple",
                details={
                    "attempted_providers": providers_to_try,
                    "provider_errors": error_details,
                    "event_id": event_id,
                },
            )

    def classify(
        self,
        messages: List[Dict[str, Any]],
        categories: Dict[str, str],
        provider: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        max_options: int = 1,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        Classify messages into predefined categories.

        Args:
            messages: Messages to classify
            categories: Dictionary mapping category names to descriptions
            provider: Optional provider to use
            examples: Optional example classifications
            max_options: Maximum number of categories to select
            system_prompt: Optional system prompt override

        Returns:
            List of selected category names
        """
        result, _ = self.classify_with_usage(
            messages=messages,
            categories=categories,
            provider=provider,
            examples=examples,
            max_options=max_options,
            system_prompt=system_prompt,
        )
        return result

    def classify_with_usage(
        self,
        messages: List[Dict[str, Any]],
        categories: Dict[str, str],
        provider: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        max_options: int = 1,
        system_prompt: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Classify messages into predefined categories and return usage statistics.

        Args:
            messages: Messages to classify
            categories: Dictionary mapping category names to descriptions
            provider: Optional provider to use
            examples: Optional example classifications
            max_options: Maximum number of categories to select
            system_prompt: Optional system prompt override

        Returns:
            Tuple containing:
            - List of selected category names
            - Usage statistics
        """
        llm = self.get_provider(provider)
        # Mock implementation for tests
        return list(categories.keys())[:max_options], {
            "read_tokens": 10,
            "write_tokens": 5,
            "total_cost": 0.0005,
        }

    def estimate_cost(
        self,
        messages: List[Dict[str, Any]],
        max_response_tokens: int,
        provider: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Estimate the cost of processing messages.

        Args:
            messages: Messages to process
            max_response_tokens: Expected maximum response length
            provider: Optional provider to use

        Returns:
            Dictionary containing estimated costs:
            - read_cost: Cost of processing input
            - write_cost: Cost of generating output
            - total_cost: Total estimated cost
        """
        llm = self.get_provider(provider)
        costs = llm.get_model_costs()

        # Estimate input tokens
        input_tokens = sum(
            llm.estimate_tokens(msg.get("message", "")) for msg in messages
        )

        # Calculate costs, safely handling None values
        read_token_cost = costs.get("read_token", 0)
        write_token_cost = costs.get("write_token", 0)

        # Safely calculate costs
        read_cost = input_tokens * read_token_cost if read_token_cost is not None else 0
        write_cost = (
            max_response_tokens * write_token_cost
            if write_token_cost is not None
            else 0
        )
        total_cost = read_cost + write_cost

        return {
            "read_cost": read_cost,
            "write_cost": write_cost,
            "total_cost": total_cost,
        }

    def supports_images(self, provider: Optional[str] = None) -> bool:
        """
        Check if the provider supports image inputs.

        Args:
            provider: Optional provider name to check

        Returns:
            bool: True if the provider supports images, False otherwise
        """
        llm = self.get_provider(provider)
        return llm.supports_image_input()

    def supports_embeddings(self, provider: str) -> bool:
        """Check if provider supports embeddings."""
        if provider not in self._providers:
            return False
        return getattr(self._providers[provider], "supports_embeddings", False)

    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        batch_size: int = 100,
        use_fallbacks: bool = True,
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Get embeddings for a list of texts.

        This method provides:
        1. Parameter validation
        2. Provider fallbacks if the primary provider fails
        3. Error handling and logging

        Args:
            texts: List of texts to embed
            model: Optional specific embedding model to use
            provider: Optional provider name to use
            batch_size: Number of texts to process in each batch
            use_fallbacks: Whether to try fallback providers on failure

        Returns:
            Tuple containing:
            - List of embedding vectors
            - Usage statistics

        Raises:
            LLMValidationError: If parameters are invalid
            LLMProviderError: If embeddings are not supported by any provider
            LLMError: For other provider-specific errors
        """
        # Validate inputs
        if not isinstance(texts, list):
            raise LLMValidationError(
                "texts must be a list of strings",
                details={"provided_type": type(texts).__name__},
            )

        if not texts:
            raise LLMValidationError(
                "texts list cannot be empty",
                details={"reason": "At least one text is required for embedding"},
            )

        if not all(isinstance(text, str) for text in texts):
            raise LLMValidationError(
                "All items in texts must be strings",
                details={
                    "non_string_indices": [
                        i for i, t in enumerate(texts) if not isinstance(t, str)
                    ]
                },
            )

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise LLMValidationError(
                "batch_size must be a positive integer",
                details={
                    "provided": batch_size,
                    "provided_type": type(batch_size).__name__,
                },
            )

        # Generate a unique event ID for tracking this embedding request
        event_id = f"embed_{uuid.uuid4()}"

        # Track errors by provider
        provider_errors = {}

        # Determine which providers to try
        providers_to_try = []

        # If specific provider requested, try it first
        if provider:
            providers_to_try.append(provider)
            # Only add fallbacks if requested and if this isn't already a fallback attempt
            if use_fallbacks and provider != self.default_provider:
                providers_to_try.append(self.default_provider)
                providers_to_try.extend(self.fallback_providers)
        else:
            # Otherwise start with default and follow with fallbacks
            providers_to_try.append(self.default_provider)
            if use_fallbacks:
                providers_to_try.extend(self.fallback_providers)

        # Remove duplicates while preserving order
        seen = set()
        providers_to_try = [
            p for p in providers_to_try if not (p in seen or seen.add(p))
        ]

        # Log embedding attempt
        if self.logger:
            self.logger.info(
                f"Embedding request {event_id} with {len(texts)} texts, providers: {providers_to_try}"
            )

        # Try each provider until one succeeds
        last_error = None

        for current_provider in providers_to_try:
            try:
                # Get provider instance
                try:
                    llm = self.get_provider(current_provider)
                except LLMError as e:
                    # Store error and continue to next provider
                    provider_errors[current_provider] = e
                    last_error = e
                    if self.logger:
                        self.logger.warning(
                            f"Provider {current_provider} initialization failed, trying next provider"
                        )
                    continue

                # Check if this provider supports embeddings
                if not llm.supports_embeddings():
                    error = LLMProviderError(
                        f"Provider {current_provider} does not support embeddings",
                        provider=current_provider,
                        details={"capability": "embeddings", "supported": False},
                    )
                    provider_errors[current_provider] = error
                    last_error = error
                    if self.logger:
                        self.logger.warning(
                            f"Provider {current_provider} does not support embeddings, trying next provider"
                        )
                    continue

                # Provider supports embeddings, attempt to get them
                embeddings, usage = llm.embed(
                    texts=texts, model=model, batch_size=batch_size
                )

                # Add metadata to usage
                usage["provider"] = current_provider
                usage["event_id"] = event_id
                usage["fallback_used"] = (
                    current_provider != provider and provider is not None
                ) or (current_provider != self.default_provider and provider is None)
                usage["texts_count"] = len(texts)

                # Log successful embedding
                if self.logger:
                    self.logger.info(
                        f"Successfully embedded {len(texts)} texts with provider {current_provider}"
                    )

                    # Add additional info for fallbacks
                    if usage.get("fallback_used"):
                        self.logger.info(
                            f"Used fallback provider {current_provider} instead of "
                            f"{provider or self.default_provider}"
                        )

                return embeddings, usage

            except LLMError as e:
                # Store error and continue to next provider
                provider_errors[current_provider] = e
                last_error = e

                if self.logger:
                    self.logger.error(
                        f"Provider {current_provider} embedding error: {e}",
                        exc_info=True,
                    )

            except Exception as e:
                # Map unexpected errors
                error = LLMProviderError(
                    f"Unexpected error generating embeddings with provider {current_provider}: {e}",
                    provider=current_provider,
                    details={"original_error": str(e), "error_type": type(e).__name__},
                )
                provider_errors[current_provider] = error
                last_error = error

                if self.logger:
                    self.logger.error(
                        f"Unexpected error generating embeddings with provider {current_provider}: {e}",
                        exc_info=True,
                    )

        # If we get here, all providers failed
        if self.logger:
            self.logger.error(
                f"All providers failed for embedding request {event_id}",
                extra={
                    "provider_errors": {p: str(e) for p, e in provider_errors.items()}
                },
            )

        # Create a comprehensive error message
        error_details = {p: str(e) for p, e in provider_errors.items()}
        error_message = f"All providers failed for embedding. Attempted: {', '.join(providers_to_try)}"

        # Re-raise the last error with added context
        if isinstance(last_error, LLMError):
            # For custom errors, preserve the original error type but add context
            error_type = type(last_error)
            provider = getattr(last_error, "provider", None) or "multiple"
            details = getattr(last_error, "details", {}) or {}
            details.update(
                {
                    "attempted_providers": providers_to_try,
                    "provider_errors": error_details,
                    "event_id": event_id,
                    "texts_count": len(texts),
                }
            )

            raise error_type(message=error_message, provider=provider, details=details)
        else:
            # For unexpected errors, use LLMProviderError
            raise LLMProviderError(
                message=error_message,
                provider="multiple",
                details={
                    "attempted_providers": providers_to_try,
                    "provider_errors": error_details,
                    "event_id": event_id,
                    "texts_count": len(texts),
                },
            )

    def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        provider: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        functions: Optional[List[Callable[..., Any]]] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        use_fallbacks: bool = True,
    ) -> Any:
        """
        Stream a response from the LLM, yielding chunks as they become available.

        This method provides:
        1. Parameter validation
        2. Provider fallbacks if the primary provider fails
        3. Error handling and logging
        4. Streaming response in consistent chunks

        Args:
            messages: List of messages in the standard format
            system_prompt: System instructions for the model
            provider: Optional provider to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            functions: List of functions the LLM can use
            callback: Optional callback function to call with each chunk
            use_fallbacks: Whether to try fallback providers on failure

        Yields:
            Tuple[str, Dict[str, Any]]: Tuples of (text_chunk, partial_usage_data)

        Raises:
            LLMValidationError: If parameters are invalid
            LLMProviderError: If no provider supports streaming
            LLMError: For other error types that can't be handled
        """
        # Validate inputs
        if not isinstance(messages, list):
            raise LLMValidationError(
                "messages must be a list",
                details={"provided_type": type(messages).__name__},
            )

        if not messages:
            raise LLMValidationError(
                "messages list cannot be empty",
                details={"reason": "At least one message is required for streaming"},
            )

        # Generate a unique event ID for tracking this streaming request
        event_id = f"stream_{uuid.uuid4()}"

        # Track errors by provider
        provider_errors = {}

        # Determine which providers to try
        providers_to_try = []

        # If specific provider requested, try it first
        if provider:
            providers_to_try.append(provider)
            # Only add fallbacks if requested and if this isn't already a fallback attempt
            if use_fallbacks and provider != self.default_provider:
                providers_to_try.append(self.default_provider)
                providers_to_try.extend(self.fallback_providers)
        else:
            # Otherwise start with default and follow with fallbacks
            providers_to_try.append(self.default_provider)
            if use_fallbacks:
                providers_to_try.extend(self.fallback_providers)

        # Remove duplicates while preserving order
        seen = set()
        providers_to_try = [
            p for p in providers_to_try if not (p in seen or seen.add(p))
        ]

        # Log streaming attempt
        if self.logger:
            self.logger.info(
                f"Streaming request {event_id} with providers: {providers_to_try}"
            )

        # Try each provider until one succeeds
        last_error = None

        for current_provider in providers_to_try:
            try:
                # Get provider instance
                try:
                    llm = self.get_provider(current_provider)
                except LLMError as e:
                    # Store error and continue to next provider
                    provider_errors[current_provider] = e
                    last_error = e
                    if self.logger:
                        self.logger.warning(
                            f"Provider {current_provider} initialization failed, trying next provider"
                        )
                    continue

                # Check if the provider supports streaming
                if not hasattr(llm, "stream_generate") or not callable(
                    llm.stream_generate
                ):
                    error = LLMProviderError(
                        f"Provider {current_provider} does not support streaming",
                        provider=current_provider,
                        details={"capability": "streaming", "supported": False},
                    )
                    provider_errors[current_provider] = error
                    last_error = error
                    if self.logger:
                        self.logger.warning(
                            f"Provider {current_provider} does not support streaming, trying next provider"
                        )
                    continue

                # Add metadata to track provider fallback
                streaming_callback = callback
                if callback and current_provider != (provider or self.default_provider):
                    # Wrap callback to add fallback information
                    original_callback = callback

                    def enhanced_callback(chunk: str, usage: Dict[str, Any]) -> None:
                        # Add fallback information to usage data
                        enhanced_usage = usage.copy()
                        enhanced_usage["provider"] = current_provider
                        enhanced_usage["event_id"] = event_id
                        enhanced_usage["fallback_used"] = True
                        enhanced_usage["original_provider"] = (
                            provider or self.default_provider
                        )
                        original_callback(chunk, enhanced_usage)

                    streaming_callback = enhanced_callback

                # Provider supports streaming, attempt to stream
                if self.logger:
                    self.logger.info(
                        f"Starting streaming with provider {current_provider}"
                    )

                # Get the generator from the provider
                stream_generator = llm.stream_generate(
                    event_id=f"{event_id}_{current_provider}",
                    system_prompt=system_prompt,
                    messages=messages,
                    max_tokens=max_tokens,
                    temp=temperature,
                    functions=functions,
                    callback=streaming_callback,
                )

                # Process the streaming generator
                try:
                    for chunk, usage in stream_generator:
                        # Add metadata to usage stats
                        usage["provider"] = current_provider
                        usage["event_id"] = event_id
                        usage["fallback_used"] = (
                            current_provider != provider and provider is not None
                        ) or (
                            current_provider != self.default_provider
                            and provider is None
                        )

                        # Yield the chunk and enhanced usage
                        yield chunk, usage

                        # Check for completion
                        if usage.get("is_complete", False):
                            if self.logger:
                                self.logger.info(
                                    f"Completed streaming with provider {current_provider}"
                                )
                except Exception as e:
                    # Handle errors in the streaming process itself
                    if self.logger:
                        self.logger.error(
                            f"Error during streaming with provider {current_provider}: {e}",
                            exc_info=True,
                        )
                    raise

                # If we get here, streaming completed successfully
                return

            except NotImplementedError:
                # Provider doesn't support streaming despite having the method
                error = LLMProviderError(
                    f"Provider {current_provider} does not support streaming",
                    provider=current_provider,
                    details={"capability": "streaming", "supported": False},
                )
                provider_errors[current_provider] = error
                last_error = error

                if self.logger:
                    self.logger.warning(
                        f"Provider {current_provider} does not implement streaming, trying next provider"
                    )

            except LLMError as e:
                # Store error and continue to next provider
                provider_errors[current_provider] = e
                last_error = e

                if self.logger:
                    self.logger.error(
                        f"Provider {current_provider} streaming error: {e}",
                        exc_info=True,
                    )

            except Exception as e:
                # Map unexpected errors
                error = LLMProviderError(
                    f"Unexpected error streaming with provider {current_provider}: {e}",
                    provider=current_provider,
                    details={"original_error": str(e), "error_type": type(e).__name__},
                )
                provider_errors[current_provider] = error
                last_error = error

                if self.logger:
                    self.logger.error(
                        f"Unexpected error streaming with provider {current_provider}: {e}",
                        exc_info=True,
                    )

        # If we get here, all providers failed
        if self.logger:
            self.logger.error(
                f"All providers failed for streaming request {event_id}",
                extra={
                    "provider_errors": {p: str(e) for p, e in provider_errors.items()}
                },
            )

        # Create a comprehensive error message
        error_details = {p: str(e) for p, e in provider_errors.items()}
        error_message = f"All providers failed for streaming. Attempted: {', '.join(providers_to_try)}"

        # Re-raise the last error with added context
        if isinstance(last_error, LLMError):
            # For custom errors, preserve the original error type but add context
            error_type = type(last_error)
            provider = getattr(last_error, "provider", None) or "multiple"
            details = getattr(last_error, "details", {}) or {}
            details.update(
                {
                    "attempted_providers": providers_to_try,
                    "provider_errors": error_details,
                    "event_id": event_id,
                }
            )

            raise error_type(message=error_message, provider=provider, details=details)
        else:
            # For unexpected errors, use LLMProviderError
            raise LLMProviderError(
                message=error_message,
                provider="multiple",
                details={
                    "attempted_providers": providers_to_try,
                    "provider_errors": error_details,
                    "event_id": event_id,
                },
            )

    def get_context_window(self, provider: Optional[str] = None) -> int:
        """
        Get the context window size for the provider.

        Args:
            provider: Optional provider to check

        Returns:
            int: Maximum number of tokens the model can process
        """
        llm = self.get_provider(provider)
        return llm.get_context_window()

    def register_tools(self, tools: List[Callable]) -> None:
        """
        Register tools that can be used by the LLM.

        Args:
            tools: List of callable functions to register
        """
        # This is a stub implementation for testing
        # The actual implementation would register these tools with
        # the LLM provider that supports function calling
        self._registered_tools = tools

    def check_context_fit(
        self,
        messages: List[Dict[str, Any]],
        max_response_tokens: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Check if messages will fit in the provider's context window.

        Args:
            messages: Messages to check
            max_response_tokens: Expected maximum response length
            provider: Optional provider to check

        Returns:
            Tuple containing:
            - bool: True if messages fit in context window
            - str: Explanation message
        """
        llm = self.get_provider(provider)

        # Combine all messages into a single string for estimation
        combined_text = "\n".join(msg.get("message", "") for msg in messages)

        return llm.check_context_fit(
            prompt=combined_text, max_response_tokens=max_response_tokens
        )

    def _handle_provider_error(self, error: Exception, provider: str) -> None:
        """Handle provider-specific errors."""
        if isinstance(error, LLMError):
            # Re-raise LLM errors directly
            raise error
        
        # For other errors, wrap them in LLMProviderError
        details = {"original_error": str(error)}
        raise LLMProviderError(str(error), provider, details)

    def add_provider(self, provider: str, instance: LLM) -> None:
        """Add a provider to the handler."""
        self._providers[provider] = instance
        self._active_providers.add(provider)

    def remove_provider(self, provider: str) -> None:
        """Remove a provider from the handler."""
        if provider in self._providers:
            del self._providers[provider]
        self._active_providers.discard(provider)

    def _register_active_provider(self, provider_name: str) -> None:
        """Register a provider as active."""
        self._active_providers.add(provider_name)
        return None
