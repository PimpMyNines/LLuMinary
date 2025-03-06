"""
Base class for LLM providers.
All provider-specific implementations should inherit from this class.

This base class provides common functionality, type definitions, and standard
error handling patterns that all provider implementations should follow.
It defines the interface and core behaviors expected from any provider.
"""

import inspect
import random
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from ..exceptions import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContentError,
    LLMError,
    LLMFormatError,
    LLMMistake,
    LLMProviderError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMThinkingError,
    LLMToolError,
    LLMValidationError,
)
from .classification.config import ClassificationConfig


class LLM(ABC):
    """
    Abstract base class for LLM provider implementations.

    All provider implementations must inherit from this class and implement
    its abstract methods. This class defines the standard interface that all
    LLM providers must conform to, ensuring consistent behavior across different
    implementations.

    Class Variables:
        CONTEXT_WINDOW: Maximum token limits per model
        COST_PER_MODEL: Cost per token per model
        SUPPORTED_MODELS: List of supported model identifiers
        THINKING_MODELS: Models supporting thinking/reasoning
        EMBEDDING_MODELS: Models supporting embedding generation
        RERANKING_MODELS: Models supporting document reranking
    """

    CONTEXT_WINDOW: ClassVar[Dict[str, int]] = {}
    COST_PER_MODEL: ClassVar[Dict[str, Dict[str, Union[float, None]]]] = {}
    SUPPORTED_MODELS: ClassVar[List[str]] = []
    THINKING_MODELS: ClassVar[List[str]] = []
    EMBEDDING_MODELS: ClassVar[List[str]] = []  # Models that support embeddings
    RERANKING_MODELS: ClassVar[List[str]] = []  # Models that support reranking

    def generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        retry_limit: int = 3,
        result_processing_function: Optional[Callable[[str], str]] = None,
        thinking_budget: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate a response with retry logic and error handling.

        This is the main entry point for generation that handles:
        1. Parameter validation
        2. Retry logic for recoverable errors
        3. Error handling and mapping to standard exceptions
        4. Usage statistics tracking
        5. Response processing and validation

        Args:
            event_id: Unique identifier for this generation
            system_prompt: Instructions for the model
            messages: List of messages to process
            max_tokens: Maximum tokens to generate
            temp: Temperature for generation
            tools: List of tools/functions to use
            retry_limit: Maximum number of retries
            result_processing_function: Function to process the result
            thinking_budget: Number of tokens for thinking (if supported)

        Returns:
            Tuple containing:
            - Generated response text
            - Usage statistics
            - Updated messages including the response

        Raises:
            LLMValidationError: If parameters are invalid
            LLMMistake: If generation fails after all retries
            LLMProviderError: For provider-specific errors
        """
        provider_name = self.__class__.__name__.replace("LLM", "")

        # Generate a unique event ID if not provided
        if not event_id:
            event_id = f"{provider_name.lower()}_{uuid.uuid4()}"

        # Validate all parameters
        try:
            # Validate basic parameters
            if not isinstance(messages, list):
                raise LLMValidationError(
                    "messages must be a list",
                    details={"provided_type": type(messages).__name__},
                )

            if not messages:
                raise LLMValidationError(
                    "messages list cannot be empty",
                    details={
                        "reason": "At least one message is required for generation"
                    },
                )

            # Validate system prompt
            if system_prompt is not None and not isinstance(system_prompt, str):
                raise LLMValidationError(
                    "system_prompt must be a string or None",
                    details={"provided_type": type(system_prompt).__name__},
                )

            # Validate max_tokens
            self._validate_max_tokens(max_tokens)

            # Validate temperature
            self._validate_temperature(temp)

            # Validate tools if provided
            if tools is not None:
                self._validate_tools(tools)

            # Validate thinking budget if provided
            if thinking_budget is not None:
                if not self.has_thinking_budget:
                    raise LLMValidationError(
                        f"Model {self.model_name} does not support thinking budget",
                        details={"model": self.model_name},
                    )
                if not isinstance(thinking_budget, int) or thinking_budget <= 0:
                    raise LLMValidationError(
                        "thinking_budget must be a positive integer",
                        details={
                            "provided": thinking_budget,
                            "provided_type": type(thinking_budget).__name__,
                        },
                    )

        except LLMValidationError:
            # Re-raise validation errors without retry
            raise

        # Initialize variables for retry logic
        attempts = 0
        raw_response = ""
        working_messages = messages.copy()

        # Initialize cumulative usage stats
        cumulative_usage = {
            "read_tokens": 0,
            "write_tokens": 0,
            "total_tokens": 0,
            "read_cost": 0.0,
            "write_cost": 0.0,
            "total_cost": 0.0,
            "images": 0,
            "image_cost": 0.0,
            "retry_count": 0,
            "event_id": event_id,
            "model": self.model_name,
            "provider": provider_name,
        }

        # Generation with retry loop
        while attempts < retry_limit:
            try:
                # Generate raw response
                raw_response, usage = self._raw_generate(
                    event_id=f"{event_id}_attempt_{attempts}",
                    system_prompt=system_prompt,
                    messages=working_messages,
                    max_tokens=max_tokens,
                    temp=temp,
                    tools=tools,
                    thinking_budget=thinking_budget,
                )

                # Update usage statistics
                for key in usage:
                    if key in cumulative_usage:
                        cumulative_usage[key] += usage[key]

                # Process and validate response
                if result_processing_function:
                    try:
                        response = result_processing_function(raw_response)
                    except Exception as proc_error:
                        raise LLMFormatError(
                            f"Response processing failed: {proc_error!s}",
                            provider=provider_name,
                            details={
                                "raw_response": raw_response,
                                "error": str(proc_error),
                                "error_type": type(proc_error).__name__,
                            },
                        )
                else:
                    response = raw_response

                # Check for empty response
                if not response and not usage.get("tool_use"):
                    raise LLMContentError(
                        "Model returned empty response with no tool calls",
                        provider=provider_name,
                        details={"usage": usage},
                    )

                # Prepare response message
                ai_message = {"message_type": "ai", "message": response}

                # Add tool use data if present
                if usage.get("tool_use"):
                    ai_message["tool_use"] = usage["tool_use"]

                # Add thinking data if present
                if usage.get("thinking"):
                    ai_message["thinking"] = usage["thinking"]

                # Create updated messages list
                updated_messages = working_messages + [ai_message]

                # Update final usage stats
                cumulative_usage["model"] = self.model_name
                cumulative_usage["provider"] = provider_name
                cumulative_usage["retry_count"] = attempts
                cumulative_usage["successful"] = True

                return response, cumulative_usage, updated_messages

            except (LLMRateLimitError, LLMServiceUnavailableError) as e:
                # These errors are always retried
                attempts += 1
                cumulative_usage["retry_count"] = attempts

                if attempts >= retry_limit:
                    # We've exhausted retries for rate limit/service errors
                    raise LLMProviderError(
                        f"Provider error persisted after {retry_limit} retries: {e}",
                        provider=provider_name,
                        details={
                            "retry_limit": retry_limit,
                            "last_error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

                # Calculate retry delay with exponential backoff
                retry_after = getattr(e, "retry_after", None)
                if retry_after is not None:
                    wait_time = retry_after
                else:
                    wait_time = (2**attempts) * 1.0  # 1, 2, 4, 8 seconds...

                # Add jitter to avoid thundering herd
                wait_time *= 0.75 + 0.5 * random.random()  # 75-125% of wait_time

                # Wait before retrying
                time.sleep(wait_time)

            except LLMMistake as e:
                # Potentially recoverable errors
                attempts += 1
                cumulative_usage["retry_count"] = attempts

                if attempts >= retry_limit:
                    # We've exhausted retries for recoverable errors
                    raise LLMMistake(
                        f"Failed to get valid response after {retry_limit} attempts",
                        error_type="retry_limit_exceeded",
                        provider=provider_name,
                        details={
                            "retry_limit": retry_limit,
                            "last_error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

                # Add failed response to working messages
                working_messages.append(
                    {
                        "message_type": "human",
                        "message": f"Error in previous response: {e!s}. Please try again.",
                    }
                )

            except (
                LLMAuthenticationError,
                LLMValidationError,
                LLMConfigurationError,
            ) as e:
                # Non-recoverable errors - re-raise immediately
                cumulative_usage["error"] = str(e)
                cumulative_usage["error_type"] = type(e).__name__
                cumulative_usage["successful"] = False
                raise

            except Exception as e:
                # Unexpected errors - map to appropriate type and re-raise
                mapped_error = self._map_provider_error(e)
                cumulative_usage["error"] = str(mapped_error)
                cumulative_usage["error_type"] = type(mapped_error).__name__
                cumulative_usage["successful"] = False
                raise mapped_error

        # This should never be reached due to the retry loop
        raise LLMProviderError(
            "Unexpected end of generation method",
            provider=provider_name,
            details={"event_id": event_id},
        )

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """
        Initialize the LLM provider with the specified model.

        Args:
            model_name: Name of the model to use
            **kwargs: Additional provider-specific configuration

        Raises:
            LLMValidationError: If the model is not supported or configuration is invalid
        """
        from ..utils.validators import validate_model_name

        # Validate model name
        validate_model_name(model_name, self.get_supported_models())

        # Set model name and configuration
        self.model_name = model_name
        self.config = kwargs

        # Validate provider configuration
        self._validate_provider_config(self.config)

        # Initialize capability detection
        self._initialize_capabilities()

        # Authenticate with the provider
        self.auth()

    @abstractmethod
    def _validate_provider_config(self, config: Dict[str, Any]) -> None:
        """
        Validate provider-specific configuration.
        Override this method in provider implementations.

        Args:
            config: Provider configuration

        Raises:
            LLMValidationError: If configuration is invalid
        """
        pass

    def _validate_max_tokens(
        self, max_tokens: int, context_window: Optional[int] = None
    ) -> None:
        """
        Validate max_tokens parameter against model's context window.

        Args:
            max_tokens: Maximum tokens to generate
            context_window: Optional context window override (defaults to model's window)

        Raises:
            LLMValidationError: If max_tokens is invalid
        """
        if not isinstance(max_tokens, int):
            raise LLMValidationError(
                "max_tokens must be an integer",
                details={
                    "provided": max_tokens,
                    "provided_type": type(max_tokens).__name__,
                },
            )

        if max_tokens <= 0:
            raise LLMValidationError(
                "max_tokens must be greater than 0", details={"provided": max_tokens}
            )

        # Get model's context window if not provided
        if context_window is None:
            context_window = self.get_context_window()

        # Check if max_tokens exceeds context window
        if max_tokens > context_window:
            raise LLMValidationError(
                f"max_tokens ({max_tokens}) exceeds model context window ({context_window})",
                details={
                    "max_tokens": max_tokens,
                    "context_window": context_window,
                    "model": self.model_name,
                },
            )

    def _validate_temperature(self, temp: float) -> None:
        """
        Validate temperature parameter.

        Args:
            temp: Temperature for generation

        Raises:
            LLMValidationError: If temperature is invalid
        """
        if not isinstance(temp, (int, float)):
            raise LLMValidationError(
                "Temperature must be a number",
                details={"provided": temp, "provided_type": type(temp).__name__},
            )

        # Most providers use temperature in range [0.0, 2.0]
        if temp < 0.0 or temp > 2.0:
            raise LLMValidationError(
                "Temperature should be between 0.0 and 2.0", details={"provided": temp}
            )

    def _validate_tools(self, tools: List[Dict[str, Any]]) -> None:
        """
        Validate tools parameter for function calling.

        Args:
            tools: List of tool/function definitions

        Raises:
            LLMValidationError: If tools format is invalid
        """
        from ..utils.validators import validate_tools

        if not isinstance(tools, list):
            raise LLMValidationError(
                "tools must be a list", details={"provided_type": type(tools).__name__}
            )

        # Use common validator
        try:
            validate_tools(tools)
        except Exception as e:
            raise LLMValidationError(
                f"Invalid tools format: {e!s}", details={"error": str(e)}
            )

        # Check for required capabilities
        if not self.has_tool_calling and not self.has_functions:
            raise LLMValidationError(
                f"Model {self.model_name} does not support function/tool calling",
                details={"model": self.model_name},
            )

    def _initialize_capabilities(self) -> None:
        """
        Initialize capability detection for the selected model.
        This sets up the has_capability attributes.
        """
        from .capabilities import Capability, CapabilityRegistry

        # Get class name for provider detection
        provider_name = self.__class__.__name__.replace("LLM", "").lower()

        # Set up capability detection attributes
        self._provider_capabilities = CapabilityRegistry.get_provider_capabilities(
            provider_name
        )
        self._model_capabilities = CapabilityRegistry.get_model_capabilities(
            self.model_name
        )

        # Initialize capability attributes
        self.has_text_generation = (
            Capability.TEXT_GENERATION in self._model_capabilities
        )
        self.has_text_embeddings = (
            Capability.TEXT_EMBEDDINGS in self._model_capabilities
        )
        self.has_image_input = Capability.IMAGE_INPUT in self._model_capabilities
        self.has_image_generation = (
            Capability.IMAGE_GENERATION in self._model_capabilities
        )
        self.has_streaming = Capability.STREAMING in self._model_capabilities
        self.has_functions = Capability.FUNCTIONS in self._model_capabilities
        self.has_reranking = Capability.RERANKING in self._model_capabilities
        self.has_thinking_budget = (
            Capability.THINKING_BUDGET in self._model_capabilities
        )
        self.has_json_mode = Capability.JSON_MODE in self._model_capabilities
        self.has_tool_calling = Capability.TOOL_CALLING in self._model_capabilities
        self.has_vision = Capability.VISION in self._model_capabilities

    def _map_provider_error(self, error: Exception) -> LLMError:
        """
        Map provider-specific errors to standardized LLM error types.

        This base implementation provides simple string matching to categorize
        common error types. Provider implementations should override this method
        with provider-specific error mapping logic.

        Args:
            error: The original exception from the provider

        Returns:
            LLMError: Appropriate mapped exception with context
        """
        provider_name = self.__class__.__name__.replace("LLM", "")
        error_message = str(error).lower()
        error_type = type(error).__name__

        # Check for common error patterns
        if any(
            term in error_message
            for term in [
                "auth",
                "key",
                "credential",
                "permission",
                "unauthorized",
                "access denied",
                "token",
                "signature",
            ]
        ):
            return LLMAuthenticationError(
                message=f"{provider_name} authentication failed: {error}",
                provider=provider_name,
                details={"original_error": str(error), "error_type": error_type},
            )

        elif any(
            term in error_message
            for term in [
                "rate limit",
                "quota",
                "throttl",
                "too many requests",
                "requests per minute",
                "capacity",
                "overloaded",
            ]
        ):
            return LLMRateLimitError(
                message=f"{provider_name} rate limit exceeded: {error}",
                provider=provider_name,
                retry_after=60,  # Default retry delay
                details={"original_error": str(error), "error_type": error_type},
            )

        elif any(
            term in error_message
            for term in [
                "unavailable",
                "down",
                "outage",
                "maintenance",
                "500",
                "502",
                "503",
                "504",
                "server error",
            ]
        ):
            return LLMServiceUnavailableError(
                message=f"{provider_name} service unavailable: {error}",
                provider=provider_name,
                details={"original_error": str(error), "error_type": error_type},
            )

        elif any(
            term in error_message
            for term in [
                "content",
                "moderation",
                "policy",
                "violation",
                "harmful",
                "inappropriate",
                "unsafe",
                "blocked",
            ]
        ):
            return LLMContentError(
                message=f"{provider_name} content policy violation: {error}",
                provider=provider_name,
                details={"original_error": str(error), "error_type": error_type},
            )

        elif any(
            term in error_message
            for term in [
                "format",
                "json",
                "xml",
                "parse",
                "serialization",
                "deserialization",
                "invalid format",
            ]
        ):
            return LLMFormatError(
                message=f"{provider_name} format error: {error}",
                provider=provider_name,
                details={"original_error": str(error), "error_type": error_type},
            )

        elif any(
            term in error_message
            for term in ["tool", "function", "parameter", "argument", "input"]
        ):
            return LLMToolError(
                message=f"{provider_name} tool error: {error}",
                provider=provider_name,
                details={"original_error": str(error), "error_type": error_type},
            )

        elif any(
            term in error_message
            for term in ["thinking", "reasoning", "thought", "budget"]
        ):
            return LLMThinkingError(
                message=f"{provider_name} thinking error: {error}",
                provider=provider_name,
                details={"original_error": str(error), "error_type": error_type},
            )

        elif any(
            term in error_message
            for term in [
                "invalid",
                "configuration",
                "parameter",
                "setting",
                "value",
                "not supported",
                "unsupported",
                "deprecated",
                "required",
            ]
        ):
            return LLMConfigurationError(
                message=f"{provider_name} configuration error: {error}",
                provider=provider_name,
                details={"original_error": str(error), "error_type": error_type},
            )

        # Default case
        return LLMProviderError(
            message=f"{provider_name} error: {error}",
            provider=provider_name,
            details={"original_error": str(error), "error_type": error_type},
        )

    def _call_with_retry(
        self,
        func: Callable,
        *args: Any,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        retryable_errors: Optional[List[Type[Exception]]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Call a function with exponential backoff retry logic.

        This method provides standardized retry logic for API calls that may
        experience transient failures. It implements exponential backoff with
        jitter to avoid overwhelming services during recovery.

        Args:
            func: The function to call
            *args: Positional arguments to pass to the function
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Multiplier for backoff on each retry
            retryable_errors: List of exception types that should trigger a retry
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Any: The return value from the function

        Raises:
            LLMError: The mapped exception after all retries are exhausted
        """
        if retryable_errors is None:
            retryable_errors = [LLMRateLimitError, LLMServiceUnavailableError]

        retry_count = 0
        backoff_time = initial_backoff
        last_exception = None

        while retry_count <= max_retries:  # <= because first attempt is not a retry
            try:
                return func(*args, **kwargs)

            except tuple(retryable_errors) as e:
                retry_count += 1
                last_exception = e

                if retry_count > max_retries:
                    # We've exhausted retries, re-raise the last error
                    raise

                # Get retry-after time if available (for rate limits)
                retry_after = None
                if isinstance(e, LLMRateLimitError) and hasattr(e, "retry_after"):
                    retry_after = e.retry_after

                # Use retry-after if available, otherwise use exponential backoff
                wait_time = retry_after if retry_after else backoff_time

                # Add jitter to avoid thundering herd problem
                jitter = random.uniform(0, 0.1 * wait_time)
                wait_time += jitter

                # Log retry attempt (if provider has a logger)
                if hasattr(self, "logger") and self.logger:
                    provider_name = self.__class__.__name__
                    logger = self.logger
                    logger.warning(
                        f"Retrying {func.__name__} after error: {e!s}. "
                        f"Attempt {retry_count}/{max_retries} in {wait_time:.2f}s"
                    )

                # Wait before retry
                time.sleep(wait_time)

                # Increase backoff for next retry
                backoff_time *= backoff_factor

            except Exception as e:
                # For other exceptions, map to appropriate error type and raise
                mapped_error = self._map_provider_error(e)
                raise mapped_error

        # Should never get here, but just in case
        if last_exception:
            raise last_exception

        # Generic fallback error if we somehow get here
        provider_name = self.__class__.__name__.replace("LLM", "")
        raise LLMProviderError(
            message=f"Failed after {max_retries} retries", provider=provider_name
        )

    @abstractmethod
    def _format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert standard message format into model-specific format.

        Standard format:
        [
            {
                "message_type": "human",
                "message": "Hello how are you today?",
                "image_paths": ["path/to/image.jpg"],
                "image_urls": ["www.url.com/image.jpg"]
            },
            {
                "message_type": "ai",
                "message": "I am doing great, how are you?"
            }
        ]

        Args:
            messages (List[Dict[str, Any]]): List of messages in standard format

        Returns:
            List[Dict[str, Any]]: Messages formatted for specific model API
        """
        pass

    @abstractmethod
    def auth(self) -> None:
        """
        Authenticate with the LLM provider.
        Should be called during initialization.

        Raises:
            Exception: If authentication fails
        """
        pass

    @abstractmethod
    def _raw_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        top_k: int = 200,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking_budget: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response from the LLM without error correction.

        Args:
            event_id (str): Unique identifier for this generation event
            system_prompt (str): System-level instructions for the model
            messages (List[Dict[str, Any]]): List of messages in the standard format
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1000.
            temp (float, optional): Temperature for generation. Defaults to 0.0.
            top_k (int, optional): Top K tokens to consider. Defaults to 200.
            tools (List[Dict[str, Any]], optional): List of tools to use. Defaults to None.
            thinking_budget (int, optional): Number of tokens allowed for thinking. Defaults to None.

        Returns:
            Tuple[str, Dict[str, Any]]: Tuple containing:
                - str: The generated response
                - Dict[str, Any]: Usage statistics including:
                    - read_tokens: Number of input tokens
                    - write_tokens: Number of output tokens
                    - images: Number of images processed
                    - total_tokens: Total tokens used
                    - read_cost: Cost of input tokens
                    - write_cost: Cost of output tokens
                    - image_cost: Cost of image processing
                    - total_cost: Total cost of the request
        """
        pass

    # Include remaining methods from original file
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_scores: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Rerank a list of documents based on their relevance to a query.

        Args:
            query (str): The search query to rank documents against
            documents (List[str]): List of document texts to rerank
            top_n (int, optional): Number of top documents to return. If None, returns all documents
            return_scores (bool): Whether to include relevance scores in the output
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'ranked_documents' (List[str]): List of reranked document texts
                - 'indices' (List[int]): Original indices of the reranked documents
                - 'scores' (List[float], optional): Relevance scores if return_scores=True
                - 'usage' (Dict): Token usage and cost information

        Raises:
            LLMValidationError: If input parameters are invalid
            NotImplementedError: If the model doesn't support reranking
            LLMProviderError: For provider-specific errors
        """
        from ..exceptions import LLMValidationError

        # Validate that model has reranking capability
        if not self.has_reranking:
            available_models = [
                model
                for model in self.get_supported_models()
                if model in self.RERANKING_MODELS
            ]
            raise NotImplementedError(
                f"Model {self.model_name} does not support document reranking. "
                f"Available reranking models: {', '.join(available_models)}"
            )

        # Validate input parameters
        if not isinstance(query, str):
            raise LLMValidationError(
                "Query must be a string",
                details={"provided_type": type(query).__name__},
            )

        if not query.strip():
            raise LLMValidationError(
                "Query cannot be empty",
                details={"reason": "Empty query would result in meaningless rankings"},
            )

        if not isinstance(documents, list):
            raise LLMValidationError(
                "Documents must be a list",
                details={"provided_type": type(documents).__name__},
            )

        if not documents:
            raise LLMValidationError(
                "Documents list cannot be empty",
                details={"reason": "At least one document is required for reranking"},
            )

        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise LLMValidationError(
                    f"Document at index {i} must be a string, got {type(doc).__name__}",
                    details={"index": i, "provided_type": type(doc).__name__},
                )

        if top_n is not None:
            if not isinstance(top_n, int):
                raise LLMValidationError(
                    "top_n must be an integer or None",
                    details={"provided_type": type(top_n).__name__},
                )

            if top_n <= 0:
                raise LLMValidationError(
                    "top_n must be positive",
                    details={"provided_value": top_n},
                )

            if top_n > len(documents):
                raise LLMValidationError(
                    f"top_n ({top_n}) exceeds the number of documents ({len(documents)})",
                    details={"top_n": top_n, "document_count": len(documents)},
                )

        # This method must be implemented by the provider
        raise NotImplementedError("This method must be implemented by the provider")

        # Unreachable code, but needed for type checking
        return {"ranked_documents": [], "indices": [], "scores": [], "usage": {}}

    def stream_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        functions: Optional[List[Callable[..., Any]]] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Stream a response from the LLM, yielding chunks as they become available.

        Args:
            event_id (str): Unique identifier for this generation event
            system_prompt (str): System-level instructions for the model
            messages (List[Dict[str, Any]]): List of messages in the standard format
            max_tokens (int): Maximum number of tokens to generate
            temp (float): Temperature for generation
            functions (List[Callable]): List of functions the LLM can use
            callback (Callable): Optional callback function for each chunk

        Yields:
            Tuple[str, Dict[str, Any]]: Tuples of (text_chunk, partial_usage_data)

        Raises:
            NotImplementedError: If the provider doesn't implement streaming
        """
        raise NotImplementedError(
            f"Streaming is not implemented for provider {self.__class__.__name__}"
        )

    def _convert_function_to_tool(self, function: Callable[..., Any]) -> Dict[str, Any]:
        """Convert a function to a tool"""
        # Get function name
        name = function.__name__

        # Get function docstring
        docstring = function.__doc__.strip() if function.__doc__ else ""

        # Extract main description (first line or paragraph of docstring)
        description = docstring.split("\n\n")[0].strip()

        # Get function signature
        signature = inspect.signature(function)

        # Build input schema
        properties = {}
        required = []

        for param_name, param in signature.parameters.items():
            # Skip *args, **kwargs, and self/cls parameters
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if param_name in ("self", "cls"):
                continue

            # Add parameter to required list if it has no default value
            if param.default == param.empty:
                required.append(param_name)

            # Determine parameter type
            param_type = "string"  # default type
            if param.annotation != param.empty:
                if param.annotation == str:
                    param_type = "string"
                elif param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                # Add more type mappings as needed

            # Try to extract parameter description from docstring
            param_desc = ""
            if docstring:
                # Look for Args section in docstring
                args_match = re.search(
                    r"Args:(.*?)(?:\n\n|\n[A-Z]|\Z)", docstring, re.DOTALL
                )
                if args_match:
                    args_section = args_match.group(1)
                    # Look for this parameter in the Args section
                    param_match = re.search(
                        rf"\s+{param_name}\s*(?:\(.*?\))?\s*:\s*(.*?)(?:\n\s+\w+\s*:|$)",
                        args_section,
                        re.DOTALL,
                    )
                    if param_match:
                        param_desc = param_match.group(1).strip()

            # Add parameter to properties
            properties[param_name] = {
                "type": param_type,
                "description": param_desc
                or f"Parameter {param_name} for function {name}",
            }

        # Build the tool dictionary
        tool = {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        return tool

    def _convert_functions_to_tools(
        self, functions: List[Callable[..., Any]]
    ) -> List[Dict[str, Any]]:
        """Convert a list of functions to a list of tools"""
        return [self._convert_function_to_tool(function) for function in functions]

        # Second generate method - removed to avoid duplication
        """
        Generate a response with automatic error correction.

        Args:
            event_id: Unique identifier for this generation event
            system_prompt: System-level instructions
            messages: List of messages in standard format
            max_tokens: Maximum tokens to generate
            temp: Temperature for generation
            result_processing_function: Function to validate and transform response
            retry_limit: Maximum number of attempts (including first try)
            functions: List of functions to use as tools
            thinking_budget: Number of tokens allowed for thinking

        Returns:
            Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
                - Processed response (type depends on processing function)
                - Usage statistics
                - Updated messages including the response

        Raises:
            LLMValidationError: If input parameters are invalid
            LLMMistake: If valid response not obtained within retry_limit
            LLMProviderError: For provider-specific errors
        """
        from ..exceptions import LLMValidationError
        from ..utils.validators import (
            validate_max_tokens,
            validate_messages,
            validate_temperature,
            validate_tools,
        )

        # Validate input parameters
        if not isinstance(event_id, str) or not event_id:
            raise LLMValidationError(
                "event_id must be a non-empty string",
                details={"provided_type": type(event_id).__name__},
            )

        if not isinstance(system_prompt, str):
            raise LLMValidationError(
                "system_prompt must be a string",
                details={"provided_type": type(system_prompt).__name__},
            )

        validate_messages(messages)
        validate_temperature(temp)
        validate_max_tokens(max_tokens, self.get_context_window())

        if retry_limit < 1:
            raise LLMValidationError(
                "retry_limit must be at least 1",
                details={"provided_value": retry_limit},
            )

        # Initialize tracking variables
        attempts = 0
        cumulative_usage = {
            "read_tokens": 0,
            "write_tokens": 0,
            "images": 0,
            "total_tokens": 0,
            "read_cost": 0,
            "write_cost": 0,
            "image_cost": 0,
            "total_cost": 0,
            "retry_count": 0,
        }
        raw_response = ""

        # Initialize working messages with a deep copy to avoid modifying original
        working_messages = [message.copy() for message in messages]

        # Handle tool conversion for different providers
        tools: Optional[List[Dict[str, Any]]] = None
        if functions:
            if not isinstance(functions, list):
                raise LLMValidationError(
                    "functions must be a list",
                    details={"provided_type": type(functions).__name__},
                )

            # Different handling for Gemini vs. other models
            if "gemini" in self.model_name:
                # Cast to expected type for Gemini
                tools = functions  # type: ignore
            else:
                tools = self._convert_functions_to_tools(functions)
                validate_tools(tools)

        # Validate thinking budget if model supports it and value is provided
        if thinking_budget is not None:
            if not self.has_thinking_budget:
                raise LLMValidationError(
                    f"Model {self.model_name} does not support thinking budget",
                    details={"model": self.model_name},
                )

            if not isinstance(thinking_budget, int) or thinking_budget <= 0:
                raise LLMValidationError(
                    "thinking_budget must be a positive integer",
                    details={"provided_value": thinking_budget},
                )

        # Generation with retry loop
        while attempts < retry_limit:  # Includes first attempt
            try:
                # Generate raw response, with thinking budget if supported
                if self.has_thinking_budget and thinking_budget is not None:
                    raw_response, usage = self._raw_generate(
                        event_id=f"{event_id}_attempt_{attempts}",
                        system_prompt=system_prompt,
                        messages=working_messages,
                        max_tokens=max_tokens,
                        temp=temp,
                        tools=tools,
                        thinking_budget=thinking_budget,
                    )
                else:
                    raw_response, usage = self._raw_generate(
                        event_id=f"{event_id}_attempt_{attempts}",
                        system_prompt=system_prompt,
                        messages=working_messages,
                        max_tokens=max_tokens,
                        temp=temp,
                        tools=tools,
                    )

                # Update usage statistics
                for key in cumulative_usage:
                    if key in usage:
                        cumulative_usage[key] += usage[key]

                # Track special usage fields
                if "thinking" in usage:
                    cumulative_usage["thinking"] = usage["thinking"]
                    cumulative_usage["thinking_signature"] = usage["thinking_signature"]

                if "tool_use" in usage:
                    cumulative_usage["tool_use"] = usage["tool_use"]

                # Process and validate response
                if result_processing_function:
                    try:
                        response = result_processing_function(raw_response)
                    except Exception as proc_error:
                        # Convert processing errors to LLMMistake for retry
                        from ..exceptions import LLMFormatError

                        raise LLMFormatError(
                            f"Response processing failed: {proc_error!s}",
                            provider=self.__class__.__name__.replace("LLM", ""),
                            details={"raw_response": raw_response},
                        )
                else:
                    response = raw_response

                # Prepare response message
                ai_message = {"message_type": "ai", "message": response}

                # Add tool use information if available
                if "tool_use" in usage:
                    ai_message["tool_use"] = usage["tool_use"]

                # Add thinking information if available
                if "thinking" in usage:
                    ai_message["thinking"] = {
                        "thinking": usage["thinking"],
                        "thinking_signature": usage["thinking_signature"],
                    }

                # Create updated messages list
                updated_messages = working_messages + [ai_message]

                return raw_response, cumulative_usage, updated_messages

            except LLMMistake as e:
                attempts += 1
                cumulative_usage["retry_count"] = attempts

                if attempts >= retry_limit:
                    raise LLMMistake(
                        f"Failed to get valid response after {retry_limit} attempts. "
                        f"Last error: {e!s}",
                        error_type="retry_limit_exceeded",
                        provider=self.__class__.__name__.replace("LLM", ""),
                        details={
                            "retry_limit": retry_limit,
                            "last_error": str(e),
                            "last_response": raw_response,
                        },
                    )

                # Add failed response and error to working messages
                working_messages.extend(
                    [
                        {
                            "message_type": "ai",
                            "message": raw_response,
                            "image_paths": [],
                            "image_urls": [],
                        },
                        {
                            "message_type": "human",
                            "message": f"Error in previous response: {e!s}. Please try again.",
                            "image_paths": [],
                            "image_urls": [],
                        },
                    ]
                )
            except Exception as e:
                # For non-LLMMistake exceptions, propagate them immediately
                # but wrap in a standardized exception type
                from ..exceptions import LLMProviderError

                raise LLMProviderError(
                    f"Provider error during generation: {e!s}",
                    provider=self.__class__.__name__.replace("LLM", ""),
                    details={"original_error": str(e)},
                )

    def get_context_window(self) -> int:
        """
        Get the context window size for the current model.

        Returns:
            int: Maximum number of tokens the model can process

        Raises:
            Exception: If context window information is not available
        """
        try:
            return self.CONTEXT_WINDOW[self.model_name]
        except KeyError:
            raise Exception(
                f"Context window information not available for model {self.model_name}"
            )

    def get_model_costs(self) -> Dict[str, Union[float, None]]:
        """
        Get the cost information for the current model.

        Returns:
            Dict[str, Union[float, None]]: Dictionary containing read_token, write_token, and image_cost

        Raises:
            Exception: If cost information is not available
        """
        try:
            return self.COST_PER_MODEL[self.model_name]
        except KeyError:
            raise Exception(
                f"Cost information not available for model {self.model_name}"
            )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        This is a rough estimate based on words/characters.
        For accurate counts, implement provider-specific token counting.

        Args:
            text (str): Text to estimate tokens for

        Returns:
            int: Estimated number of tokens
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4 + 1

    def check_context_fit(
        self, prompt: str, max_response_tokens: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Check if a prompt will fit in the model's context window.

        Args:
            prompt (str): The input prompt
            max_response_tokens (Optional[int]): Expected maximum response length in tokens

        Returns:
            Tuple[bool, str]: (True if fits, explanation message)
        """
        context_window = self.get_context_window()
        estimated_prompt_tokens = self.estimate_tokens(prompt)
        max_response = (
            max_response_tokens or context_window // 4
        )  # Default to 1/4 of context window

        total_tokens = estimated_prompt_tokens + max_response
        fits = total_tokens <= context_window

        message = (
            f"Estimated usage: {estimated_prompt_tokens} prompt tokens + {max_response} max response tokens = {total_tokens} total\n"
            f"Context window: {context_window} tokens\n"
            f"Status: {'Fits within context window' if fits else 'Exceeds context window'}"
        )

        return fits, message

    def estimate_cost(
        self,
        prompt: str,
        expected_response_tokens: Optional[int] = None,
        images: Optional[List[Tuple[int, int, str]]] = None,
        num_images: Optional[int] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate the cost for a generation based on input and expected output.

        Args:
            prompt (str): The input prompt
            expected_response_tokens (Optional[int]): Expected response length in tokens
            images (Optional[List[Tuple[int, int, str]]]): List of (width, height, detail) for each image
            num_images (Optional[int]): Simple count of images (used if images parameter not provided)

        Returns:
            Tuple[float, Dict[str, float]]: (Total cost, Breakdown of costs)

        Raises:
            Exception: If cost calculation fails
        """
        costs = self.get_model_costs()
        prompt_tokens = self.estimate_tokens(prompt)
        response_tokens = (
            expected_response_tokens or prompt_tokens
        )  # Default to same length as prompt

        # Calculate costs
        read_token_cost = costs["read_token"] or 0.0
        write_token_cost = costs["write_token"] or 0.0
        prompt_cost = prompt_tokens * read_token_cost
        response_cost = response_tokens * write_token_cost

        # Calculate image cost based on available information
        image_count = len(images) if images is not None else (num_images or 0)
        image_cost = (costs["image_cost"] or 0) * image_count

        cost_breakdown = {
            "prompt_cost": round(prompt_cost, 6),
            "response_cost": round(response_cost, 6),
            "image_cost": round(image_cost, 6),
        }
        total_cost = sum(cost_breakdown.values())

        return round(total_cost, 6), cost_breakdown

    def supports_image_input(self) -> bool:
        """
        Check if the current model supports image input.

        Returns:
            bool: True if the model supports image input
        """
        costs = self.get_model_costs()
        return costs["image_cost"] is not None

    def get_supported_models(self) -> List[str]:
        """
        Returns a list of model names supported by this provider.

        Returns:
            List[str]: List of supported model names
        """
        return self.SUPPORTED_MODELS

    def validate_model(self, model_name: str) -> bool:
        """
        Validate if a given model name is supported by this provider.

        Args:
            model_name (str): Name of the model to validate

        Returns:
            bool: True if model is supported, False otherwise
        """
        return model_name in self.SUPPORTED_MODELS

    def classify(
        self,
        messages: List[Dict[str, Any]],
        categories: Dict[str, str],
        examples: Optional[List[Dict[str, Any]]] = None,
        max_options: int = 1,
        system_prompt: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Classify input into predefined categories using the LLM.

        Args:
            messages: Standard message format list
            categories: Dict mapping category names to descriptions
            examples: Optional list of example classifications
            max_options: Maximum number of categories to select (default: 1)
            system_prompt: Optional override for system prompt

        Returns:
            Tuple[List[str], Dict[str, Any]]: Selected category names and usage stats

        Raises:
            LLMValidationError: If input parameters are invalid
            LLMMistake: If classification fails
            LLMProviderError: For provider-specific errors
        """
        from ..exceptions import LLMMistake, LLMValidationError
        from ..utils.validators import validate_categories, validate_messages

        # Validate input parameters
        validate_messages(messages)
        validate_categories(categories)

        # Validate examples if provided
        if examples is not None:
            if not isinstance(examples, list):
                raise LLMValidationError(
                    "Examples must be a list",
                    details={"provided_type": type(examples).__name__},
                )

            required_keys = {"user_input", "doc_str", "selection"}
            for i, example in enumerate(examples):
                if not isinstance(example, dict):
                    raise LLMValidationError(
                        f"Example at index {i} must be a dictionary, got {type(example).__name__}",
                        details={"index": i, "provided_type": type(example).__name__},
                    )

                missing_keys = required_keys - set(example.keys())
                if missing_keys:
                    raise LLMValidationError(
                        f"Example at index {i} is missing required keys: {', '.join(missing_keys)}",
                        details={
                            "index": i,
                            "missing_keys": list(missing_keys),
                            "example": example,
                        },
                    )

                # Validate selection is in categories
                if example["selection"] not in categories:
                    raise LLMValidationError(
                        f"Example at index {i} has invalid selection: '{example['selection']}'. "
                        f"Valid categories are: {', '.join(categories.keys())}",
                        details={
                            "index": i,
                            "invalid_selection": example["selection"],
                            "valid_categories": list(categories.keys()),
                        },
                    )

        # Validate max_options
        if not isinstance(max_options, int):
            raise LLMValidationError(
                "max_options must be an integer",
                details={"provided_type": type(max_options).__name__},
            )

        if max_options < 1:
            raise LLMValidationError(
                "max_options must be at least 1",
                details={"provided_value": max_options},
            )

        if max_options > len(categories):
            raise LLMValidationError(
                f"max_options ({max_options}) exceeds the number of categories ({len(categories)})",
                details={"max_options": max_options, "category_count": len(categories)},
            )

        if system_prompt is not None and not isinstance(system_prompt, str):
            raise LLMValidationError(
                "system_prompt must be a string or None",
                details={"provided_type": type(system_prompt).__name__},
            )

        # Store categories for response parsing
        self.categories = categories

        # Format the classification prompt
        formatted_prompt = self._format_classification_prompt(
            categories=categories, examples=examples, max_options=max_options
        )

        # Add the classification prompt to the system prompt
        effective_system_prompt = self._combine_system_prompts(
            formatted_prompt, system_prompt
        )

        # Create a processing function that will validate and parse the response
        def process_classification_response(response: str) -> List[str]:
            try:
                return self._parse_classification_response(response, max_options)
            except ValueError as e:
                raise LLMMistake(
                    str(e),
                    error_type="classification_format",
                    provider=self.__class__.__name__.replace("LLM", ""),
                    details={"response": response, "max_options": max_options},
                )

        # Generate classification using the retry mechanism
        # The result_processing_function returns List[str]
        result, usage, _ = self.generate(
            event_id=f"classify_{uuid.uuid4()}",
            system_prompt=effective_system_prompt,
            messages=messages,
            max_tokens=100,  # Classifications should be short
            temp=0.0,  # Use deterministic output for classifications
            result_processing_function=process_classification_response,
            retry_limit=3,  # Use default retry limit
        )

        # result is already List[str] due to the result_processing_function
        return result, usage

    def _format_classification_prompt(
        self,
        categories: Dict[str, str],
        examples: Optional[List[Dict[str, Any]]] = None,
        max_options: int = 1,
    ) -> str:
        """Format the classification prompt with categories and examples."""
        # Build category list
        category_text = "\n".join(
            f"{i + 1}. {name}: {desc}"
            for i, (name, desc) in enumerate(categories.items())
        )

        # Build examples text if provided
        examples_text = ""
        if examples:
            examples_text = "\nExamples:\n" + "\n".join(
                f"Input: {ex['user_input']}\n"
                f"Reasoning: {ex['doc_str']}\n"
                f"Selection: {ex['selection']}"
                for ex in examples
            )

        # Build the prompt
        prompt = f"""You are a classification assistant. Your task is to classify the input into {'one category' if max_options == 1 else f'up to {max_options} categories'} from the following list:

{category_text}

{examples_text}

Rules:
1. Output ONLY the category number(s) in XML tags
2. For single selection use: <choice>N</choice> where N is the category number
3. For multiple selections use: <choices>N,M</choices> where N,M are category numbers
4. Numbers must be between 1 and {len(categories)}
5. Select at most {max_options} categories
6. Do not include any other text in your response
7. Do not explain your choice or add any additional formatting

Example outputs:
Single category: <choice>1</choice>
Multiple categories: <choices>1,3</choices>

Analyze the input and select the most appropriate {'category' if max_options == 1 else 'categories'}."""

        return prompt

    def _parse_classification_response(
        self, response: str, max_options: int = 1
    ) -> List[str]:
        """Parse the model's response into a list of category names."""
        # Try both patterns, ignoring everything outside the tags
        single_match = re.search(r"<choice>(\d+)</choice>", response)
        multi_match = re.search(r"<choices>([0-9,]+)</choices>", response)

        if not single_match and not multi_match:
            raise ValueError("Could not find valid choice tags in response")

        # Convert numbers to category names
        if single_match and single_match.group(1):
            numbers = [int(single_match.group(1))]
        elif multi_match and multi_match.group(1):
            numbers = [int(n.strip()) for n in multi_match.group(1).split(",")]
        else:
            raise ValueError("Invalid choice format in response")

        # Validate number of selections
        if len(numbers) > max_options:
            raise ValueError(
                f"Too many categories selected: {len(numbers)} > {max_options}"
            )

        # Validate numbers
        if any(n < 1 or n > len(self.categories) for n in numbers):
            raise ValueError(f"Invalid category numbers in response: {numbers}")

        # Get category names
        category_names = list(self.categories.keys())
        return [category_names[n - 1] for n in numbers]

    def _combine_system_prompts(
        self, classification_prompt: str, user_prompt: Optional[str]
    ) -> str:
        """Combine classification prompt with user's system prompt."""
        if not user_prompt:
            return classification_prompt
        return f"{user_prompt}\n\n{classification_prompt}"

    def classify_from_file(
        self,
        config_path: str,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Classify messages using configuration from a file.

        Args:
            config_path: Path to classification config JSON file
            messages: Messages to classify
            system_prompt: Optional system prompt override

        Returns:
            Tuple of (selected categories, usage statistics)
        """
        # Load and validate config
        config = ClassificationConfig.from_file(config_path)
        config.validate()

        # Perform classification
        return self.classify(
            messages=messages,
            categories=config.categories,
            examples=config.examples,
            max_options=config.max_options,
            system_prompt=system_prompt,
        )
