"""
Template for creating new LLM provider integration.
Copy and adapt this template to implement a new provider.

This template includes all the required methods and configurations needed to
implement a new provider in LLuMinary. Provider implementations should define
their capabilities, supported models, and implement the required methods.
"""

import base64
import json
import os
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple, Union

from ...exceptions import LLMMistake, LLMValidationError
from ..base import LLM


def validate_messages(messages: List[Dict[str, Any]]) -> None:
    """
    Validate messages format.

    Args:
        messages: List of message dictionaries to validate

    Raises:
        LLMValidationError: If messages are not properly formatted
    """
    if not isinstance(messages, list):
        raise LLMValidationError(
            "Messages must be a list",
            details={"provided_type": type(messages).__name__},
        )

    if not messages:
        raise LLMValidationError(
            "Messages list cannot be empty",
            details={"reason": "At least one message is required"},
        )

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise LLMValidationError(
                f"Message at index {i} must be a dictionary",
                details={"index": i, "provided_type": type(msg).__name__},
            )

        if "message_type" not in msg:
            raise LLMValidationError(
                f"Message at index {i} missing required 'message_type' field",
                details={"index": i, "fields_present": list(msg.keys())},
            )

        if "message" not in msg:
            raise LLMValidationError(
                f"Message at index {i} missing required 'message' field",
                details={"index": i, "fields_present": list(msg.keys())},
            )


class ProviderNameLLM(LLM):
    """
    Implementation of the LLM interface for the Provider Name API.
    Replace ProviderName with your provider's name (e.g., CohereLLM).

    Capabilities:
        - Define which capabilities your provider supports by registering them
          in the CapabilityRegistry or by overriding the _initialize_capabilities method.
        - Common capabilities include: TEXT_GENERATION, TEXT_EMBEDDINGS, IMAGE_INPUT,
          STREAMING, FUNCTIONS, RERANKING, etc.

    Authentication:
        - Implement the auth method to handle provider-specific authentication.
        - Support both environment variables and AWS Secrets Manager.

    Required Methods:
        - _raw_generate: Main generation method
        - _format_messages_for_model: Convert standard messages to provider format
        - auth: Authentication method

    Optional Methods:
        - embed: If your provider supports embeddings
        - rerank: If your provider supports document reranking
        - stream_generate: If your provider supports streaming
    """

    # Define context window sizes for this provider's models
    CONTEXT_WINDOW: ClassVar[Dict[str, int]] = {
        "provider-model-1": 16000,  # Example model's context window
        "provider-model-2": 32000,  # Example model's context window
    }

    # Define token costs for this provider's models
    COST_PER_MODEL: ClassVar[Dict[str, Dict[str, Union[float, None]]]] = {
        "provider-model-1": {
            "read_token": 0.00001,  # Example cost per input token
            "write_token": 0.00003,  # Example cost per output token
            "image_token": 0.00004,  # Example cost per image token
        },
        "provider-model-2": {
            "read_token": 0.00002,
            "write_token": 0.00006,
            "image_token": 0.00008,
        },
    }

    # List of supported models (used for validation)
    SUPPORTED_MODELS: ClassVar[List[str]] = [
        "provider-model-1",
        "provider-model-2",
    ]

    # List of models supporting "thinking" capability
    THINKING_MODELS: ClassVar[List[str]] = [
        "provider-model-2",  # Only if this model supports thinking
    ]

    # Define embedding dimensions for models
    EMBEDDING_DIMENSIONS: ClassVar[Dict[str, int]] = {
        "provider-model-1": 1536,
        "provider-model-2": 3072,
    }

    # Define models that support reranking
    RERANKING_MODELS: ClassVar[List[str]] = [
        "provider-model-2",
    ]

    def __init__(self, model_name: str, **kwargs):
        """Initialize the provider-specific client."""
        super().__init__(model_name, **kwargs)

        # Get other configuration options
        self.api_base = kwargs.get("api_base")
        self.timeout = kwargs.get("timeout", 30)

    def auth(self) -> None:
        """
        Authenticate with the provider's API.

        This method retrieves the API key from one of these sources, in order:
        1. Environment variable PROVIDER_API_KEY
        2. Constructor config parameter "api_key"
        3. AWS Secrets Manager with key "provider_api_key"

        Raises:
            LLMAuthenticationError: If no valid API key is found
        """
        # Get API key from environment variables
        self.api_key = os.environ.get("PROVIDER_API_KEY")

        # Check config
        if not self.api_key and "api_key" in self.config:
            self.api_key = self.config["api_key"]

        # Fallback to AWS Secrets Manager if available
        if not self.api_key:
            try:
                from ...utils.aws import get_secret

                secret_value = get_secret("provider_api_key")
                if isinstance(secret_value, dict) and "api_key" in secret_value:
                    self.api_key = str(secret_value["api_key"])
                else:
                    self.api_key = (
                        str(secret_value) if secret_value is not None else None
                    )
            except Exception as e:
                # Log error but don't raise yet
                print(f"Error accessing AWS Secrets Manager: {e!s}")

        # Validate API key
        if not self.api_key:
            from ...exceptions import LLMAuthenticationError

            raise LLMAuthenticationError(
                "No API key found for provider. Please set the PROVIDER_API_KEY "
                "environment variable, pass api_key in the config, or set up AWS "
                "Secrets Manager with the key provider_api_key.",
                provider="provider_name",
                details={
                    "env_var": "PROVIDER_API_KEY",
                    "config_key": "api_key",
                    "secret_name": "provider_api_key",
                },
            )

        # Validate API key format
        from ...utils.validators import validate_api_key

        try:
            validate_api_key(self.api_key, "provider_name")
        except LLMValidationError as e:
            from ...exceptions import LLMAuthenticationError

            raise LLMAuthenticationError(
                f"Invalid API key format: {e!s}",
                provider="provider_name",
                details=e.details if hasattr(e, "details") else {},
            )

        # Initialize provider's official SDK or client library
        # Example: self.client = provider_sdk.Client(api_key=self.api_key)

    def _format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert standard message format to provider-specific format.
        """
        formatted_messages = []

        for message in messages:
            message_type = message["message_type"]
            content = message["message"]

            # Map message types to provider-specific roles
            if message_type == "human":
                role = "user"  # or provider's equivalent
            elif message_type == "ai":
                role = "assistant"  # or provider's equivalent
            elif message_type == "tool_result":
                # Handle tool results if provider supports tool use
                # This will likely need custom handling per provider
                role = "tool"  # or provider's equivalent

            # Handle images if present and provider supports images
            images = []
            if self.supports_image_input() and message_type == "human":
                # Process image paths if present
                for img_path in message.get("image_paths", []):
                    # Convert and encode image as required by provider
                    encoded_image = self._process_image_file(img_path)
                    images.append(encoded_image)

                # Process image URLs if present
                for img_url in message.get("image_urls", []):
                    # Download and encode image as required by provider
                    encoded_image = self._process_image_url(img_url)
                    images.append(encoded_image)

            # Format message in provider-specific structure
            formatted_message = {"role": role, "content": content}

            # Add images if present
            if images:
                # Adapt this to match the provider's API for multi-modal inputs
                formatted_message["images"] = images

            formatted_messages.append(formatted_message)

        return formatted_messages

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
        Generate a response from the provider's LLM.

        Args:
            event_id: Unique identifier for this generation event
            system_prompt: System-level instructions for the model
            messages: List of messages in the standard format
            max_tokens: Maximum number of tokens to generate
            temp: Temperature for generation
            top_k: Top K tokens to consider
            tools: Optional list of tools the model can use
            thinking_budget: Optional number of tokens for thinking (if supported)

        Returns:
            Tuple containing the generated text response and usage statistics

        Raises:
            LLMValidationError: If input parameters are invalid
            LLMMistake: If generation fails
        """
        # Validate input parameters
        from ...utils.validators import (
            validate_max_tokens,
            validate_temperature,
            validate_tools,
        )

        validate_messages(messages)
        validate_temperature(temp)
        validate_max_tokens(max_tokens, self.get_context_window())

        if tools:
            validate_tools(tools)

        if thinking_budget is not None and not self.has_thinking_budget:
            from ...exceptions import LLMValidationError

            raise LLMValidationError(
                f"Model {self.model_name} does not support thinking budget",
                details={"model": self.model_name},
            )

        # Format messages for the provider API
        _ = self._format_messages_for_model(messages)

        # Handle system prompt (provider specific)
        # Some providers have a system message type, others prepend to the messages

        # Track image count for cost calculations
        image_count = 0
        for message in messages:
            if message.get("message_type") == "human":
                image_paths = message.get("image_paths", [])
                image_urls = message.get("image_urls", [])

                # Validate image paths and URLs if provider supports images
                if self.has_image_input:
                    from ...utils.validators import (
                        validate_image_path,
                        validate_image_url,
                    )

                    for img_path in image_paths:
                        validate_image_path(img_path)

                    for img_url in image_urls:
                        validate_image_url(img_url)

                image_count += len(image_paths)
                image_count += len(image_urls)

        # Call the provider's API
        try:
            # This is a placeholder - replace with actual API call
            # Example:
            # response = self.client.chat.completions.create(
            #     model=self.model_name,
            #     messages=formatted_messages,
            #     max_tokens=max_tokens,
            #     temperature=temp,
            #     tools=tools,
            #     system=system_prompt,
            # )

            # For template purposes, we'll simulate a response
            response: Dict[str, Any] = {
                "choices": [
                    {"message": {"content": "This is a placeholder response."}}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            }

            # Extract text response from provider-specific response format
            result = response["choices"][0]["message"]["content"]

            # Calculate usage statistics
            read_tokens = response["usage"]["prompt_tokens"]
            write_tokens = response["usage"]["completion_tokens"]
            total_tokens = response["usage"]["total_tokens"]

            # Calculate costs
            model_costs = self.get_model_costs()
            read_cost = read_tokens * (model_costs["read_token"] or 0.0)
            write_cost = write_tokens * (model_costs["write_token"] or 0.0)
            image_cost = 0.0
            if (
                image_count > 0
                and "image_token" in model_costs
                and model_costs["image_token"] is not None
            ):
                image_cost = image_count * model_costs["image_token"]

            total_cost = read_cost + write_cost + image_cost

            # Return response and usage statistics
            usage = {
                "read_tokens": read_tokens,
                "write_tokens": write_tokens,
                "images": image_count,
                "total_tokens": total_tokens,
                "read_cost": read_cost,
                "write_cost": write_cost,
                "image_cost": image_cost,
                "total_cost": total_cost,
                "event_id": event_id,
                "model": self.model_name,
                # Add tool use information if tools were used
                "tool_use": {},  # Add actual tool use data here if available
            }

            return result, usage

        except Exception as e:
            # Handle provider-specific errors and convert to standard errors
            raise LLMMistake(
                f"Error generating text with {self.__class__.__name__}: {e!s}",
                error_type="api_error",
                provider=self.__class__.__name__,
                details={"original_error": str(e)},
            )

    def supports_image_input(self) -> bool:
        """Check if the current model supports image inputs."""
        # Return True if this provider's model supports images
        return self.model_name in ["provider-model-2"]  # Example

    def _process_image_file(self, image_path: str) -> Any:
        """
        Process and encode an image file for the provider's API.

        Returns:
            Provider-specific image encoding
        """
        # This is a placeholder - implement based on provider's requirements
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            raise LLMMistake(
                f"Error processing image file: {e!s}",
                error_type="image_processing_error",
                provider=self.__class__.__name__,
                details={"path": image_path, "original_error": str(e)},
            )

    def _process_image_url(self, image_url: str) -> Any:
        """
        Download and encode an image from URL for the provider's API.

        Returns:
            Provider-specific image encoding
        """
        # This is a placeholder - implement based on provider's requirements
        try:
            import requests

            response = requests.get(image_url)
            response.raise_for_status()
            return base64.b64encode(response.content).decode("utf-8")
        except Exception as e:
            raise LLMMistake(
                f"Error processing image URL: {e!s}",
                error_type="image_url_error",
                provider=self.__class__.__name__,
                details={"url": image_url, "original_error": str(e)},
            )

    def get_context_window(self, default: int = 4096) -> int:
        """
        Get the context window size for the current model.

        Args:
            default: Default context window size if not defined

        Returns:
            Context window size in tokens
        """
        return self.CONTEXT_WINDOW.get(self.model_name, default)

    def get_model_costs(self) -> Dict[str, Union[float, None]]:
        """
        Get the token costs for the current model.

        Returns:
            Dictionary with cost per token type
        """
        default_costs: Dict[str, Union[float, None]] = {
            "read_token": 0.00001,
            "write_token": 0.00002,
            "image_token": 0.00004,
        }

        try:
            return self.COST_PER_MODEL[self.model_name]
        except KeyError:
            return default_costs

    def count_tokens_from_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            int: Total token count
        """
        # This is a placeholder - in a real implementation, use a tokenizer
        # Example:
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # token_count = 0
        # for message in messages:
        #     token_count += len(tokenizer.encode(message["message"]))
        # return token_count

        # For template purposes, use a simple approximation
        token_count = 0
        for message in messages:
            # Count ~4 chars per token as a rough estimate
            token_count += len(str(message.get("message", ""))) // 4
        return token_count

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count before API call.

        Args:
            text: Text to estimate tokens for

        Returns:
            int: Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4 + 1

    def get_actual_tokens(
        self, messages: List[Dict[str, Any]], api_response: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Get actual token count from API response.

        Args:
            messages: List of messages sent to the API
            api_response: Response from the API

        Returns:
            Dict[str, int]: Dictionary with token usage information
        """
        usage = api_response.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Generate embeddings for the provided texts.

        Args:
            texts (List[str]): List of texts to embed
            model (Optional[str]): Specific embedding model to use
            batch_size (int): Number of texts to process in each batch
            **kwargs (Any): Additional provider-specific parameters

        Returns:
            Tuple[List[List[float]], Dict[str, Any]]: Tuple containing:
                - List of embedding vectors (one per input text)
                - Usage statistics including tokens, cost, and dimensions
        """
        try:
            if not texts or not all(isinstance(t, str) for t in texts):
                raise ValueError("Texts must be a non-empty list of strings")

            # If no texts to embed, return empty results
            if not texts:
                return [], {"total_tokens": 0, "total_cost": 0.0, "dimensions": 0}

            # Use the specified model or default
            embedding_model = model or self.model_name

            # Validate that we have a valid embedding model
            if not hasattr(self, "EMBEDDING_DIMENSIONS"):
                raise ValueError(
                    f"Embedding dimensions not defined for {self.__class__.__name__}"
                )

            dimension = self.EMBEDDING_DIMENSIONS.get(embedding_model, 1536)
            if dimension is None:
                raise ValueError(f"Model {embedding_model} does not support embeddings")

            # Create batches to avoid rate limits and large requests
            _ = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

            # For template purposes, simulate embeddings
            embeddings = []
            for _ in texts:
                # Generate a simple deterministic embedding
                embedding = [0.1 * (i % 10) for i in range(dimension)]
                embeddings.append(embedding)

            # Calculate token usage and cost
            total_tokens = sum(len(text.split()) for text in texts)
            total_cost = 0.0001 * len(texts)  # Example cost calculation

            return embeddings, {
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "dimensions": dimension,
                "model": embedding_model,
            }

        except Exception as e:
            raise LLMMistake(
                f"Error generating embeddings with {self.__class__.__name__}: {e!s}",
                error_type="embedding_error",
                provider=self.__class__.__name__,
                details={
                    "original_error": str(e),
                    "texts_count": len(texts) if texts else 0,
                },
            )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_scores: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The query string
            documents: List of documents to rerank
            top_n: Optional number of top results to return
            return_scores: Whether to return relevance scores
            **kwargs: Additional parameters (e.g., top_n)

        Returns:
            Dict[str, Any]: Dictionary containing:
                - results: List of reranked documents or document/score pairs
                - usage: Usage statistics
        """
        try:
            # Validate inputs
            if not query:
                raise LLMValidationError("Query cannot be empty")
            if not documents:
                raise LLMValidationError("Documents list cannot be empty")

            # Get optional top_n parameter
            top_n = top_n or len(documents)

            # This is a placeholder - in a real implementation, call the API
            # For now, just return the documents in the same order with dummy scores
            results = []
            for i, doc in enumerate(documents[:top_n]):
                if return_scores:
                    results.append({"document": doc, "score": 1.0 - (i * 0.1)})
                else:
                    results.append({"document": doc})

            # Calculate token usage (placeholder)
            total_tokens = len(query.split()) + sum(
                len(doc.split()) for doc in documents
            )

            # Return results and usage stats
            return {
                "results": results,
                "usage": {
                    "total_tokens": total_tokens,
                    "total_cost": 0.00002 * total_tokens,
                },
            }

        except Exception as e:
            # Convert to LLMMistake if not already
            if not isinstance(e, LLMMistake):
                raise LLMMistake(
                    f"Error during reranking with {self.__class__.__name__}: {e!s}",
                    error_type="reranking_error",
                    provider=self.__class__.__name__,
                    details={
                        "original_error": str(e),
                        "query_length": len(query) if isinstance(query, str) else 0,
                        "documents_count": len(documents) if documents else 0,
                    },
                )
            raise

    def _format_tool_for_provider(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert standard tool format to provider-specific format.

        Args:
            tool: Standard format tool

        Returns:
            Provider-specific format
        """
        if tool.get("type") == "function":
            function = tool["function"]
            return {
                "function_name": function["name"],
                "function_description": function["description"],
                "parameters": function["parameters"],
            }
        return tool

    def _parse_tool_response(self, provider_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse provider-specific tool response to standard format.

        Args:
            provider_response: Provider-specific response

        Returns:
            Standard tool response format
        """
        if "function_call" in provider_response:
            function_call = provider_response["function_call"]
            return {
                "name": function_call["name"],
                "arguments": json.loads(function_call["arguments"]),
            }
        return {}

    def _validate_tool_parameters(self, tool: Dict[str, Any]) -> bool:
        """
        Validate tool parameters for provider compatibility.

        Args:
            tool: Tool configuration

        Returns:
            True if valid, otherwise raises ValueError
        """
        if not tool.get("type"):
            raise ValueError("Tool must have a 'type' field")

        if tool["type"] == "function":
            if not tool.get("function"):
                raise ValueError("Function tool must have a 'function' field")

            function = tool["function"]
            if not function.get("name"):
                raise ValueError("Function must have a 'name'")

            if not function.get("parameters"):
                raise ValueError("Function must have 'parameters'")

            if function["parameters"].get("type") != "object":
                raise ValueError("Parameters must have type 'object'")

        return True

    def calculate_image_tokens(self, width: int, height: int) -> int:
        """
        Calculate tokens required for an image based on dimensions.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Token count
        """
        # Simple algorithm example - providers have different formulas
        return (width * height) // 1024

    def _stream_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        top_k: int = 200,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking_budget: Optional[int] = None,
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Stream generate responses from the provider's LLM.

        Yields:
            Tuples of (chunk, usage_stats)
        """
        # Format messages for the provider API
        _ = self._format_messages_for_model(messages)

        # This is a placeholder - in a real implementation, call the streaming API
        # Example:
        # for chunk in self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=formatted_messages,
        #     max_tokens=max_tokens,
        #     temperature=temp,
        #     tools=tools,
        #     system=system_prompt,
        #     stream=True
        # ):
        #     # Process chunk and yield with usage stats

        # For template purposes, yield simulated chunks
        base_usage = {
            "read_tokens": 10,
            "write_tokens": 0,
            "total_tokens": 10,
            "event_id": event_id,
            "model": self.model_name,
            "tool_use": {},
        }

        # First chunk
        usage1 = base_usage.copy()
        usage1["write_tokens"] = 2
        yield "Hello", usage1

        # Second chunk
        usage2 = base_usage.copy()
        usage2["write_tokens"] = 3
        yield " world", usage2

        # Final chunk with is_complete flag
        final_usage = base_usage.copy()
        final_usage["write_tokens"] = 5
        final_usage["is_complete"] = True
        yield "!", final_usage


# Uncomment to register this provider
# Register your provider for automatic discovery
# register_provider("provider_name", ProviderNameLLM)
