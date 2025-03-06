#!/usr/bin/env python3
"""
Create a clean base.py file without duplicates or errors.
"""

# This is a clean version of the LLM base class

BASE_CLASS = '''"""
Base class for LLM providers.
All provider-specific implementations should inherit from this class.
"""

import inspect
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Tuple, Union

from .classification.config import ClassificationConfig
from ..exceptions import LLMMistake


class LLM(ABC):
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
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate a response with retry logic and error handling.
        
        Args:
            event_id: Unique identifier for this generation
            system_prompt: Instructions for the model
            messages: List of messages to process
            max_tokens: Maximum tokens to generate
            temp: Temperature for generation
            tools: List of tools/functions to use
            retry_limit: Maximum number of retries
            result_processing_function: Function to process the result
            
        Returns:
            Tuple containing:
            - Generated response text
            - Usage statistics
            - Updated messages including the response
        """
        from ..exceptions import LLMValidationError
        from ..utils.validators import validate_tools
        
        # Validate parameters
        if not isinstance(messages, list):
            raise LLMValidationError(
                "messages must be a list",
                details={"provided_type": type(messages).__name__},
            )
            
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
        }
        
        # If tools is provided, validate it
        if tools is not None:
            if not isinstance(tools, list):
                raise LLMValidationError(
                    "tools must be a list",
                    details={"provided_type": type(tools).__name__},
                )
            validate_tools(tools)
        
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
                
                # Create updated messages list
                updated_messages = working_messages + [ai_message]
                
                return response, cumulative_usage, updated_messages
                
            except LLMMistake as e:
                attempts += 1
                cumulative_usage["retry_count"] = attempts
                
                if attempts >= retry_limit:
                    raise LLMMistake(
                        f"Failed to get valid response after {retry_limit} attempts",
                        error_type="retry_limit_exceeded",
                        provider=self.__class__.__name__.replace("LLM", ""),
                        details={"retry_limit": retry_limit, "last_error": str(e)},
                    )
                
                # Add failed response to working messages
                working_messages.append({
                    "message_type": "human",
                    "message": f"Error in previous response: {e!s}. Please try again.",
                })
        
        # This should never be reached due to the retry loop
        raise Exception("Unexpected end of generation method")

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
'''

def create_clean_base():
    """Create a clean version of base.py."""
    try:
        # Read the existing file to get the rest of the content
        with open('src/lluminary/models/base.py', 'r') as f:
            full_content = f.read()
            
        # Find the rerank method onwards
        rerank_pos = full_content.find('    def rerank(')
        
        if rerank_pos == -1:
            print("Could not find rerank method, saving only the basic version")
            rest_content = ""
        else:
            rest_content = full_content[rerank_pos:]
            
        # Create the clean file
        with open('src/lluminary/models/base.py', 'w') as f:
            f.write(BASE_CLASS + rest_content)
            
        print("Successfully created a clean base.py file")
        
    except Exception as e:
        print(f"Error creating clean base.py: {e}")

if __name__ == "__main__":
    create_clean_base()