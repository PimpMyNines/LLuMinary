"""
Base class for LLM providers.
All provider-specific implementations should inherit from this class.
"""
import inspect
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..exceptions import LLMMistake
from .classification.config import ClassificationConfig


class LLM(ABC):
    CONTEXT_WINDOW: Dict[str, int] = {}
    COST_PER_MODEL: Dict[str, Dict[str, Union[float, None]]] = {}
    SUPPORTED_MODELS: list[str] = []
    THINKING_MODELS: list[str] = []
    EMBEDDING_MODELS: list[str] = []  # Models that support embeddings
    RERANKING_MODELS: list[str] = []  # Models that support reranking

    def __init__(self, model_name: str, **kwargs):
        if not self.validate_model(model_name):
            raise ValueError(
                f"Model {model_name} is not supported. Supported models: {self.get_supported_models()}"
            )
        self.model_name = model_name
        self.config = kwargs
        self.auth()

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
        tools: List[Dict[str, Any]] = None,
        thinking_budget: int = None,
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

    def embed(
        self, texts: List[str], model: Optional[str] = None, batch_size: int = 100
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts (List[str]): List of texts to embed
            model (Optional[str]): Specific embedding model to use, defaults to a provider-specific model
            batch_size (int): Number of texts to process in each batch, defaults to 100

        Returns:
            Tuple[List[List[float]], Dict[str, Any]]: Tuple containing:
                - List of embedding vectors (one per input text)
                - Usage statistics including:
                    - tokens: Number of tokens processed
                    - cost: Cost of the embedding operation

        Raises:
            NotImplementedError: If the provider doesn't implement embedding
            ValueError: If embedding fails
        """
        raise NotImplementedError(
            f"Embedding is not implemented for provider {self.__class__.__name__}"
        )

    def supports_embeddings(self) -> bool:
        """
        Check if this provider supports embeddings.

        Returns:
            bool: True if embeddings are supported, False otherwise
        """
        return (
            len(self.EMBEDDING_MODELS) > 0 and self.model_name in self.EMBEDDING_MODELS
        )

    def supports_reranking(self) -> bool:
        """
        Check if this LLM instance supports document reranking.

        Returns:
            bool: True if this model supports document reranking, False otherwise
        """
        return self.model_name in self.RERANKING_MODELS

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = None,
        return_scores: bool = True,
        **kwargs,
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
            NotImplementedError: If the model doesn't support reranking
            ValueError: If there's an issue with the input parameters
        """
        if not self.supports_reranking():
            raise NotImplementedError(
                f"Model {self.model_name} does not support document reranking. "
                f"Available reranking models: {self.RERANKING_MODELS}"
            )

        raise NotImplementedError("This method must be implemented by the provider")

    def stream_generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        functions: List[Callable] = None,
        callback: Callable[[str, Dict[str, Any]], None] = None,
    ):
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

    def _convert_function_to_tool(self, function: Callable) -> dict[str, Any]:
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
        self, functions: List[Callable]
    ) -> List[Dict[str, Any]]:
        """Convert a list of functions to a list of tools"""
        return [self._convert_function_to_tool(function) for function in functions]

    def generate(
        self,
        event_id: str,
        system_prompt: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temp: float = 0.0,
        result_processing_function: Optional[Callable] = None,
        retry_limit: int = 3,
        functions: List[Callable] = None,
        thinking_budget: int = None,
    ) -> Tuple[Any, Dict[str, Any]]:
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
            Tuple[Any, Dict[str, Any]]:
                - Processed response (type depends on processing function)
                - Usage statistics

        Raises:
            Exception: If valid response not obtained within retry_limit
        """
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

        # Initialize working messages with a deep copy to avoid modifying original
        working_messages = [message.copy() for message in messages]
        tools = None
        if functions and not "gemini" in self.model_name:
            tools = self._convert_functions_to_tools(functions)
        elif functions and "gemini" in self.model_name:
            tools = functions

        while attempts < retry_limit:  # Includes first attempt
            try:
                # Generate raw response
                if self.model_name in self.THINKING_MODELS:
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

                if "thinking" in usage:
                    cumulative_usage["thinking"] = usage["thinking"]
                    cumulative_usage["thinking_signature"] = usage["thinking_signature"]

                if "tool_use" in usage:
                    cumulative_usage["tool_use"] = usage["tool_use"]

                # Process and validate response
                if result_processing_function:
                    response = result_processing_function(raw_response)
                else:
                    response = raw_response

                ai_message = {"message_type": "ai", "message": response}

                if "tool_use" in usage:
                    ai_message["tool_use"] = usage["tool_use"]

                if "thinking" in usage:
                    ai_message["thinking"] = {
                        "thinking": usage["thinking"],
                        "thinking_signature": usage["thinking_signature"],
                    }

                updated_messages = working_messages + [ai_message]

                return raw_response, cumulative_usage, updated_messages

            except LLMMistake as e:
                attempts += 1
                cumulative_usage["retry_count"] = attempts

                if attempts >= retry_limit:
                    raise Exception(
                        f"Failed to get valid response after {retry_limit} attempts. "
                        f"Last error: {str(e)}"
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
                            "message": str(e),
                            "image_paths": [],
                            "image_urls": [],
                        },
                    ]
                )
            except Exception as e:
                # For non-LLMMistake exceptions, propagate them immediately
                raise Exception(str(e))

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
        prompt_cost = prompt_tokens * costs["read_token"]
        response_cost = response_tokens * costs["write_token"]

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

    def get_supported_models(self) -> list[str]:
        """
        Returns a list of model names supported by this provider.

        Returns:
            list[str]: List of supported model names
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
        """
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
                raise LLMMistake(str(e))

        # Generate classification using the retry mechanism
        selected_categories, usage, _ = self.generate(
            event_id=f"classify_{uuid.uuid4()}",
            system_prompt=effective_system_prompt,
            messages=messages,
            max_tokens=100,  # Classifications should be short
            temp=0.0,  # Use deterministic output for classifications
            result_processing_function=process_classification_response,
            retry_limit=3,  # Use default retry limit
        )

        return selected_categories, usage

    def _format_classification_prompt(
        self,
        categories: Dict[str, str],
        examples: Optional[List[Dict[str, Any]]] = None,
        max_options: int = 1,
    ) -> str:
        """Format the classification prompt with categories and examples."""
        # Build category list
        category_text = "\n".join(
            f"{i+1}. {name}: {desc}"
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
            raise ValueError(f"Could not find valid choice tags in response")

        # Convert numbers to category names
        if single_match:
            numbers = [int(single_match.group(1))]
        else:
            numbers = [int(n.strip()) for n in multi_match.group(1).split(",")]

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
