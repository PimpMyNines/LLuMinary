"""
Implementation of the LLM interface for the Cohere API.
"""

import base64
import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import requests
from PIL import Image

from ...exceptions import LLMMistake
from ...utils.aws import get_secret
from ..base import LLM
from ..router import register_provider


class CohereLLM(LLM):
    """
    Implementation of the LLM interface for the Cohere API.
    Supports Cohere's command and chat models.
    """

    # Define context window sizes for Cohere models
    CONTEXT_WINDOW = {
        "command": 4096,
        "command-light": 4096,
        "command-r": 128000,
        "command-r-plus": 128000,
    }

    # Define token costs for Cohere models (approximate values)
    COST_PER_MODEL = {
        "command": {
            "read_token": 0.0000015,  # $1.50 per million tokens
            "write_token": 0.0000015,  # $1.50 per million tokens
            "image_token": 0.00001,  # Approximated for image processing
        },
        "command-light": {
            "read_token": 0.0000003,  # $0.30 per million tokens
            "write_token": 0.0000003,  # $0.30 per million tokens
            "image_token": 0.000002,  # Approximated for image processing
        },
        "command-r": {
            "read_token": 0.0000015,  # $1.50 per million tokens
            "write_token": 0.0000015,  # $1.50 per million tokens
            "image_token": 0.00001,  # Approximated for image processing
        },
        "command-r-plus": {
            "read_token": 0.000003,  # $3.00 per million tokens
            "write_token": 0.000003,  # $3.00 per million tokens
            "image_token": 0.00002,  # Approximated for image processing
        },
    }

    # List of supported models
    SUPPORTED_MODELS = [
        "command",
        "command-light",
        "command-r",
        "command-r-plus",
    ]

    # Models that support embeddings
    EMBEDDING_MODELS = [
        "embed-english-v3.0",
        "embed-multilingual-v3.0",
        "embed-english-light-v3.0",
        "embed-multilingual-light-v3.0",
    ]

    # Models that support reranking
    RERANKING_MODELS = ["rerank-english-v3.0", "rerank-multilingual-v3.0"]

    # Default embedding model
    DEFAULT_EMBEDDING_MODEL = "embed-english-v3.0"

    # Default reranking model
    DEFAULT_RERANKING_MODEL = "rerank-english-v3.0"

    # Embedding costs (per million tokens)
    embedding_costs = {
        "embed-english-v3.0": 0.00001,  # $0.01 per million tokens
        "embed-multilingual-v3.0": 0.00001,  # $0.01 per million tokens
        "embed-english-light-v3.0": 0.000001,  # $0.001 per million tokens
        "embed-multilingual-light-v3.0": 0.000001,  # $0.001 per million tokens
    }

    # Reranking costs (per million tokens)
    reranking_costs = {
        "rerank-english-v3.0": 0.00001,  # $0.01 per million tokens
        "rerank-multilingual-v3.0": 0.00001,  # $0.01 per million tokens
    }

    # Models that support "thinking"
    THINKING_MODELS = []  # Cohere doesn't officially support thinking yet

    def __init__(self, model_name: str, **kwargs):
        """Initialize the Cohere client."""
        super().__init__(model_name, **kwargs)
        self.api_base = kwargs.get("api_base", "https://api.cohere.ai/v1")
        self.timeout = kwargs.get("timeout", 60)
        self.api_version = kwargs.get("api_version", "2023-05-01")

    def auth(self) -> None:
        """
        Authenticate with the Cohere API.
        """
        # Get API key from environment variables
        self.api_key = os.environ.get("COHERE_API_KEY")

        # Fallback to AWS Secrets Manager if available
        if not self.api_key and "aws_secret_name" in self.config:
            secret_name = self.config.get("aws_secret_name", "cohere_api_key")
            self.api_key = self._get_api_key_from_aws(secret_name)

        if not self.api_key:
            raise ValueError(
                "API key not found. Set the COHERE_API_KEY environment variable or configure AWS Secrets Manager."
            )

        # Initialize the HTTP session for API calls
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _get_api_key_from_aws(self, secret_name: str) -> Optional[str]:
        """
        Get the API key from AWS Secrets Manager.

        Args:
            secret_name (str): Name of the secret in AWS Secrets Manager

        Returns:
            Optional[str]: API key if found, None otherwise
        """
        try:
            # Get AWS profile and region from config if available
            aws_profile = self.config.get(
                "aws_profile", self.config.get("profile_name")
            )
            aws_region = self.config.get("aws_region")

            # Get the secret using the core get_secret function
            # Avoiding import issues by using the imported function
            secret_data = get_secret(
                secret_id=secret_name,
                required_keys=["api_key"],
                aws_profile=aws_profile,
                aws_region=aws_region,
            )

            if isinstance(secret_data, dict) and "api_key" in secret_data:
                return str(secret_data["api_key"])
            return None
        except Exception:
            # If any error occurs, return None and let the caller handle it
            return None

    def _format_messages_for_model(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert standard message format to Cohere chat format.
        """
        formatted_messages = []

        for message in messages:
            message_type = message["message_type"]
            content = message["message"]

            # Map message types to Cohere roles
            if message_type == "human":
                role = "USER"
            elif message_type == "ai":
                role = "CHATBOT"
            elif message_type == "tool_result":
                # Handle tool results (Cohere doesn't have a native tool role)
                # We'll format tool results as system messages
                role = "SYSTEM"
                tool_result = message.get("tool_result", {})
                content = f"Tool Result: {json.dumps(tool_result)}"
            else:
                # Default to system for other types
                role = "SYSTEM"

            # Create base message
            formatted_message = {"role": role, "message": content}

            # Handle images if present and in a user message
            if self.supports_image_input() and message_type == "human":
                attachments = []

                # Process image paths
                for img_path in message.get("image_paths", []):
                    try:
                        image_data = self._process_image_file(img_path)
                        if image_data:
                            attachments.append(
                                {
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_data,
                                    }
                                }
                            )
                    except Exception as e:
                        print(f"Warning: Failed to process image {img_path}: {e!s}")

                # Process image URLs
                for img_url in message.get("image_urls", []):
                    try:
                        attachments.append({"source": {"type": "url", "url": img_url}})
                    except Exception as e:
                        print(f"Warning: Failed to add image URL {img_url}: {e!s}")

                # Add attachments if present
                if attachments:
                    formatted_message["attachments"] = attachments

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
        Generate a response using the Cohere chat API.

        Args:
            event_id (str): Unique identifier for this generation event
            system_prompt (str): System-level instructions for the model
            messages (List[Dict[str, Any]]): List of messages in the standard format
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 1000.
            temp (float, optional): Temperature for generation. Defaults to 0.0.
            top_k (int, optional): Top-k sampling parameter. Defaults to 200.
            tools (Optional[List[Dict[str, Any]]], optional): Function/tool definitions. Defaults to None.
            thinking_budget (Optional[int], optional): Budget for thinking steps. Defaults to None.

        Returns:
            Tuple[str, Dict[str, Any]]: Generated text and usage statistics
        """
        # Convert messages to Cohere format
        formatted_messages = self._format_messages_for_model(messages)

        # Add system prompt as a SYSTEM message if provided
        if system_prompt:
            formatted_messages.insert(0, {"role": "SYSTEM", "message": system_prompt})

        # Count images for cost calculation
        image_count = 0
        for message in messages:
            if message.get("message_type") == "human":
                image_count += len(message.get("image_paths", []))
                image_count += len(message.get("image_urls", []))

        # Prepare request data
        request_data = {
            "model": self.model_name,
            "message": formatted_messages[-1]["message"] if formatted_messages else "",
            "chat_history": (
                formatted_messages[:-1] if len(formatted_messages) > 1 else []
            ),
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": False,
        }

        # Add tools if provided
        if tools:
            # Format tools for Cohere
            cohere_tools = []
            for tool in tools:
                cohere_tool = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameter_definitions": {},
                }

                # Add parameters
                if "parameters" in tool:
                    parameters = tool["parameters"]
                    if "properties" in parameters:
                        for param_name, param_def in parameters["properties"].items():
                            cohere_tool["parameter_definitions"][param_name] = {
                                "description": param_def.get("description", ""),
                                "type": param_def.get("type", "string"),
                            }

                cohere_tools.append(cohere_tool)

            if cohere_tools:
                request_data["tools"] = cohere_tools
                request_data["tool_results"] = []  # Initialize empty tool results

        # Make request to Cohere API
        try:
            response = self.session.post(
                f"{self.api_base}/chat", json=request_data, timeout=self.timeout
            )
            response.raise_for_status()
            response_data = response.json()

            # Extract the response text
            result = response_data.get("text", "")

            # Extract usage information
            tokens_used = response_data.get("meta", {}).get("billed_units", {})
            read_tokens = tokens_used.get("input_tokens", 0)
            write_tokens = tokens_used.get("output_tokens", 0)
            total_tokens = read_tokens + write_tokens

            # Calculate costs
            model_costs = self.get_model_costs()

            # Use safe access with default values to handle possible None values
            read_token_cost = float(model_costs.get("read_token", 0.0) or 0.0)
            write_token_cost = float(model_costs.get("write_token", 0.0) or 0.0)

            # Calculate costs with proper type conversion
            read_cost = float(read_tokens) * read_token_cost
            write_cost = float(write_tokens) * write_token_cost

            # Calculate image cost
            image_cost = 0.0
            if image_count > 0 and "image_token" in model_costs:
                image_token_cost = float(model_costs.get("image_token", 0.0) or 0.0)
                image_cost = float(image_count) * image_token_cost

            total_cost = read_cost + write_cost + image_cost

            # Process tool calls if present
            tool_use_info = {}
            if "tool_calls" in response_data:
                tool_calls = response_data.get("tool_calls", [])
                if tool_calls:
                    tool_call = tool_calls[0]  # Use the first tool call
                    tool_use_info = {
                        "id": tool_call.get("id", ""),
                        "name": tool_call.get("name", ""),
                        "arguments": tool_call.get("parameters", {}),
                    }

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
                "tool_use": tool_use_info,
            }

            return result, usage

        except requests.exceptions.RequestException as e:
            # Handle HTTP errors
            error_message = f"Cohere API request failed: {e!s}"
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_message = (
                        f"Cohere API error: {error_data.get('message', str(e))}"
                    )
                except:
                    error_message = f"Cohere API error: {e.response.text}"

            raise LLMMistake(
                error_message,
                error_type="api_error",
                provider="cohere",
                details={
                    "status_code": (
                        e.response.status_code
                        if hasattr(e, "response") and e.response is not None
                        else None
                    )
                },
            )
        except Exception as e:
            # Handle other errors
            raise LLMMistake(
                f"Error generating text with Cohere: {e!s}",
                error_type="general_error",
                provider="cohere",
            )

    def supports_image_input(self) -> bool:
        """Check if the current model supports image inputs."""
        # Cohere Command models support image inputs
        return self.model_name in ["command-r", "command-r-plus"]

    def _process_image_file(self, image_path: str) -> str:
        """
        Process and encode an image file for the Cohere API.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string
        """
        try:
            with Image.open(image_path) as img:
                # Convert to JPEG if not already
                if img.format != "JPEG":
                    # Create a BytesIO object to save the JPEG
                    buffer = BytesIO()
                    # Convert to RGB if it has an alpha channel
                    if img.mode == "RGBA":
                        img = img.convert("RGB")
                    # Save as JPEG
                    img.save(buffer, format="JPEG", quality=90)
                    buffer.seek(0)
                    image_data = buffer.read()
                else:
                    # If already JPEG, just read the file
                    with open(image_path, "rb") as f:
                        image_data = f.read()

                # Encode to base64
                return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            print(f"Error processing image file {image_path}: {e!s}")
            # Return an empty string instead of None
            return ""

    def _process_image_url(self, image_url: str) -> str:
        """
        Download and encode an image from URL for the Cohere API.

        Args:
            image_url: URL of the image

        Returns:
            Base64 encoded image string or empty string if using direct URL
        """
        # For Cohere, we can use the URL directly in the attachments
        # We don't need to download and encode the image
        return ""

    def supports_embeddings(self) -> bool:
        """
        Check if this provider supports embeddings.

        Returns:
            bool: True if embeddings are supported, False otherwise
        """
        return True  # Cohere supports embeddings through separate models

    def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 25,
        **kwargs,
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Get embeddings for a list of texts using Cohere's embedding models.

        Args:
            texts (List[str]): List of texts to embed
            model (Optional[str]): Specific embedding model to use
            batch_size (int): Number of texts to embed in each batch
            **kwargs: Additional arguments to pass to the embedding API

        Returns:
            Tuple[List[List[float]], Dict[str, Any]]:
                - List of embedding vectors (one per input text)
                - Usage data including:
                  - total_tokens: Total tokens used
                  - total_cost: Cost incurred
                  - model: Model used

        Raises:
            ImportError: If cohere package is not installed
            ValueError: If embedding fails or model is invalid
        """
        if not texts:
            return [], {"total_tokens": 0, "total_cost": 0.0, "model": None}

        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Cohere package not installed. Install with 'pip install cohere'"
            )

        # Use specified model or default
        embedding_model = model or self.DEFAULT_EMBEDDING_MODEL

        # Validate model
        if embedding_model not in self.EMBEDDING_MODELS:
            raise ValueError(
                f"Embedding model {embedding_model} not supported. Supported models: {self.EMBEDDING_MODELS}"
            )

        # Initialize Cohere client
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError(
                "Cohere API key not found. Please provide it when initializing the CohereLLM."
            )

        client = cohere.Client(api_key)

        try:
            all_embeddings = []
            total_tokens = 0

            # Process in batches to avoid rate limits and large requests
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                response = client.embed(
                    texts=batch,
                    model=embedding_model,
                    input_type=kwargs.get("input_type", "search_document"),
                )

                # Add embeddings to result
                all_embeddings.extend(response.embeddings)

                # Estimate tokens (Cohere doesn't return token count directly)
                batch_tokens = sum(
                    len(text.split()) * 1.3 for text in batch
                )  # Rough estimate
                total_tokens += int(batch_tokens)

            # Calculate cost
            cost_per_token = self.embedding_costs.get(
                embedding_model, 0.00001
            )  # Default if unknown
            total_cost = total_tokens * cost_per_token

            # Return embeddings and usage data
            usage = {
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "model": embedding_model,
            }

            return all_embeddings, usage

        except Exception as e:
            raise ValueError(f"Error getting embeddings from Cohere: {e!s}")

    def supports_reranking(self) -> bool:
        """
        Check if this provider and model supports document reranking.

        Returns:
            bool: True if reranking is supported, False otherwise
        """
        return True  # Cohere has dedicated reranking models

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_scores: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Rerank documents using Cohere's dedicated reranking API.

        Args:
            query (str): The search query to rank documents against
            documents (List[str]): List of document texts to rerank
            top_n (Optional[int]): Number of top documents to return, None for all
            return_scores (bool): Whether to include relevance scores in the output
            **kwargs: Additional provider-specific parameters

        Returns:
            Dict[str, Any]: Dictionary containing ranked documents and metadata
        """
        if not documents:
            return {
                "ranked_documents": [],
                "indices": [],
                "scores": [] if return_scores else None,
                "usage": {"total_tokens": 0, "total_cost": 0.0},
            }

        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Cohere package not installed. Install with 'pip install cohere'"
            )

        # Use specified model or default
        rerank_model = kwargs.get("model", self.DEFAULT_RERANKING_MODEL)

        # Validate model
        if rerank_model not in self.RERANKING_MODELS:
            raise ValueError(
                f"Reranking model {rerank_model} not supported. Supported models: {self.RERANKING_MODELS}"
            )

        # Initialize Cohere client
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError(
                "Cohere API key not found. Please provide it when initializing the CohereLLM."
            )

        client = cohere.Client(api_key)

        try:
            # Call Cohere's reranking endpoint
            response = client.rerank(
                query=query,
                documents=documents,
                model=rerank_model,
                top_n=top_n or len(documents),
                return_documents=(top_n is not None),
            )

            # Process results
            ranked_indices = [result.index for result in response.results]

            # Get ranked documents - either from response or reindex original documents
            if top_n is not None and hasattr(response.results[0], "document"):
                ranked_documents = [result.document for result in response.results]
            else:
                ranked_documents = [documents[idx] for idx in ranked_indices]

            # Get scores if requested
            scores = (
                [result.relevance_score for result in response.results]
                if return_scores
                else None
            )

            # Estimate token usage (since Cohere doesn't provide this directly)
            query_tokens = len(query.split()) * 1.3  # Rough estimate
            doc_tokens = sum(
                len(doc.split()) * 1.3 for doc in documents
            )  # Rough estimate
            total_tokens = int(query_tokens + doc_tokens)

            # Calculate cost
            cost_per_token = self.reranking_costs.get(
                rerank_model, 0.00001
            )  # Default if unknown
            total_cost = total_tokens * cost_per_token

            return {
                "ranked_documents": ranked_documents,
                "indices": ranked_indices,
                "scores": scores,
                "usage": {
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                    "model": rerank_model,
                },
            }

        except Exception as e:
            raise ValueError(f"Error reranking documents with Cohere: {e!s}")


# Register the provider with explicit type casting to avoid mypy error
register_provider("cohere", cast(Type[LLM], CohereLLM))
