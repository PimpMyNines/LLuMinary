#!/usr/bin/env python
"""
Backend server for the LLuMinary Chat Demo app.

This Flask application provides API endpoints for the React frontend to interact with
LLuMinary, handling model selection, authentication, and generation requests.
"""
import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Dict

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

# Add parent directory to python path so we can import LLuMinary
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lluminary import get_llm_from_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store API keys in memory (in a real app, use a secure storage)
api_keys: Dict[str, str] = {}


def validate_provider_key(provider: str, api_key: str) -> bool:
    """Validate that an API key works for a given provider."""
    try:
        # Try to initialize a model with the given key
        if provider == "bedrock":
            # For AWS Bedrock, we store the profile name
            os.environ["AWS_PROFILE"] = api_key
            llm = get_llm_from_model("anthropic.claude-3-haiku-20240307-v1:0")
        elif provider == "openai":
            llm = get_llm_from_model("gpt-4o-mini", api_key=api_key)
        elif provider == "anthropic":
            llm = get_llm_from_model("claude-3-haiku-20240307", api_key=api_key)
        elif provider == "google":
            llm = get_llm_from_model("gemini-1.5-flash", api_key=api_key)
        elif provider == "cohere":
            llm = get_llm_from_model("command-r", api_key=api_key)
        else:
            return False

        # Try to authenticate (this will throw if the key is invalid)
        llm.auth()
        return True
    except Exception as e:
        logger.error(f"Error validating {provider} API key: {e}")
        return False


@app.route("/api/validate-key", methods=["POST"])
def validate_key():
    """Validate an API key for a specific provider."""
    data = request.json
    provider = data.get("provider")
    api_key = data.get("api_key")

    if not provider or not api_key:
        return jsonify({"error": "Missing provider or API key"}), 400

    # Validate the key
    is_valid = validate_provider_key(provider, api_key)

    if is_valid:
        # Store the key for future use
        api_keys[provider] = api_key
        return jsonify({"valid": True})
    else:
        return jsonify({"valid": False, "error": "Invalid API key"}), 401


@app.route("/api/providers", methods=["GET"])
def get_providers():
    """Get a list of available providers based on configured API keys."""
    providers = []

    for provider, api_key in api_keys.items():
        if provider == "openai":
            model_names = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
        elif provider == "anthropic":
            model_names = [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ]
        elif provider == "google":
            model_names = ["gemini-1.5-pro", "gemini-1.5-flash"]
        elif provider == "cohere":
            model_names = ["command-r-plus", "command-r"]
        elif provider == "bedrock":
            model_names = [
                "anthropic.claude-3-opus-20240229-v1:0",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
            ]
        else:
            model_names = []

        # Add provider with models
        providers.append(
            {
                "id": provider,
                "name": provider.capitalize(),
                "models": [{"id": model, "name": model} for model in model_names],
            }
        )

    return jsonify(providers)


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate a completion using LLuMinary."""
    data = request.json
    model_id = data.get("model_id")
    system_prompt = data.get("system_prompt", "You are a helpful assistant.")
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 1000)
    provider = data.get("provider")
    stream = data.get("stream", False)

    if not model_id:
        return jsonify({"error": "Missing model_id parameter"}), 400

    if not provider:
        # Try to determine provider from model_id
        if model_id.startswith("gpt-"):
            provider = "openai"
        elif model_id.startswith("claude-"):
            provider = "anthropic"
        elif model_id.startswith("gemini-"):
            provider = "google"
        elif model_id.startswith("command-"):
            provider = "cohere"
        elif "anthropic." in model_id or "amazon." in model_id:
            provider = "bedrock"
        else:
            return jsonify({"error": "Could not determine provider from model_id"}), 400

    # Get API key for this provider
    api_key = api_keys.get(provider)
    if not api_key and provider != "bedrock":
        return jsonify({"error": f"No API key configured for {provider}"}), 401

    try:
        # Initialize the model
        kwargs = {}
        if provider != "bedrock":
            kwargs["api_key"] = api_key
        elif provider == "bedrock":
            os.environ["AWS_PROFILE"] = api_key

        llm = get_llm_from_model(model_id, **kwargs)

        # Format messages for LLuMinary
        formatted_messages = [
            {
                "message_type": (
                    "human"
                    if m["role"] == "user"
                    else "ai" if m["role"] == "assistant" else m["role"]
                ),
                "message": m["content"],
                "image_paths": [],
                "image_urls": m.get("images", []),
            }
            for m in messages
        ]

        # Generate completion
        event_id = str(uuid.uuid4())

        if stream:
            return Response(
                stream_with_context(
                    stream_generate(
                        llm,
                        event_id,
                        system_prompt,
                        formatted_messages,
                        max_tokens,
                        temperature,
                    )
                ),
                content_type="text/event-stream",
            )
        else:
            # For non-streaming responses
            response, usage, updated_messages = llm.generate(
                event_id=event_id,
                system_prompt=system_prompt,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temp=temperature,
            )

            return jsonify(
                {
                    "text": response,
                    "usage": {
                        "read_tokens": usage.get("read_tokens", 0),
                        "write_tokens": usage.get("write_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "total_cost": usage.get("total_cost", 0),
                    },
                }
            )

    except Exception as e:
        logger.exception(f"Error generating completion: {e}")
        return jsonify({"error": str(e)}), 500


def stream_generate(llm, event_id, system_prompt, messages, max_tokens, temperature):
    """Stream the output using LLuMinary's streaming interface."""
    try:
        full_response = ""
        for chunk, usage in llm.stream_generate(
            event_id=event_id,
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temp=temperature,
        ):
            full_response += chunk
            # Format as server-sent event
            data = json.dumps(
                {
                    "chunk": chunk,
                    "usage": {
                        "read_tokens": usage.get("read_tokens", 0),
                        "write_tokens": usage.get("write_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "total_cost": usage.get("total_cost", 0),
                    },
                    "done": False,
                }
            )
            yield f"data: {data}\n\n"

        # Send a final event with the complete response
        final_data = json.dumps(
            {
                "text": full_response,
                "usage": {
                    "read_tokens": usage.get("read_tokens", 0),
                    "write_tokens": usage.get("write_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "total_cost": usage.get("total_cost", 0),
                },
                "done": True,
            }
        )
        yield f"data: {final_data}\n\n"
    except Exception as e:
        logger.exception(f"Error in streaming: {e}")
        error_data = json.dumps({"error": str(e), "done": True})
        yield f"data: {error_data}\n\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLuMinary Chat Demo API Server")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
