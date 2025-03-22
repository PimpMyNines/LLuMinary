# LLuMinary

A comprehensive and type-safe Python library providing a unified interface to multiple LLM providers.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/PimpMyNines/LLuMinary/actions/workflows/ci.yml/badge.svg)](https://github.com/PimpMyNines/LLuMinary/actions/workflows/ci.yml)
[![Matrix Tests](https://github.com/PimpMyNines/LLuMinary/actions/workflows/matrix-docker-tests.yml/badge.svg)](https://github.com/PimpMyNines/LLuMinary/actions/workflows/matrix-docker-tests.yml)
[![Codecov](https://codecov.io/gh/PimpMyNines/LLuMinary/branch/main/graph/badge.svg)](https://codecov.io/gh/PimpMyNines/LLuMinary)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://pimpmy9s.github.io/LLuMinary/)

## Project Status: Version 1.0.0

This is the initial release of LLuMinary, a production-ready abstraction layer for Large Language Models. While feature-complete for the providers listed below, there are enhancements planned for future releases.

### What's Included in This Release
- Complete implementations of OpenAI, Anthropic, Google AI, Cohere, and AWS Bedrock providers
- Type-safe interfaces with comprehensive error handling
- Extensive test coverage (>85%) with both unit and integration tests
- Docker-based testing and CI/CD pipeline
- Thorough documentation and examples for all features
- Robust error handling with standardized exceptions and proper context
- Comprehensive type annotations and TypedDict definitions

### Coming in Future Releases
See our [roadmap](ROADMAP.md) for a detailed view of upcoming features and improvements.

## Credits

This project is based on the original llm-handler created by [Chase Brown (@chaseabrown)](https://github.com/chaseabrown). The original work provided the foundation and inspiration for this enhanced version. Chase is a key contributor to this project.

## Overview

LLuMinary provides a robust, extensible interface for working with various LLM providers including OpenAI, Anthropic, Google, Cohere, and AWS Bedrock. It handles all the complexity of provider-specific implementations, message formatting, and error handling, allowing you to focus on building applications.

## Features

- 🤝 **Unified API** across multiple providers (OpenAI, Anthropic, Google, Cohere)
- 🔄 **Provider Registry Pattern** for easy extension with custom providers
- 📝 **Text Generation** with consistent interface and error handling
- 🌊 **Streaming Responses** from all major providers
- 🔍 **Embeddings** generation for semantic search and similarity matching
- 📚 **Document Reranking** for improved search relevance
- 🔧 **Function Calling** for tool use and external API integration
- 📊 **Token Counting & Cost Tracking** for budget management
- 🖼️ **Image Input Support** for multimodal models
- 🏷️ **Classification** for categorizing text inputs
- 🧪 **Enhanced Type Validation** for complex nested types and robust error handling
- 🖥️ **CLI Tools** for classification management and testing

## Documentation

- [API Reference](./docs/API_REFERENCE.md) - Detailed documentation of all classes and methods
- [Architecture](./docs/ARCHITECTURE.md) - Overview of the library's design and component relationships
- [Tutorials](./docs/TUTORIALS.md) - Step-by-step guides for common use cases
- [Examples](./examples/) - Working code examples for all features
- [Updated Components](./docs/UPDATED_COMPONENTS.md) - Information about recently enhanced components
- [Test Coverage](./docs/TEST_COVERAGE.md) - Test coverage reports and implementation guides

## Installation

```bash
pip install lluminary
```

## Quick Start

```python
from lluminary import get_llm_from_model

# Initialize an LLM (automatically selects the right provider)
llm = get_llm_from_model("gpt-4o", api_key="your-api-key")

# Generate a simple response
response, usage, _ = llm.generate(
    event_id="quick-start",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "What is machine learning?"}
    ]
)

print(response)
print(f"Tokens used: {usage['total_tokens']}")
print(f"Cost: ${usage['total_cost']}")
```

## Supported Providers and Models

### OpenAI

- GPT-4o, GPT-4, GPT-3.5-Turbo
- Text embedding models
- Image input support

```python
llm = get_llm_from_model("gpt-4o", api_key="your-openai-api-key")
```

### Anthropic

- Claude 3 (Opus, Sonnet, Haiku)
- Claude 2
- Streaming support

```python
llm = get_llm_from_model("claude-haiku-3.5", api_key="your-anthropic-api-key")
```

### Google

- Gemini models (Pro, Flash)
- Streaming support
- Image input support

```python
llm = get_llm_from_model("gemini-2.0-flash", api_key="your-google-api-key")
```

### Cohere

- Command models
- Embedding models
- Reranking models

```python
llm = get_llm_from_model("cohere-command", api_key="your-cohere-api-key")
```

## Basic Features

### Text Generation

Generate text responses from LLMs:

```python
response, usage, _ = llm.generate(
    event_id="text-generation",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "Explain the concept of machine learning."}
    ],
    max_tokens=500,
    temp=0.7
)

print(response)
```

### Embeddings

Generate embeddings for text with supported models:

```python
from lluminary import get_llm_from_model

# Initialize an LLM that supports embeddings
llm = get_llm_from_model("text-embedding-3-small")

# Check if the model supports embeddings
if llm.supports_embeddings():
    # Generate embeddings
    texts = ["This is a sample text", "Another example"]
    embeddings, usage = llm.embed(texts=texts)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding dimensions: {len(embeddings[0])}")
    print(f"Tokens used: {usage['total_tokens']}")
    print(f"Cost: ${usage['total_cost']}")
```

### Document Reranking

Rerank documents based on their relevance to a query:

```python
from lluminary import get_llm_from_model

# Initialize an LLM that supports reranking
llm = get_llm_from_model("text-embedding-3-small")  # OpenAI
# or
llm = get_llm_from_model("rerank-english-v3.0")  # Cohere

# Check if the model supports reranking
if llm.supports_reranking():
    # Sample documents
    documents = [
        "Python is a popular programming language.",
        "The Eiffel Tower is in Paris, France.",
        "Machine learning is a subset of AI.",
        "Cats are common pets worldwide."
    ]

    # Rerank documents based on a query
    query = "Which programming language is popular?"
    results = llm.rerank(
        query=query,
        documents=documents,
        top_n=2,  # Optional: limit to top N results
        return_scores=True  # Optional: include relevance scores
    )

    # Display ranked results
    for i, (doc, score) in enumerate(zip(
        results["ranked_documents"],
        results["scores"]
    )):
        print(f"{i+1}. [{score:.4f}] {doc}")

    # Usage information
    print(f"Tokens used: {results['usage']['total_tokens']}")
    print(f"Cost: ${results['usage']['total_cost']}")
```

### Streaming Responses

Stream responses from LLMs for real-time output:

```python
from lluminary import get_llm_from_model

# Initialize an LLM that supports streaming (OpenAI, Anthropic, or Google)
llm = get_llm_from_model("gpt-4o")  # OpenAI
# or
llm = get_llm_from_model("claude-haiku-3.5")  # Anthropic
# or
llm = get_llm_from_model("gemini-2.0-flash")  # Google

# Define a callback function to process chunks
def process_chunk(chunk, usage_data):
    if chunk:  # Empty chunk signals completion
        print(chunk, end="", flush=True)
    else:
        print("\nStream completed")
        print(f"Total tokens: {usage_data['total_tokens']}")
        print(f"Cost: ${usage_data.get('total_cost', 0)}")

# Stream a response
for chunk, usage in llm.stream_generate(
    event_id="my_stream",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "Explain quantum computing briefly."}
    ],
    max_tokens=300,
    temp=0.7,
    callback=process_chunk  # Optional: Process chunks as they arrive
):
    # You can also process chunks here if you prefer
    # This loop will yield each chunk as it's received
    pass  # The callback handles the chunks, so nothing to do here
```

### Image Processing

Work with multimodal models that support image inputs:

```python
from lluminary import get_llm_from_model

# Initialize a multimodal model
llm = get_llm_from_model("gpt-4o")  # OpenAI
# or
llm = get_llm_from_model("claude-haiku-3.5")  # Anthropic
# or
llm = get_llm_from_model("gemini-2.0-pro")  # Google

# Generate a response that includes image analysis
response, usage, _ = llm.generate(
    event_id="image_analysis",
    system_prompt="You are a helpful assistant that can analyze images.",
    messages=[
        {
            "message_type": "human",
            "message": "What's in this image?",
            "image_paths": ["path/to/local/image.jpg"],  # Local file path
            "image_urls": []  # Can also use URLs
        }
    ],
    max_tokens=300
)

print(response)
print(f"Tokens used: {usage['total_tokens']}")
print(f"Images processed: {usage['images']}")
print(f"Cost: ${usage['total_cost']}")
```

### Classification

Categorize text inputs using LLMs:

```python
from lluminary import get_llm_from_model

# Initialize any LLM
llm = get_llm_from_model("gpt-3.5-turbo")

# Define categories
categories = {
    "question": "A query seeking information",
    "command": "A directive to perform an action",
    "statement": "A declarative sentence"
}

# Optional: Provide examples for better accuracy
examples = [
    {
        "user_input": "What is the weather like?",
        "doc_str": "This is a question seeking information about weather",
        "selection": "question"
    },
    {
        "user_input": "Turn on the lights",
        "doc_str": "This is a command to perform an action",
        "selection": "command"
    }
]

# Messages to classify
messages = [
    {
        "message_type": "human",
        "message": "Tell me how to bake a cake."
    }
]

# Perform classification
selection, usage = llm.classify(
    messages=messages,
    categories=categories,
    examples=examples
)

print(f"Classification: {selection}")
print(f"Tokens used: {usage['total_tokens']}")
print(f"Cost: ${usage['total_cost']}")
```

### Token Counting & Cost Tracking

Monitor token usage and costs for budget management:

```python
from lluminary import get_llm_from_model

# Initialize any LLM
llm = get_llm_from_model("gpt-4o")

# Generate a response
response, usage, _ = llm.generate(
    event_id="token_tracking",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "Write a short poem about AI."}
    ],
    max_tokens=100
)

# Access detailed token usage information
print(f"Input tokens: {usage['read_tokens']}")
print(f"Output tokens: {usage['write_tokens']}")
print(f"Total tokens: {usage['total_tokens']}")

# Access cost information
print(f"Input cost: ${usage['read_cost']:.6f}")
print(f"Output cost: ${usage['write_cost']:.6f}")
print(f"Total cost: ${usage['total_cost']:.6f}")

# With image inputs (if applicable)
print(f"Images processed: {usage.get('images', 0)}")
print(f"Image cost: ${usage.get('image_cost', 0):.6f}")
```

## Advanced Features

### Function Calling

```python
def get_weather(location: str, unit: str = "celsius"):
    """Get the current weather for a location."""
    # Implementation details...
    return {"location": location, "temperature": 22, "condition": "sunny"}

response, usage, _ = llm.generate(
    event_id="function-calling",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "What's the weather in San Francisco?"}
    ],
    functions=[get_weather]
)
```

### Enhanced Type Validation

Robust type validation for complex nested types:

```python
from typing import Dict, List, Union, Any
from lluminary.tools.validators import validate_return_type, validate_tool

@validate_tool
@validate_return_type(Dict[str, List[Dict[str, Union[str, int]]]])
def get_users() -> Dict[str, List[Dict[str, Union[str, int]]]]:
    """Get a list of users with their details."""
    return {
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
    }
```

### Classification CLI

Manage and test classification configurations via command-line:

```bash
# List available classification configurations
python -m lluminary.cli.classify list_configs /path/to/config/dir

# Validate a classification configuration
python -m lluminary.cli.classify validate /path/to/config.json

# Test a classification
python -m lluminary.cli.classify test /path/to/config.json "Test message" --model claude-haiku-3.5

# Create a new classification configuration interactively
python -m lluminary.cli.classify create /path/to/output.json
```

### Custom Provider Registration

```python
from lluminary.models.base import LLM
from lluminary.models.router import register_provider

class CustomLLM(LLM):
    """Custom LLM provider implementation."""
    # Implementation details...

# Register the custom provider
register_provider("custom", CustomLLM)

# Now you can use your custom provider
llm = get_llm_from_model("custom-model", api_key="your-custom-api-key")
```

## Interactive Testing Notebook

You can run the following examples in a Jupyter notebook to test the LLM Handler capabilities directly. Simply copy and paste the cells into a notebook and execute them sequentially.

### Setup

```python
# Install the library (if needed)
!pip install lluminary

# Import necessary components
from lluminary import get_llm_from_model
import os
import time

# Set up your API keys
# Either set them in the environment:
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"
# os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
# os.environ["COHERE_API_KEY"] = "your-cohere-api-key"

# Or provide them directly when initializing models:
openai_api_key = "your-openai-api-key"  # Replace with your actual key
anthropic_api_key = "your-anthropic-api-key"  # Replace with your actual key
```

### Basic Generation Test

```python
# Initialize an OpenAI model
llm = get_llm_from_model("gpt-3.5-turbo", api_key=openai_api_key)

# Generate a response
response, usage, _ = llm.generate(
    event_id="notebook_test",
    system_prompt="You are a helpful assistant.",
    messages=[
        {"message_type": "human", "message": "Hello! Tell me a quick joke."}
    ],
    max_tokens=100
)

# Print the results
print("Response:", response)
print(f"Tokens: {usage['read_tokens']} in, {usage['write_tokens']} out, {usage['total_tokens']} total")
print(f"Cost: ${usage['total_cost']:.6f}")
```

### Streaming Test

```python
# Initialize a streaming-capable model
llm = get_llm_from_model("gpt-3.5-turbo", api_key=openai_api_key)

# Define a callback for processing chunks
def process_chunk(chunk, usage_data):
    if chunk:
        print(chunk, end="", flush=True)
    else:
        print("\n\n--- Stream completed ---")
        print(f"Total tokens: {usage_data.get('total_tokens', 'unknown')}")
        print(f"Cost: ${usage_data.get('total_cost', 0):.6f}")

# Stream a response
print("Streaming response:\n")
start_time = time.time()

for _, _ in llm.stream_generate(
    event_id="notebook_stream_test",
    system_prompt="You are a helpful assistant who gives concise responses.",
    messages=[
        {"message_type": "human", "message": "Explain how transformers work in machine learning."}
    ],
    max_tokens=150,
    temp=0.7,
    callback=process_chunk
):
    pass  # The callback handles output

print(f"\nTotal time: {time.time() - start_time:.2f} seconds")
```

### Embedding Test

```python
# Initialize an embedding model
llm = get_llm_from_model("text-embedding-3-small", api_key=openai_api_key)

# Check if the model supports embeddings
if llm.supports_embeddings():
    # Sample texts
    texts = [
        "Machine learning is fascinating",
        "Natural language processing has advanced significantly",
        "Neural networks can solve complex problems"
    ]

    # Generate embeddings
    embeddings, usage = llm.embed(texts=texts)

    # Show results
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    print(f"First embedding values (first 5): {embeddings[0][:5]}")
    print(f"Tokens used: {usage['total_tokens']}")
    print(f"Cost: ${usage['total_cost']:.6f}")
else:
    print("This model doesn't support embeddings")
```

### Classification Test

```python
# Initialize any LLM
llm = get_llm_from_model("gpt-3.5-turbo", api_key=openai_api_key)

# Define categories
categories = {
    "technical": "Technical or scientific content about computers, programming, or other technologies",
    "creative": "Creative writing like stories, poems, or artistic content",
    "business": "Business, finance, or professional workplace content"
}

# Messages to classify
test_messages = [
    {
        "message_type": "human",
        "message": "How do I optimize a database query for better performance?"
    }
]

# Perform classification
classification, usage = llm.classify(
    messages=test_messages,
    categories=categories
)

print(f"Classification result: {classification}")
print(f"Tokens used: {usage['total_tokens']}")
print(f"Cost: ${usage['total_cost']:.6f}")
```

### Document Reranking Test

```python
# Initialize a model that supports reranking
llm = get_llm_from_model("text-embedding-3-small", api_key=openai_api_key)

# Check if the model supports reranking
if llm.supports_reranking():
    # Sample documents
    documents = [
        "Python is widely used for data science and machine learning.",
        "JavaScript is the primary language for web development.",
        "SQL is essential for database management and querying.",
        "Java is popular for enterprise applications and Android development.",
        "Python's simplicity makes it great for beginners to learn programming."
    ]

    # Rerank based on relevance to the query
    query = "What programming language should I learn first?"
    results = llm.rerank(
        query=query,
        documents=documents,
        top_n=3,
        return_scores=True
    )

    # Show ranked results
    print(f"Query: {query}\n")
    print("Ranked documents:")
    for i, (doc, score) in enumerate(zip(
        results["ranked_documents"],
        results["scores"]
    )):
        print(f"{i+1}. [{score:.4f}] {doc}")

    # Show usage information
    print(f"\nTokens used: {results['usage']['total_tokens']}")
    print(f"Cost: ${results['usage']['total_cost']:.6f}")
else:
    print("This model doesn't support reranking")
```

### Troubleshooting

If you encounter issues running the examples:

1. **API Key Issues**: Ensure your API keys are valid and have the necessary permissions
2. **Rate Limiting**: Some providers have rate limits; add delays between calls if needed
3. **Model Availability**: Check that the requested model is available in your region/account
4. **Billing**: Ensure your account has sufficient credits or billing set up
5. **Package Installation**: Make sure all dependencies are installed

To check your environment setup:

```python
# Diagnostic information
import sys
print(f"Python version: {sys.version}")

# Check package version
try:
    import lluminary
    print(f"LLuMinary version: {lluminary.__version__}")
except (ImportError, AttributeError):
    print("LLuMinary package not found or version not available")

# Check API key configuration (safely)
def mask_key(key):
    """Show only first and last 4 characters of a key."""
    if key and len(key) > 8:
        return key[:4] + '*' * (len(key) - 8) + key[-4:]
    return "Not set"

print(f"OpenAI API key: {mask_key(os.environ.get('OPENAI_API_KEY', ''))}")
print(f"Anthropic API key: {mask_key(os.environ.get('ANTHROPIC_API_KEY', ''))}")
print(f"Google API key: {mask_key(os.environ.get('GOOGLE_API_KEY', ''))}")
print(f"Cohere API key: {mask_key(os.environ.get('COHERE_API_KEY', ''))}")
```

## Contributing

Contributions are welcome! Please check out our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# LLuMinary GitHub Issue Management Tools

This repository contains tools to help manage GitHub issues for the LLuMinary project, specifically focusing on converting tasks in parent issues into proper sub-issues.

## Overview

The GitHub sub-issues feature allows for better organization and tracking of complex issues by breaking them down into smaller, manageable pieces. These tools help automate the process of:

1. Identifying issues with tasks that need to be converted to sub-issues
2. Creating sub-issues for each task
3. Linking sub-issues to parent issues
4. Updating parent issues to reflect the changes

## Prerequisites

- GitHub Personal Access Token with `repo` scope
- `jq` command-line tool installed
- `bc` command-line calculator installed

## Setup

1. Set your GitHub token as an environment variable:

```bash
export GITHUB_TOKEN='your_github_token'
```

2. Make sure all scripts are executable:

```bash
chmod +x *.sh
```

## Available Scripts

### 1. Test GitHub API Access

Before using the main scripts, verify that your GitHub token has the necessary permissions:

```bash
./test_github_api.sh
```

This script will:
- Test basic API access
- Test access to the sub-issues API
- Verify write permissions by creating and deleting a test label

### 2. Find Issues with Tasks

To identify which issues have tasks that need to be converted to sub-issues:

```bash
./find_issues_with_tasks.sh
```

This script will output a list of issue numbers, titles, and the number of tasks in each issue.

### 3. Create Sub-Issues

Once you've identified issues with tasks, use this script to create sub-issues:

```bash
./create_sub_issues.sh <issue_number> [issue_number2 ...]
```

Example:

```bash
./create_sub_issues.sh 86 85 62
```

This script will:
1. Get the issue details from GitHub
2. Extract tasks in the format `- [ ] Task description`
3. Create a new issue for each task
4. Add each new issue as a sub-issue to the parent
5. Update the parent issue description

## Story Points Distribution

The script distributes story points from the parent issue evenly among the sub-issues. For example, if a parent issue has 10 story points and 5 tasks, each sub-issue will get 2 story points.

## Workflow

The recommended workflow is:

1. Run `./test_github_api.sh` to verify API access
2. Run `./find_issues_with_tasks.sh` to identify issues with tasks
3. Run `./create_sub_issues.sh` with the issue numbers you want to process

## Troubleshooting

- If you get authentication errors, check that your GitHub token is set correctly and has the necessary permissions.
- If tasks aren't being extracted, ensure they follow the format `- [ ] Task description` in the issue body.
- If you encounter JSON parsing errors, check that the issue body doesn't contain special characters that might break the JSON.

## Notes

- These scripts require the GitHub API's sub-issues feature, which is currently in public preview for organizations.
- The scripts assume the repository is "PimpMyNines/LLuMinary". If you need to use them for a different repository, edit the `REPO` variable in each script.
