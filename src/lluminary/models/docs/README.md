# LLuMinary Models

This directory contains the implementation of various Large Language Model (LLM) providers. The system is designed with a flexible, provider-agnostic architecture that allows for consistent handling of text and image inputs across different LLM services.

## Architecture

### Base Class: `LLM`

The `LLM` abstract base class (`base.py`) defines the common interface that all providers must implement. Key features include:

- Token estimation and context window management
- Cost calculation and usage tracking
- Standard message format handling
- Image input support detection
- Provider authentication
- Classification capabilities
- Synchronous operation (no async/await)
- Automatic error correction and retries

### Design Principles

1. **Synchronous Operation**
   - All operations are synchronous by design
   - No async/await functionality
   - Direct, blocking API calls

2. **Error Handling**
   - Automatic retry mechanism with configurable limits
   - Error feedback loop in retry process
   - Detailed error messages for debugging

3. **Message Processing**
   - Deep copy of messages to prevent modification of originals
   - Standardized format across all providers
   - Automatic format conversion for each provider

4. **Cost Management**
   - Pre-calculation of estimated costs
   - Detailed usage tracking
   - Per-provider cost configurations

### Provider Implementations

Each provider has its own implementation class that inherits from `LLM`:

1. **OpenAI** (`openai.py`)
   - Supports GPT-4 Vision models
   - Uses base64-encoded JPEG images with data URI prefix
   - Dynamic image cost calculation based on size/detail
   - Supports both image paths and URLs

2. **Anthropic** (`anthropic.py`)
   - Supports Claude 3 models
   - Uses base64-encoded JPEG images with media type
   - Fixed per-image cost
   - Handles alpha channels by converting to RGB

3. **Bedrock** (`bedrock.py`)
   - Supports AWS Bedrock Claude models
   - Uses raw PNG bytes (no base64 encoding)
   - Preserves alpha channels for PNG format
   - AWS authentication via boto3

4. **Google** (`google.py`)
   - Supports Gemini models
   - Handles image formats according to Google API requirements
   - Supports embeddings and function calling

5. **Cohere** (`cohere.py`)
   - Supports Cohere models
   - Specialized for reranking and embeddings

### Router

The `router.py` module provides a unified interface for model selection and initialization:

- Maps friendly model names to provider implementations
- Handles model validation and instantiation
- Provides model listing and availability checking

## Message Format

The system uses a standardized message format across all providers:

```python
{
    "message_type": "human" | "ai",
    "message": str,
    "image_paths": List[str],  # Local image file paths
    "image_urls": List[str]    # Remote image URLs
}
```

## Error Correction System

The system includes automatic error correction with the following features:

1. **Retry Mechanism**
   ```python
   llm.generate(
       event_id="unique_id",
       system_prompt="prompt",
       messages=messages,
       result_processing_function=process_func,  # Optional
       retry_limit=3  # Default
   )
   ```

2. **Processing Functions**
   - Custom validation and transformation
   - XML-based response parsing
   - Automatic error detection

3. **Error Feedback**
   - Failed responses are added to conversation
   - Error messages guide the model
   - Cumulative usage tracking across retries

## Classification Feature

The system includes a powerful classification capability that works across all providers:

### Basic Usage
```python
from lluminary.models.router import get_llm_from_model

# Initialize a model
llm = get_llm_from_model("claude-sonnet-3.5")

# Define categories
categories = {
    "technical": "Questions about programming or software",
    "support": "Requests for help or assistance",
    "feedback": "Comments about product experience"
}

# Optional examples for better accuracy
examples = [{
    "user_input": "Python is giving me a TypeError",
    "doc_str": "This is a technical issue with code",
    "selection": "technical"
}]

# Classify a message
selections, usage = llm.classify(
    messages=[{
        "message_type": "human",
        "message": "How do I fix this bug?",
        "image_paths": [],
        "image_urls": []
    }],
    categories=categories,
    examples=examples
)
```

### File-Based Classification
The system supports loading classification configurations from JSON files:

```python
# Load and use a classification config
selections, usage = llm.classify_from_file(
    "configs/bug_classifier.json",
    messages=[{
        "message_type": "human",
        "message": "The app crashes on startup",
        "image_paths": [],
        "image_urls": []
    }]
)
```

Example configuration file (`bug_classifier.json`):
```json
{
    "name": "bug_classifier",
    "description": "Classifies bug reports by severity",
    "categories": {
        "critical": "System-breaking issues",
        "major": "Significant issues",
        "minor": "Small issues"
    },
    "examples": [
        {
            "user_input": "System crashes on startup",
            "doc_str": "Critical system failure",
            "selection": "critical"
        }
    ],
    "max_options": 1,
    "metadata": {
        "version": "1.0",
        "author": "team_name"
    }
}
```

### Response Format
The classification system uses a strict XML-based response format:

1. Single Category:
   ```xml
   <choice>1</choice>
   ```

2. Multiple Categories:
   ```xml
   <choices>1,3</choices>
   ```

The system is designed to be robust against model verbosity:
- Only content within XML tags is considered
- Additional explanatory text is ignored
- Numbers must correspond to category positions (1-based indexing)
- Invalid or missing tags will raise appropriate errors

## Image Handling

Each provider has specific image handling requirements:

1. **OpenAI**
   - Format: JPEG
   - Encoding: Base64 with data URI prefix
   - Size: Automatically scaled to optimal dimensions
   - Cost: Based on image size and detail level

2. **Anthropic**
   - Format: JPEG
   - Encoding: Base64 with media type
   - Quality: 95%
   - Alpha: Converted to RGB with white background

3. **Bedrock**
   - Format: PNG
   - Encoding: Raw bytes
   - Alpha: Preserved
   - Size: Original dimensions maintained

4. **Google**
   - Format: Varies by model
   - Encoding: According to Google API requirements
   - Size: Optimized for selected model

## Usage Example

```python
from lluminary.models.router import get_llm_from_model

# Initialize a model
llm = get_llm_from_model("claude-sonnet-3.5")

# Generate a response
response, usage = llm.generate(
    event_id="test_event",
    system_prompt="You are a helpful AI assistant.",
    messages=[{
        "message_type": "human",
        "message": "What's in this image?",
        "image_paths": ["path/to/image.jpg"],
        "image_urls": ["https://example.com/image.jpg"]
    }],
    max_tokens=300,
    temp=0.0
)

# Access usage statistics
print(f"Cost: ${usage['total_cost']:.6f}")
print(f"Tokens: {usage['total_tokens']}")
```

## Available Models

- **OpenAI**
  - `gpt-4o`
  - `gpt-4o-mini`
  - `o1`
  - `o3-mini`

- **Anthropic**
  - `claude-haiku-3.5`
  - `claude-sonnet-3.5`

- **Bedrock**
  - `bedrock-claude-haiku-3.5`
  - `bedrock-claude-sonnet-3.5-v1`
  - `bedrock-claude-sonnet-3.5-v2`

- **Google**
  - `gemini-pro`
  - `gemini-pro-vision`
  - `gemini-ultra`

- **Cohere**
  - `cohere-embed`
  - `cohere-rerank`

## Authentication

Each provider requires specific authentication. There are two authentication methods:

### Environment Variables
1. **OpenAI**: Set `OPENAI_API_KEY` environment variable
2. **Anthropic**: Set `ANTHROPIC_API_KEY` environment variable
3. **Google**: Set `GOOGLE_API_KEY` environment variable
4. **Bedrock**: AWS credentials via boto3 (default credentials chain)
5. **Cohere**: Set `COHERE_API_KEY` environment variable

### AWS Secrets Manager (Fallback)
If environment variables are not set, the library falls back to AWS Secrets Manager:
1. **OpenAI**: API key from AWS Secrets Manager (`openai_api_key`)
2. **Anthropic**: API key from AWS Secrets Manager (`anthropic_api_key`)
3. **Google**: API key from AWS Secrets Manager (`google_api_key`)
4. **Bedrock**: AWS credentials via boto3 (default credentials chain)
5. **Cohere**: API key from AWS Secrets Manager (`cohere_api_key`)

## Best Practices

1. **Error Handling**
   - Use `LLMMistake` for expected errors
   - Provide clear error messages
   - Set appropriate retry limits

2. **Message Management**
   - Always use the standard message format
   - Don't modify original message lists
   - Include all required fields

3. **Cost Control**
   - Monitor cumulative usage
   - Set appropriate token limits
   - Use cost estimation before large operations

4. **Testing**
   - Run full test suite before commits
   - Test with all supported providers
   - Verify error handling paths
