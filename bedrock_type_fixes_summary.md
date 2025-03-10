# Bedrock Provider Type Fixes Summary

## Issues Identified

1. **Return Type Mismatch in `stream_generate` Method**:
   - Base LLM class defines return type as `Iterator[Tuple[str, Dict[str, Any]]]`
   - BedrockLLM implementation uses `Generator[Tuple[str, Dict[str, Any]], None, None]`
   - This causes a type compatibility error

2. **Undefined `callback` Variables**:
   - There are references to undefined `callback` variables in the code
   - This suggests there might be multiple implementations of the `stream_generate` method

3. **Redundant Cast to "bytes"**:
   - There's a redundant cast to "bytes" at line 501
   - This is a minor issue that doesn't affect functionality

4. **Missing Type Stubs for External Libraries**:
   - Missing type stubs for boto3, requests, and botocore.exceptions
   - These can be ignored with `--ignore-missing-imports` or by installing type stubs

## Recommended Fixes

### 1. Fix Return Type Mismatch

```python
# Change this:
def stream_generate(
    self,
    event_id: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1000,
    temp: float = 0.0,
    functions: Optional[List[Callable[..., Any]]] = None,
    callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    # ...

# To this:
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
    # ...
```

### 2. Fix Undefined Callback Variables

The current implementation has issues with undefined callback variables. We should ensure that the callback parameter is properly defined and used:

```python
# Ensure callback is properly defined in the method signature
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
    # ...

    # Use callback properly
    if callback:
        callback(response_text, usage_info)
```

### 3. Fix Arithmetic Operations

Ensure all arithmetic operations have explicit type conversions:

```python
# Extract cost values with proper type handling
read_token_cost = float(costs.get("input_cost", 0.0) or 0.0)
write_token_cost = float(costs.get("output_cost", 0.0) or 0.0)
image_cost = float(costs.get("image_cost", 0.0) or 0.0)

# Ensure we're working with floats for all values before arithmetic operations
read_tokens = float(usage_stats["read_tokens"])
write_tokens = float(usage_stats["write_tokens"])
image_count = float(usage_stats.get("images", 0))

# Calculate costs
read_cost = read_tokens * read_token_cost
write_cost = write_tokens * write_token_cost
image_cost_total = image_count * image_cost if image_count > 0 else 0.0
total_cost = read_cost + write_cost + image_cost_total
```

### 4. Use TypedDict for Complex Dictionary Structures

Define TypedDict classes for all complex dictionary structures:

```python
class TextContent(TypedDict):
    text: str

class ImageSource(TypedDict):
    bytes: bytes

class ImageFormat(TypedDict):
    format: str
    source: ImageSource

class ImageContent(TypedDict):
    image: ImageFormat

# ... more TypedDict definitions ...

# Use these TypedDict classes in the code
def _format_messages_for_model(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted_messages: List[Dict[str, List[ContentPart]]] = []

    for msg in messages:
        # ... code ...

        # Add text content
        if msg.get("message"):
            text_content: TextContent = {"text": msg["message"]}
            content.append(text_content)

        # ... more code ...
```

### 5. Install Type Stubs for External Libraries

```bash
pip install types-requests
```

## Verification Steps

After implementing these fixes, verify them with:

```bash
python -m mypy src/lluminary/models/providers/bedrock.py --ignore-missing-imports
python -m pytest tests/unit/test_bedrock_provider.py
```

## Conclusion

By implementing these fixes, we can resolve the type checking issues in the Bedrock provider implementation. This will improve the type safety of the code and make it easier to maintain in the future.
