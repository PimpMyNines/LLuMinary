# LLM Models Support Matrix

This document provides a comprehensive overview of all the models supported by the LLM handler framework, including their capabilities, limitations, and pricing information.

## Models by Provider

| Provider | Model | Context Window | Image Support | Tool/Function Calling | Reasoning/Thinking | Input Cost (per token) | Output Cost (per token) | Image Cost |
|----------|-------|----------------|---------------|----------------------|-------------------|------------------------|-------------------------|------------|

### Google (Gemini)

| Model | Context Window | Image Support | Tool/Function Calling | Reasoning/Thinking | Input Cost (per token) | Output Cost (per token) | Image Cost |
|-------|----------------|---------------|----------------------|-------------------|------------------------|-------------------------|------------|
| gemini-2.0-flash | 128,000 | ✅ | ✅ | ❌ | $0.0000025 | $0.00001 | $0.001 per image |
| gemini-2.0-flash-lite-preview-02-05 | 128,000 | ✅ | ✅ | ❌ | $0.000001 | $0.000004 | $0.0005 per image |
| gemini-2.0-pro-exp-02-05 | 128,000 | ✅ | ✅ | ❌ | $0.000003 | $0.000012 | $0.002 per image |
| gemini-2.0-flash-thinking-exp-01-21 | 128,000 | ✅ | ✅ | ✅ | $0.000004 | $0.000016 | $0.002 per image |

### OpenAI

| Model | Context Window | Image Support | Tool/Function Calling | Reasoning/Thinking | Input Cost (per token) | Output Cost (per token) | Image Cost |
|-------|----------------|---------------|----------------------|-------------------|------------------------|-------------------------|------------|
| gpt-4.5-preview | 128,000 | ✅ | ✅ | ❌ | $0.0000750 | $0.00015 | Variable* |
| gpt-4o | 128,000 | ✅ | ✅ | ❌ | $0.0000025 | $0.00001 | Variable* |
| gpt-4o-mini | 128,000 | ✅ | ✅ | ❌ | $0.00000015 | $0.0000006 | Variable* |
| o1 | 200,000 | ✅ | ✅ | ✅ | $0.000015 | $0.00006 | Variable* |
| o3-mini | 200,000 | ✅ | ✅ | ✅ | $0.0000011 | $0.0000044 | Variable* |

\* OpenAI's image costs depend on resolution and detail level. Low detail: 85 tokens, High detail: varies by size and tiles.

### Anthropic (Direct API)

| Model | Context Window | Image Support | Tool/Function Calling | Reasoning/Thinking | Input Cost (per token) | Output Cost (per token) | Image Cost |
|-------|----------------|---------------|----------------------|-------------------|------------------------|-------------------------|------------|
| claude-3-5-sonnet-20240620-v1:0 | 200,000 | ✅ | ✅ | ❌ | $0.000003 | $0.000015 | $0.024 per image |
| claude-3-haiku-20240307-v1:0 | 200,000 | ✅ | ✅ | ❌ | $0.00000025 | $0.00000125 | $0.024 per image |
| claude-3-opus-20240229-v1:0 | 200,000 | ✅ | ✅ | ❌ | $0.000015 | $0.000075 | $0.024 per image |
| claude-3-sonnet-20240229-v1:0 | 200,000 | ✅ | ✅ | ❌ | $0.000003 | $0.000015 | $0.024 per image |
| claude-3-haiku-20240307-v1:0 | 200,000 | ✅ | ✅ | ❌ | $0.00000025 | $0.00000125 | $0.024 per image |
| claude-3-7-sonnet-preview-20240626-v1:0 | 200,000 | ✅ | ✅ | ✅ | $0.000003 | $0.000015 | $0.024 per image |

### AWS Bedrock (Claude models)

| Model | Context Window | Image Support | Tool/Function Calling | Reasoning/Thinking | Input Cost (per token) | Output Cost (per token) | Image Cost |
|-------|----------------|---------------|----------------------|-------------------|------------------------|-------------------------|------------|
| us.anthropic.claude-3-5-haiku-20241022-v1:0 | 200,000 | ✅ | ✅ | ❌ | $0.000001 | $0.000005 | $0.024 per image |
| us.anthropic.claude-3-5-sonnet-20240620-v1:0 | 200,000 | ✅ | ✅ | ❌ | $0.000003 | $0.000015 | $0.024 per image |
| us.anthropic.claude-3-5-sonnet-20241022-v2:0 | 200,000 | ✅ | ✅ | ❌ | $0.000003 | $0.000015 | $0.024 per image |
| us.anthropic.claude-3-7-sonnet-20250219-v1:0 | 200,000 | ✅ | ✅ | ✅ | $0.000003 | $0.000015 | $0.024 per image |

## Feature Support Details

### Image Support

All models listed with image support (✅) can process images as part of the input, but with differences in:

- **Format**:
  - Google: Supports multiple image formats
  - OpenAI: Processes images as JPEG with base64 encoding
  - Anthropic: Supports multiple image formats with various sizing
  - AWS Bedrock: Uses PNG format with preserved transparency

- **Costing**:
  - Google: Fixed cost per image
  - OpenAI: Variable cost based on resolution and detail level
  - Anthropic: Fixed cost per image
  - AWS Bedrock: Fixed cost per image

### Tool/Function Calling

Models supporting tool/function calling can:
- Parse and understand function schemas
- Choose appropriate functions to call
- Format arguments correctly for function execution
- Process function results and incorporate them into responses

Each provider has a unique format for tool definitions and responses:
- Google: Uses Part.from_function_call and Part.from_function_response
- OpenAI: Uses a function calling format with "arguments" as a JSON string
- Anthropic: Uses a structured content format with toolUse and toolResult objects
- AWS Bedrock: Uses toolSpec, toolUse, and toolResult formats

### Reasoning/Thinking

Models with reasoning/thinking capabilities can:
- Generate detailed step-by-step reasoning
- Provide internal thought process (visible or invisible to user)
- Solve complex problems more systematically
- Include "reasoning signatures" in some providers

Available on:
- Google: gemini-2.0-flash-thinking-exp-01-21
- OpenAI: o1, o3-mini
- Anthropic: claude-3-7-sonnet-preview models
- AWS Bedrock: claude-3-7-sonnet models

## Usage Notes

1. **Context Window**: Represents the maximum number of tokens the model can process in a single conversation (input + output).

2. **Token Calculation**: Different providers calculate tokens differently:
   - Text tokens: ~4 characters per token (English, varies by language)
   - Image tokens: Different calculation methods per provider

3. **Costs**: Prices are in USD and subject to change. Always check the provider's pricing page for current rates.

4. **Model Availability**: Some models may be experimental or in preview. Availability might change.

5. **Rate Limits**: Each provider implements different rate limiting policies. Check provider documentation for details.

## Provider-Specific Considerations

### Google (Gemini)
- Requires specific API version for thinking models (v1alpha)
- Uses Content and Part objects for message structuring
- Simple image cost model with fixed cost per image

### OpenAI
- Implements reasoning via "reasoning_effort" parameter
- Uses complex image token calculation based on resolution and detail
- Can use either string content or array of content objects

### Anthropic
- Advanced image handling capabilities
- Structured content format with multiple parts
- Needs specific authentication via API keys

### AWS Bedrock
- Implements automated retries for AWS-specific errors
- Uses PNG image format with transparency support
- Requires AWS credentials and permissions
- Direct billing through AWS account
