# Debug Utilities

This directory contains various debugging and development utilities used during the development of the LLMHandler library.

## Debug Files

- `debug_bedrock.py` - Debug utilities for AWS Bedrock provider integration
- `debug_generation_test.py` - Test utilities for text generation functionality
- `debug_google.py` - Debug utilities for Google provider integration
- `debug_message_test.py` - Test utilities for message formatting
- `debug_openai.py` - Debug utilities for OpenAI provider integration
- `debug_router.py` - Debug utilities for the model router
- `debug_test.py` - General debug test utilities
- `debug_tools_test.py` - Debug utilities for tool functionality
- `debug_type_validator.py` - Debug utilities for type validation
- `debug_union_test.py` - Debug utilities for union type handling
- `test.py` - Main test utility for manually testing various provider features

## Usage

These files are primarily for development purposes and should not be imported or used in production code. They provide convenient ways to test and debug specific functionality during the development process.

To run a debug file:

```bash
python -m debug.debug_filename
```

For example:

```bash
python -m debug.debug_openai
```

## Notes

- The debug files are excluded from the package distribution via the MANIFEST.in file
- Some debug files may require specific environment variables to be set (e.g., API keys for providers)
- These utilities are not covered by tests and may not always be up-to-date with the latest API changes
