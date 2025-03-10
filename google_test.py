"""
Standalone test for Google Gemini API using GOOGLE_GEMINI_API_TOKEN environment variable.
This script also captures full response structures for use in mocks.
"""

import json
import os
import sys

# Verify environment variable exists
token = os.environ.get("GOOGLE_GEMINI_API_TOKEN")
if not token:
    print("❌ Error: GOOGLE_GEMINI_API_TOKEN environment variable not found")
    sys.exit(1)

print(
    f"✓ Found GOOGLE_GEMINI_API_TOKEN: {token[:5]}...{token[-5:] if len(token) > 10 else ''}"
)

# Map to GOOGLE_API_KEY for the test
os.environ["GOOGLE_API_KEY"] = token
print("✓ Mapped GOOGLE_GEMINI_API_TOKEN to GOOGLE_API_KEY for testing")

# Verify Google API packages
try:
    print("Importing Google Generative AI package...")
    import google.generativeai as genai

    print("✓ Google Generative AI package found")
except ImportError:
    print("❌ Error: Google Generative AI package not installed")
    print("   Please install with: pip install google-generativeai")
    sys.exit(1)

# Create directory for mock data if it doesn't exist
mock_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_data")
os.makedirs(mock_dir, exist_ok=True)


def save_response_for_mock(model_name, response, prefix="response"):
    """Save response structure for use in mocks."""
    try:
        # Convert response to dict for saving
        response_dict = {
            "text": response.text,
            "candidates": [
                {
                    "content": {
                        "parts": (
                            [{"text": part.text} for part in candidate.content.parts]
                            if hasattr(candidate.content, "parts")
                            else []
                        )
                    },
                    "finish_reason": getattr(candidate, "finish_reason", "STOP"),
                    "safety_ratings": getattr(candidate, "safety_ratings", []),
                    "index": getattr(candidate, "index", 0),
                }
                for candidate in response.candidates
            ],
            "usage_metadata": {
                "prompt_token_count": getattr(
                    response.usage_metadata, "prompt_token_count", 0
                ),
                "candidates_token_count": getattr(
                    response.usage_metadata, "candidates_token_count", 0
                ),
                "total_token_count": getattr(
                    response.usage_metadata, "total_token_count", 0
                ),
            },
        }

        # Save to file
        filename = f"{prefix}_{model_name.replace('-', '_')}.json"
        filepath = os.path.join(mock_dir, filename)
        with open(filepath, "w") as f:
            json.dump(response_dict, f, indent=2)

        print(f"✓ Saved response structure to {filepath}")

        # Print token counts for reference
        print(
            f"  Token counts: prompt={response_dict['usage_metadata']['prompt_token_count']}, "
            + f"output={response_dict['usage_metadata']['candidates_token_count']}, "
            + f"total={response_dict['usage_metadata']['total_token_count']}"
        )

    except Exception as e:
        print(f"❌ Failed to save response structure: {e}")


# Simple direct test with Google API
print("\nTesting Google Gemini API directly...\n")

try:
    # Set up the API key
    genai.configure(api_key=token)

    # Create a model
    print("Creating model instance...")
    # Try first with a known working model
    model_name = "gemini-1.5-flash"
    print(f"Testing with model: {model_name}")
    model = genai.GenerativeModel(model_name)
    print("✓ Model created")

    # Generate content
    print("Generating content...")
    response = model.generate_content(
        "What is the capital of France? Keep your answer brief."
    )

    # Display response
    print("\nResponse from Gemini API:")
    print(f"{response.text}\n")

    # Save response for mocks
    save_response_for_mock(model_name, response)

    # Now try with different request format (system prompt, structured messages)
    print("\nTesting with system prompt and messages format...")
    response_with_system = model.generate_content(
        contents=[{"role": "user", "parts": ["What's the weather like in Paris?"]}],
        generation_config={"temperature": 0.2},
        system_instruction="You are a helpful weather assistant. Be brief and concise.",
    )
    print(f"Response with system prompt: {response_with_system.text}\n")
    save_response_for_mock(model_name, response_with_system, "response_system")

    # Now try with a 2.0 model
    print("\nTesting with Gemini 2.0 model...")
    try:
        model_2_name = "gemini-2.0-flash"
        print(f"Testing with model: {model_2_name}")
        model_2 = genai.GenerativeModel(model_2_name)
        response_2 = model_2.generate_content(
            "What is the capital of France? Keep your answer brief."
        )
        print("\nResponse from Gemini 2.0:")
        print(f"{response_2.text}\n")

        # Save 2.0 response for mocks
        save_response_for_mock(model_2_name, response_2)

        print("✅ Gemini 2.0 model test successful!")
    except Exception as e2:
        print(f"❌ Error with Gemini 2.0 model: {type(e2).__name__}")
        print(f"   {e2!s}")
        print(
            "   Note: This may not be an API key issue, just a model availability issue"
        )

    # Try streaming (needed for mocks)
    print("\nTesting streaming response...")
    try:
        stream = model.generate_content(
            "Explain quantum computing in one sentence.", stream=True
        )
        print("Streaming response:")

        # Record all chunks for mock
        chunks = []
        full_response = ""

        for chunk in stream:
            if hasattr(chunk, "text"):
                print(f"  Chunk: {chunk.text}")
                full_response += chunk.text
                chunks.append({"text": chunk.text})

        print(f"\nFull streamed response: {full_response}")

        # Save streaming chunks for mocks
        stream_file = os.path.join(
            mock_dir, f"stream_{model_name.replace('-', '_')}.json"
        )
        with open(stream_file, "w") as f:
            json.dump(chunks, f, indent=2)
        print(f"✓ Saved streaming chunks to {stream_file}")
    except Exception as e3:
        print(f"❌ Error with streaming: {type(e3).__name__}")
        print(f"   {e3!s}")

    # If we get here, the API key is working
    print(
        "\n✅ Google Gemini API test successful! Your GOOGLE_GEMINI_API_TOKEN works properly."
    )
    print(f"Mock data saved to {mock_dir}")

except Exception as e:
    print(f"❌ Error testing Google Gemini API: {type(e).__name__}")
    print(f"   {e!s}")
