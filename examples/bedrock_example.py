"""
Example demonstrating how to use the AWS Bedrock provider with LLuMinary.

This example shows how to:
1. Use AWS profiles for authentication
2. Generate text with Claude models on Bedrock
3. Use streaming responses
4. Handle multiple message exchanges
"""

import os
from lluminary import get_llm_from_model

# Set AWS profile - for local development
AWS_PROFILE = "ai-dev"

def bedrock_basic_example():
    """Basic example showing text generation with AWS Bedrock."""
    # Create a handler with Bedrock LLM
    llm = get_llm_from_model(
        "bedrock-claude-sonnet-3.5-v2",
        provider="bedrock",
        profile_name=AWS_PROFILE
    )
    
    # Generate a response
    response, usage, _ = llm.generate(
        event_id="bedrock_basic_example",
        system_prompt="You are a helpful assistant.",
        messages=[
            {
                "message_type": "human",
                "message": "Explain how AWS Bedrock helps with LLM deployment in 3 sentences.",
                "image_paths": [],
                "image_urls": [],
            }
        ],
    )
    
    print("\n--- Basic Bedrock Example ---")
    print(response)
    

def bedrock_streaming_example():
    """Example showing streaming responses with AWS Bedrock."""
    # Create a handler with Bedrock LLM
    llm = get_llm_from_model(
        "bedrock-claude-sonnet-3.5-v2",
        provider="bedrock",
        profile_name=AWS_PROFILE
    )
    
    print("\n--- Bedrock Streaming Example ---")
    print("Response:")
    
    # Stream the response
    for chunk, usage_data in llm.stream_generate(
        event_id="bedrock_streaming_example",
        system_prompt="You are a helpful assistant.",
        messages=[
            {
                "message_type": "human",
                "message": "Count from 1 to 10, placing each number on a new line.",
                "image_paths": [],
                "image_urls": [],
            }
        ],
    ):
        print(chunk, end="", flush=True)
    print("\n")


def bedrock_multi_turn_example():
    """Example showing multi-turn conversation with AWS Bedrock."""
    # Create a handler with Bedrock LLM and keep conversation history
    llm = get_llm_from_model(
        "bedrock-claude-sonnet-3.5-v2",
        provider="bedrock",
        profile_name=AWS_PROFILE
    )
    
    print("\n--- Bedrock Multi-turn Conversation Example ---")
    
    # First message
    messages = [
        {
            "message_type": "human",
            "message": "Hi, I'm learning about cloud computing. Can you help me?",
            "image_paths": [],
            "image_urls": [],
        }
    ]
    
    response1, usage1, messages = llm.generate(
        event_id="bedrock_multi_turn_1",
        system_prompt="You are a helpful assistant.",
        messages=messages,
    )
    print("User: Hi, I'm learning about cloud computing. Can you help me?")
    print(f"Assistant: {response1}")
    
    # Add assistant's response to messages
    messages.append({
        "message_type": "human",
        "message": "What's the difference between AWS Bedrock and Amazon SageMaker?",
        "image_paths": [],
        "image_urls": [],
    })
    
    # Second message
    response2, usage2, updated_messages = llm.generate(
        event_id="bedrock_multi_turn_2",
        system_prompt="You are a helpful assistant.",
        messages=messages,
    )
    print("\nUser: What's the difference between AWS Bedrock and Amazon SageMaker?")
    print(f"Assistant: {response2}")


def bedrock_thinking_example():
    """Example showing thinking budget with Claude 3.7 Sonnet."""
    try:
        # Create a handler with the latest Claude model that supports thinking
        llm = get_llm_from_model(
            "bedrock-claude-sonnet-3.7",
            provider="bedrock",
            profile_name=AWS_PROFILE,
            thinking_budget=0.7  # Allocate 70% of tokens to thinking
        )
        
        print("\n--- Bedrock Thinking Budget Example ---")
        
        # Generate a response with thinking
        response, usage, _ = llm.generate(
            event_id="bedrock_thinking_example",
            system_prompt="You are a helpful assistant.",
            messages=[
                {
                    "message_type": "human",
                    "message": "Calculate the factorial of 10 step by step, then provide the final answer.",
                    "image_paths": [],
                    "image_urls": [],
                }
            ],
        )
        
        print(response)
    except Exception as e:
        print(f"Error with thinking budget example: {e}")
        print("This may require the Claude 3.7 Sonnet model, which might not be available.")


if __name__ == "__main__":
    print("Running AWS Bedrock examples...")
    
    try:
        bedrock_basic_example()
        bedrock_streaming_example()
        bedrock_multi_turn_example()
        bedrock_thinking_example()
    except Exception as e:
        print(f"Error running examples: {e}")
        print("\nNote: These examples require:")
        print("1. AWS credentials configured via the ai-dev profile")
        print("2. The profile must have access to the AWS Bedrock service")
        print("3. The AWS region must have the Claude models enabled")
        print("\nTo authenticate, run: aws sso login --profile ai-dev")