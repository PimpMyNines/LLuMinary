"""
Example demonstrating the thinking budget feature with Claude 3.7.
"""

from lluminary import get_llm_from_model


def thinking_budget_example():
    """
    Demonstrate the thinking budget feature with Claude 3.7.

    The thinking budget allows models to perform more complex reasoning
    before generating a response. This is particularly useful for tasks
    requiring extensive reasoning like math problems or complex analyses.
    """
    try:
        # Initialize a model that supports thinking
        llm = get_llm_from_model("claude-sonnet-3.7")

        # Check if model supports thinking budget
        if not hasattr(llm, "thinking_models") or "claude-sonnet-3.7" not in getattr(
            llm, "thinking_models", []
        ):
            print("This model doesn't support thinking budget. Try using Claude 3.7")
            return None, None

        # Complex problem requiring reasoning
        messages = [
            {
                "message_type": "human",
                "message": """Solve this step by step:

            A train leaves station A traveling at 60 mph toward station B, which is 300 miles away.
            At the same time, a train leaves station B traveling at 45 mph toward station A.

            1. How long will it take for the trains to meet?
            2. How far from station A will they meet?

            Show all your work and explain each step of your reasoning.
            """,
                "image_paths": [],
                "image_urls": [],
            }
        ]

        # Generate response with thinking budget
        print("Generating response with thinking budget...")
        response_with_thinking, usage_with_thinking, _ = llm.generate(
            event_id="thinking_budget_example",
            system_prompt="You are a math expert who solves problems step by step.",
            messages=messages,
            max_tokens=700,
            thinking_budget=8000,  # Allocate tokens for thinking
        )

        # Print results
        print("\nResponse with thinking budget:")
        print("=" * 80)
        print(response_with_thinking)
        print("=" * 80)
        print(
            f"Tokens: {usage_with_thinking['read_tokens']} read, {usage_with_thinking['write_tokens']} write"
        )
        print(f"Cost: ${usage_with_thinking['total_cost']:.6f}")

        # For comparison, generate a response without thinking budget
        print("\nGenerating response without thinking budget for comparison...")
        response_without_thinking, usage_without_thinking, _ = llm.generate(
            event_id="no_thinking_budget_example",
            system_prompt="You are a math expert who solves problems step by step.",
            messages=messages,
            max_tokens=700,
            # No thinking budget specified
        )

        # Print comparison results
        print("\nResponse without thinking budget:")
        print("=" * 80)
        print(response_without_thinking)
        print("=" * 80)
        print(
            f"Tokens: {usage_without_thinking['read_tokens']} read, {usage_without_thinking['write_tokens']} write"
        )
        print(f"Cost: ${usage_without_thinking['total_cost']:.6f}")

        # Show the difference
        print("\nComparing results:")
        print(f"With thinking budget cost: ${usage_with_thinking['total_cost']:.6f}")
        print(
            f"Without thinking budget cost: ${usage_without_thinking['total_cost']:.6f}"
        )
        print(
            f"Difference: ${usage_with_thinking['total_cost'] - usage_without_thinking['total_cost']:.6f}"
        )

        return (response_with_thinking, usage_with_thinking), (
            response_without_thinking,
            usage_without_thinking,
        )

    except Exception as e:
        print(f"Error: {e!s}")
        return None, None


if __name__ == "__main__":
    print("== Thinking Budget Example ==")
    print("This example demonstrates Claude 3.7's thinking budget feature.")
    print("Note: This requires Claude 3.7 to be available and authenticated.")
    thinking_budget_example()
