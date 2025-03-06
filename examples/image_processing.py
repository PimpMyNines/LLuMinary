"""
Image processing examples for LLMHandler package.
"""

from lluminary import get_llm_from_model


def image_description(image_url):
    """Example of analyzing an image from a URL."""
    # Initialize a model that supports images
    llm = get_llm_from_model("gpt-4o-mini")  # could also use Claude or Gemini models

    # Ensure the model supports images
    if not llm.supports_image_input():
        print("This model doesn't support image input.")
        return None, None

    # Generate a description of the image
    response, usage, _ = llm.generate(
        event_id="image_description",
        system_prompt="You are a helpful AI assistant skilled at describing images accurately.",
        messages=[
            {
                "message_type": "human",
                "message": "Describe this image in detail. What do you see?",
                "image_paths": [],
                "image_urls": [image_url],
            }
        ],
        max_tokens=300,
    )

    print("Image description:")
    print(response)
    print(f"\nCost: ${usage['total_cost']:.6f}")
    print(f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write")

    return response, usage


def multiple_images(image_urls):
    """Example of processing multiple images in one request."""
    # Initialize a model that supports multiple images
    llm = get_llm_from_model(
        "claude-haiku-3.5"
    )  # could also use GPT-4 Vision or Gemini

    # Ensure the model supports images
    if not llm.supports_image_input():
        print("This model doesn't support image input.")
        return None, None

    # Generate a response comparing the images
    response, usage, _ = llm.generate(
        event_id="multiple_images",
        system_prompt="You are a helpful AI assistant that can analyze multiple images.",
        messages=[
            {
                "message_type": "human",
                "message": "Compare these images and tell me what's similar and different between them.",
                "image_paths": [],
                "image_urls": image_urls,
            }
        ],
        max_tokens=500,
    )

    print("\nMultiple image analysis:")
    print(response)
    print(f"\nCost: ${usage['total_cost']:.6f}")
    print(f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write")

    return response, usage


def image_classification(image_url):
    """Example of classifying an image."""
    # Initialize a model that supports images
    llm = get_llm_from_model("gemini-2.0-flash-lite")

    # Ensure the model supports images
    if not llm.supports_image_input():
        print("This model doesn't support image input.")
        return None, None

    # Define categories for classification
    categories = {
        "photo": "A photograph of a real scene or object",
        "illustration": "A drawing, painting, or digital art",
        "screenshot": "A captured image of a digital screen",
        "diagram": "A schematic, chart, or explanatory drawing",
    }

    # Message with image to classify
    message = {
        "message_type": "human",
        "message": "What type of image is this?",
        "image_paths": [],
        "image_urls": [image_url],
    }

    # Perform classification
    classification, usage = llm.classify(messages=[message], categories=categories)

    print("\nImage classification result:", classification)
    print(f"Cost: ${usage['total_cost']:.6f}")

    return classification, usage


if __name__ == "__main__":
    # Example with a public domain image
    pytorch_logo = "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png"
    python_logo = (
        "https://www.python.org/static/community_logos/python-logo-master-v3-TM.png"
    )

    print("== Single Image Description Example ==")
    image_description(pytorch_logo)

    print("\n== Multiple Images Example ==")
    multiple_images([pytorch_logo, python_logo])

    print("\n== Image Classification Example ==")
    image_classification(pytorch_logo)
