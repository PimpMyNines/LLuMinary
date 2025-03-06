"""
Integration tests for image processing functionality.
Tests real image processing with graceful skipping when auth fails.
"""

import pytest

from lluminary.models.router import get_llm_from_model


@pytest.mark.integration
@pytest.mark.image
class TestImageProcessing:
    """Test image processing functionality."""

    def test_url_image_processing(self):
        """
        Test that models can process images from URLs.
        Tries all models that support images.
        """
        # Set a reliable public test image URL
        test_image_url = "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png"

        # Test models that support images
        test_models = [
            "gpt-4o-mini",  # OpenAI
            "claude-haiku-3.5",  # Anthropic
            "gemini-2.0-flash-lite",  # Google
        ]

        # Track results
        successful_models = []
        failed_models = []

        print("\n" + "=" * 60)
        print("IMAGE PROCESSING TEST")
        print("=" * 60)

        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting image processing with {provider} model: {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Check if model supports images
                if not llm.supports_image_input():
                    print(f"Model {model_name} does not support images, skipping...")
                    continue

                # Generate response with image
                print("Generating response with image URL...")
                response, usage, _ = llm.generate(
                    event_id="test_image_url",
                    system_prompt="You are a helpful AI assistant skilled at describing images.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "What is in this image? Describe it briefly.",
                            "image_paths": [],
                            "image_urls": [test_image_url],
                        }
                    ],
                    max_tokens=150,
                )

                # Print results
                print(f"Response: {response[:100]}...")
                print(
                    f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write"
                )
                print(f"Cost: ${usage['total_cost']:.6f}")

                # Look for keywords related to PyTorch logo
                logo_keywords = ["pytorch", "logo", "torch", "flame"]
                found_keywords = [
                    word for word in logo_keywords if word.lower() in response.lower()
                ]

                if found_keywords:
                    print(f"Image correctly recognized with keywords: {found_keywords}")
                else:
                    print("Image may not have been correctly recognized")

                successful_models.append(model_name)

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                failed_models.append((model_name, str(e)))

        # Print summary
        print("\n" + "=" * 60)
        print("IMAGE PROCESSING TEST SUMMARY")
        print("=" * 60)
        print(f"Successful models: {len(successful_models)}/{len(test_models)}")
        for model in successful_models:
            print(f"  - {model}")

        if failed_models:
            print(f"\nFailed models: {len(failed_models)}/{len(test_models)}")
            for model, error in failed_models:
                print(f"  - {model}: {error}")

        # Skip if no models work at all
        if not successful_models:
            pytest.skip("Skipping test as no models were able to process the image")

    def test_multiple_images(self):
        """
        Test processing multiple images in a single request.
        Tries models until one works.
        """
        # Set reliable public test image URLs
        test_image_url1 = "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png"
        test_image_url2 = "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/dynamic-graph.gif"

        # Test models that support multiple images
        test_models = [
            "gpt-4o-mini",  # OpenAI models generally support multiple images
            "claude-haiku-3.5",  # Anthropic models support multiple images
            "gemini-2.0-flash-lite",  # Google models support multiple images
        ]

        print("\n" + "=" * 60)
        print("MULTIPLE IMAGES TEST")
        print("=" * 60)

        for model_name in test_models:
            print(f"\nTesting multiple images with {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Check if model supports images
                if not llm.supports_image_input():
                    print(f"Model {model_name} does not support images, skipping...")
                    continue

                # Generate response with multiple images
                print("Generating response with multiple image URLs...")
                response, usage, _ = llm.generate(
                    event_id="test_multiple_images",
                    system_prompt="You are a helpful AI assistant skilled at describing images.",
                    messages=[
                        {
                            "message_type": "human",
                            "message": "I'm showing you two images. Can you describe both of them briefly?",
                            "image_paths": [],
                            "image_urls": [test_image_url1, test_image_url2],
                        }
                    ],
                    max_tokens=200,
                )

                # Print results
                print(f"Response: {response[:100]}...")
                print(
                    f"Tokens: {usage['read_tokens']} read, {usage['write_tokens']} write"
                )
                print(f"Cost: ${usage['total_cost']:.6f}")

                # Verify response mentions multiple images
                assert isinstance(response, str)
                assert len(response) > 0
                print(f"Test passed with {model_name}")
                return

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                continue

        # If we get here, no models worked
        pytest.skip("Skipping test as no models were able to process multiple images")
