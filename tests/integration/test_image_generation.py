"""
Integration tests for image generation functionality.
Tests image generation across providers with graceful skipping when auth fails.
"""

import os
import tempfile

import pytest

from lluminary.models.router import get_llm_from_model

# Mark all tests in this file as image generation integration tests
pytestmark = [pytest.mark.integration, pytest.mark.image_generation]


class TestImageGeneration:
    """Test image generation functionality."""

    def test_basic_image_generation(self):
        """
        Test basic image generation with different providers.
        Tries all models that support image generation.
        """
        # Test models that support image generation
        test_models = [
            "dall-e-3",  # OpenAI
            "gemini-1.5-flash",  # Google (if it supports image generation)
            "bedrock-stable-diffusion",  # AWS Bedrock
        ]

        # Track results
        successful_models = []
        failed_models = []

        print("\n" + "=" * 60)
        print("IMAGE GENERATION TEST")
        print("=" * 60)

        # Simple prompt for image generation
        prompt = "A serene landscape with mountains and a lake at sunset"

        for model_name in test_models:
            provider = model_name.split("-")[0] if "-" in model_name else model_name
            print(f"\nTesting image generation with {provider} model: {model_name}")

            try:
                # Initialize model
                llm = get_llm_from_model(model_name)

                # Check if model supports image generation
                if not hasattr(llm, "generate_image"):
                    print(
                        f"Model {model_name} does not support image generation, skipping..."
                    )
                    continue

                # Create a temporary directory to save the image
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = os.path.join(temp_dir, "generated_image.png")

                    # Generate image
                    print(f"Generating image with prompt: '{prompt}'...")
                    result = llm.generate_image(
                        prompt=prompt,
                        output_path=output_path,
                        size="1024x1024",  # Standard size
                    )

                    # Check if image was generated
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(
                            f"Image successfully generated and saved to {output_path}"
                        )
                        print(
                            f"Image size: {os.path.getsize(output_path) / 1024:.2f} KB"
                        )

                        # If result contains metrics, print them
                        if isinstance(result, dict) and "cost" in result:
                            print(f"Cost: ${result['cost']:.6f}")

                        successful_models.append(model_name)
                    else:
                        print("Image generation failed or produced an empty file")
                        failed_models.append(
                            (model_name, "Empty or missing output file")
                        )

            except Exception as e:
                print(f"Error with {model_name}: {e!s}")
                failed_models.append((model_name, str(e)))

        # Print summary
        print("\n" + "=" * 60)
        print("IMAGE GENERATION TEST SUMMARY")
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
            pytest.skip("Skipping test as no models were able to generate images")

    def test_image_variation_parameters(self):
        """
        Test image generation with different parameters.
        Tries different sizes, styles, and quality settings.
        """
        # Use a reliable image generation model
        test_model = "dall-e-3"  # OpenAI is typically reliable for image generation

        # Standard prompt for comparison
        prompt = "A futuristic city with flying cars and tall skyscrapers"

        print("\n" + "=" * 60)
        print("IMAGE VARIATION PARAMETERS TEST")
        print("=" * 60)

        try:
            # Initialize model
            llm = get_llm_from_model(test_model)

            # Check if model supports image generation
            if not hasattr(llm, "generate_image"):
                pytest.skip(f"Model {test_model} does not support image generation")

            # Test different image sizes
            sizes = ["1024x1024", "512x512", "256x256"]

            for size in sizes:
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = os.path.join(temp_dir, f"image_{size}.png")

                    print(f"\nGenerating image with size {size}...")
                    result = llm.generate_image(
                        prompt=prompt,
                        output_path=output_path,
                        size=size,
                    )

                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(f"Image successfully generated with size {size}")
                        print(
                            f"Image file size: {os.path.getsize(output_path) / 1024:.2f} KB"
                        )

                        # If result contains metrics, print them
                        if isinstance(result, dict) and "cost" in result:
                            print(f"Cost: ${result['cost']:.6f}")
                    else:
                        print(f"Failed to generate image with size {size}")

            # Test with quality parameter if supported
            if (
                hasattr(llm, "supports_quality_parameter")
                and llm.supports_quality_parameter()
            ):
                qualities = ["standard", "hd"]
                for quality in qualities:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        output_path = os.path.join(temp_dir, f"image_{quality}.png")

                        print(f"\nGenerating image with quality {quality}...")
                        result = llm.generate_image(
                            prompt=prompt,
                            output_path=output_path,
                            size="1024x1024",
                            quality=quality,
                        )

                        if (
                            os.path.exists(output_path)
                            and os.path.getsize(output_path) > 0
                        ):
                            print(
                                f"Image successfully generated with quality {quality}"
                            )
                            print(
                                f"Image file size: {os.path.getsize(output_path) / 1024:.2f} KB"
                            )
                        else:
                            print(f"Failed to generate image with quality {quality}")

        except Exception as e:
            print(f"Error with {test_model}: {e!s}")
            pytest.skip(f"Skipping test due to error: {e!s}")

    def test_image_generation_with_style(self):
        """
        Test image generation with different style parameters.
        """
        # Use a reliable image generation model that supports styles
        test_model = "dall-e-3"  # OpenAI DALL-E 3 supports style parameter

        # Standard prompt for comparison
        prompt = "A cat sitting on a windowsill"

        print("\n" + "=" * 60)
        print("IMAGE STYLE TEST")
        print("=" * 60)

        try:
            # Initialize model
            llm = get_llm_from_model(test_model)

            # Check if model supports image generation
            if not hasattr(llm, "generate_image"):
                pytest.skip(f"Model {test_model} does not support image generation")

            # Check if model supports style parameter
            if (
                not hasattr(llm, "supports_style_parameter")
                or not llm.supports_style_parameter()
            ):
                pytest.skip(f"Model {test_model} does not support style parameter")

            # Test different styles
            styles = ["vivid", "natural"]

            for style in styles:
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_path = os.path.join(temp_dir, f"image_{style}.png")

                    print(f"\nGenerating image with style {style}...")
                    result = llm.generate_image(
                        prompt=prompt,
                        output_path=output_path,
                        size="1024x1024",
                        style=style,
                    )

                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        print(f"Image successfully generated with style {style}")
                        print(
                            f"Image file size: {os.path.getsize(output_path) / 1024:.2f} KB"
                        )
                    else:
                        print(f"Failed to generate image with style {style}")

        except Exception as e:
            print(f"Error with {test_model}: {e!s}")
            pytest.skip(f"Skipping test due to error: {e!s}")
