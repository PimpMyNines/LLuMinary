"""CLI tools for classification management."""

import json
import sys
from typing import Optional

import click

from ..models.classification import ClassificationConfig, ClassificationLibrary
from ..models.router import get_llm_from_model


@click.group()
def cli() -> None:
    """Classification management tools."""
    pass


@cli.command()
@click.argument("config_dir", type=click.Path(exists=True))
def list_configs(config_dir: str) -> None:
    """List available classification configurations."""
    library = ClassificationLibrary(config_dir)
    library.load_configs()

    configs = library.list_configs()
    if not configs:
        click.echo("No classification configs found.")
        return

    for config in configs:
        click.echo(f"\n{config['name']}:")
        click.echo(f"  Description: {config['description']}")
        click.echo(f"  Categories: {', '.join(config['categories'])}")
        if config["metadata"]:
            click.echo(
                f"  Metadata: {json.dumps(config['metadata'], indent=2)}")


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str) -> None:
    """Validate a classification configuration file."""
    try:
        config = ClassificationConfig.from_file(config_path)
        config.validate()
        click.echo(f"✓ Configuration {config.name} is valid")

        # Show summary
        click.echo("\nSummary:")
        click.echo(f"- Categories: {len(config.categories)}")
        click.echo(f"- Examples: {len(config.examples)}")
        click.echo(f"- Max options: {config.max_options}")

    except Exception as e:
        click.echo(f"✗ Configuration is invalid: {e!s}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("message", type=str)
@click.option(
    "--model", default="claude-sonnet-3.5",
    help="Model to use for classification"
)
@click.option("--system-prompt", help="Optional system prompt")
def test(
        config_path: str, message: str, model: str, system_prompt: Optional[str]
) -> None:
    """Test a classification configuration with a message."""
    try:
        # Initialize model
        llm = get_llm_from_model(model)

        # Prepare message
        messages = [
            {
                "message_type": "human",
                "message": message,
                "image_paths": [],
                "image_urls": [],
            }
        ]

        # Classify
        categories, usage = llm.classify_from_file(
            config_path, messages, system_prompt=system_prompt
        )

        # Show results
        click.echo("\nClassification Results:")
        click.echo(f"Categories: {', '.join(categories)}")
        click.echo("\nUsage Statistics:")
        click.echo(f"- Total tokens: {usage['total_tokens']}")
        click.echo(f"- Cost: ${usage['total_cost']:.6f}")

    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("output_path", type=click.Path())
@click.option("--name", prompt=True, help="Name of the classifier")
@click.option("--description", prompt=True,
              help="Description of the classifier")
def create(output_path: str, name: str, description: str) -> None:
    """Create a new classification configuration interactively."""
    categories = {}
    examples = []

    # Get categories
    click.echo("\nEnter categories (empty name to finish):")
    while True:
        cat_name = click.prompt("Category name", default="", show_default=False)
        if not cat_name:
            break
        cat_desc = click.prompt("Category description")
        categories[cat_name] = cat_desc

    # Get examples
    if click.confirm("\nAdd example classifications?", default=True):
        click.echo("\nEnter examples (empty input to finish):")
        while True:
            user_input = click.prompt("User input", default="",
                                      show_default=False)
            if not user_input:
                break
            doc_str = click.prompt("Reasoning")
            selection = click.prompt(
                "Category", type=click.Choice(list(categories.keys()))
            )
            examples.append(
                {"user_input": user_input, "doc_str": doc_str,
                 "selection": selection}
            )

    # Get max options
    max_options = click.prompt(
        "Maximum number of categories to select", type=int, default=1
    )

    # Create config
    config = ClassificationConfig(
        name=name,
        description=description,
        categories=categories,
        examples=examples,
        max_options=max_options,
        metadata={
            "author": click.prompt("Author", default="unknown"),
            "version": "1.0",
            "created_at": click.prompt("Creation date", default="2024-02-08"),
            "tags": click.prompt("Tags (comma-separated)", default="").split(
                ","),
        },
    )

    # Validate and save
    try:
        config.validate()
        config.save(output_path)
        click.echo(f"\n✓ Configuration saved to {output_path}")
    except Exception as e:
        click.echo(f"Error: {e!s}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
