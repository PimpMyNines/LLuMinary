import re
from typing import List
from xml.etree import ElementTree as ET


class ValidationError(Exception):
    """Raised when classification response validation fails."""

    pass


def validate_classification_response(
    response: str, num_categories: int, max_options: int = 1
) -> List[int]:
    """
    Validate and parse classification response.

    Args:
        response: Raw response string from LLM
        num_categories: Total number of available categories
        max_options: Maximum number of categories that can be selected

    Returns:
        List of selected category indices (1-based)

    Raises:
        ValidationError: If response format is invalid
    """
    try:
        # Extract XML content
        match = re.search(r"<choice[s]?>(.*?)</choice[s]?>", response)
        if not match:
            raise ValidationError("No valid XML tags found in response")

        content = match.group(1).strip()

        # Parse selections
        if "," in content:
            selections = [int(x.strip()) for x in content.split(",")]
        else:
            selections = [int(content)]

        # Validate selections
        if not selections:
            raise ValidationError("No categories selected")

        if len(selections) > max_options:
            raise ValidationError(
                f"Too many categories selected: {len(selections)} > {max_options}"
            )

        invalid = [x for x in selections if x < 1 or x > num_categories]
        if invalid:
            raise ValidationError(
                f"Invalid category indices: {invalid}. Must be between 1 and {num_categories}"
            )

        return sorted(selections)

    except (ValueError, ET.ParseError) as e:
        raise ValidationError(f"Failed to parse response: {e!s}")


def validate_xml_format(response: str) -> bool:
    """
    Validate that response contains valid XML tags.

    Args:
        response: Raw response string

    Returns:
        True if valid XML format found, False otherwise
    """
    try:
        # Look for either <choice> or <choices> tags
        single = bool(re.search(r"<choice>.*?</choice>", response))
        multiple = bool(re.search(r"<choices>.*?</choices>", response))
        return single or multiple
    except:
        return False
