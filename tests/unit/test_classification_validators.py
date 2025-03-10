"""
Unit tests for the classification validators.
"""

import pytest

# Mark all tests in this file as classification tests
pytestmark = pytest.mark.classification

from lluminary.models.classification.validators import (
    ValidationError,
    validate_classification_response,
    validate_xml_format,
)


def test_validate_xml_format():
    """Test detection of valid XML tags in responses."""
    # Valid responses with single choice tag
    assert validate_xml_format("<choice>1</choice>") is True
    assert validate_xml_format("Some text before <choice>1</choice>") is True
    assert validate_xml_format("<choice>1</choice> and explanation after") is True

    # Valid responses with choices tag (plural)
    assert validate_xml_format("<choices>1,2,3</choices>") is True
    assert validate_xml_format("Text <choices>1,2</choices> more text") is True

    # Invalid responses
    assert validate_xml_format("No tags here") is False
    assert validate_xml_format("<wrong>1</wrong>") is False
    assert validate_xml_format("<choice>1<choice>") is False  # Unclosed tag
    assert validate_xml_format("<choices>1</choice>") is False  # Mismatched tags


def test_validate_classification_response_single():
    """Test validation of single category responses."""
    # Valid single selection
    result = validate_classification_response("<choice>1</choice>", num_categories=3)
    assert result == [1]

    # Valid with surrounding text
    result = validate_classification_response(
        "I think this is category 1: <choice>1</choice>", num_categories=3
    )
    assert result == [1]

    # Valid boundary cases
    result = validate_classification_response("<choice>3</choice>", num_categories=3)
    assert result == [3]  # Max valid value

    # Invalid out of range (too high)
    with pytest.raises(ValidationError, match="Invalid category indices"):
        validate_classification_response("<choice>4</choice>", num_categories=3)

    # Invalid out of range (too low)
    with pytest.raises(ValidationError, match="Invalid category indices"):
        validate_classification_response("<choice>0</choice>", num_categories=3)

    # Invalid out of range (negative)
    with pytest.raises(ValidationError, match="Invalid category indices"):
        validate_classification_response("<choice>-1</choice>", num_categories=3)


def test_validate_classification_response_multi():
    """Test validation of multiple category responses."""
    # Valid multiple selection
    result = validate_classification_response(
        "<choices>1,2</choices>", num_categories=3, max_options=2
    )
    assert result == [1, 2]

    # Valid with single tag but multiple values
    result = validate_classification_response(
        "<choice>1,3</choice>", num_categories=3, max_options=2
    )
    assert result == [1, 3]

    # Valid with spaces in selection
    result = validate_classification_response(
        "<choices>1, 3</choices>", num_categories=3, max_options=2
    )
    assert result == [1, 3]

    # Too many selections
    with pytest.raises(ValidationError, match="Too many categories selected"):
        validate_classification_response(
            "<choices>1,2,3</choices>", num_categories=3, max_options=2
        )

    # Invalid mixed range
    with pytest.raises(ValidationError, match="Invalid category indices"):
        validate_classification_response(
            "<choices>1,4</choices>", num_categories=3, max_options=2
        )


def test_validation_error_handling():
    """Test various error scenarios in validation."""
    # No XML tags
    with pytest.raises(ValidationError, match="No valid XML tags found in response"):
        validate_classification_response("This is category 1", num_categories=3)

    # Non-integer content
    with pytest.raises(ValidationError, match="Failed to parse response"):
        validate_classification_response("<choice>one</choice>", num_categories=3)

    # Empty selection
    with pytest.raises(
        ValidationError, match="Failed to parse response: invalid literal for int"
    ):
        validate_classification_response("<choice></choice>", num_categories=3)

    # Malformed XML
    with pytest.raises(ValidationError, match="No valid XML tags found in response"):
        validate_classification_response(
            "<choice>1</choice", num_categories=3  # Missing closing bracket
        )
