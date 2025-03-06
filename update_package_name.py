#!/usr/bin/env python3
"""
Script to update package references from lluminary to lluminary in test files.
"""
import os
import re
import sys
from pathlib import Path


def update_file(file_path):
    """Update references in a single file."""
    with open(file_path) as f:
        content = f.read()

    # Make a copy of original content for comparison
    original_content = content

    # Replace imports and references with various patterns
    patterns = [
        (r"from src\.lluminary", "from src.lluminary"),
        (r"import src\.lluminary", "import src.lluminary"),
        (r"src\.lluminary\.", "src.lluminary."),
        (r'"lluminary\.', '"lluminary.'),
        (r"'lluminary\.", "'lluminary."),
        (r'@patch\("lluminary\.', '@patch("lluminary.'),
        (r"@patch\('lluminary\.", "@patch('lluminary."),
        (r'mock\("lluminary\.', 'mock("lluminary.'),
        (r"mock\('lluminary\.", "mock('lluminary."),
        (r'"src.lluminary', '"src.lluminary'),
        (r"'src.lluminary", "'src.lluminary"),
        (r'MagicMock\(name="lluminary', 'MagicMock(name="lluminary'),
        (r"MagicMock\(name='lluminary", "MagicMock(name='lluminary"),
        (r'module="lluminary', 'module="lluminary'),
        (r"module='lluminary", "module='lluminary"),
        (r'"lluminary"', '"lluminary"'),
        (r"'lluminary'", "'lluminary'"),
        (r"\.lluminary\.", ".lluminary."),
        (r"/lluminary/", "/lluminary/"),
        (r"from lluminary", "from lluminary"),
        (r"import lluminary", "import lluminary"),
        (
            r"\bllmhandler\b",
            "lluminary",
        ),  # Word boundary for exact package name matches
    ]

    # Apply all patterns to the content
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    # Write the updated content back to the file if changed
    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        return True
    return False


def main():
    """Update all test files in the project."""
    tests_dir = Path("tests")

    if not tests_dir.exists():
        print(f"Error: {tests_dir} directory not found.")
        sys.exit(1)

    updated_count = 0
    file_count = 0

    # Process all Python files in the tests directory
    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                file_count += 1
                if update_file(file_path):
                    updated_count += 1
                    print(f"Updated: {file_path}")

    print(f"\nProcessed {file_count} files, updated {updated_count} files.")


if __name__ == "__main__":
    main()
