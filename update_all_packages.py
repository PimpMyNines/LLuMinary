#!/usr/bin/env python3
"""
Script to update package references from lluminary to lluminary in all project files.
"""
import os
import re
from pathlib import Path


def update_file(file_path):
    """Update references in a single file."""
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        try:
            content = f.read()
        except UnicodeDecodeError:
            print(
                f"Warning: Could not read {file_path} due to encoding issues. Skipping."
            )
            return False

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
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def main():
    """Update all Python files in the project."""
    project_dir = Path(".")
    excluded_dirs = [".git", ".venv", "venv", "build", "dist", "__pycache__"]

    updated_count = 0
    file_count = 0

    # Process all Python files in the project
    for root, dirs, files in os.walk(project_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

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
