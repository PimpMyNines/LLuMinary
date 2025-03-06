#!/usr/bin/env python3
"""
Script to remove sys.path manipulations from test files.
"""

import os
import re
import sys
from pathlib import Path

def fix_file(file_path):
    """Remove sys.path manipulations from the file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for common sys.path manipulation patterns
    patterns = [
        r'sys\.path\.insert\(0,.*?\)\n',
        r'sys\.path\.append\(.*?\)\n',
        r'# Add the package to path for testing\n',
        r'# Path manipulation no longer needed\n',
    ]
    
    original_content = content
    
    for pattern in patterns:
        content = re.sub(pattern, '', content)
    
    # Only write file if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix sys.path manipulation across test files."""
    roots = ["tests"]
    total_files = 0
    
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"Directory {root} does not exist, skipping")
            continue
            
        for file_path in root_path.glob('**/*.py'):
            fixed = fix_file(file_path)
            if fixed:
                total_files += 1
                print(f"Fixed sys.path manipulation in {file_path}")
    
    print(f"\nSummary: Fixed sys.path manipulation in {total_files} files")

if __name__ == "__main__":
    main()