#!/usr/bin/env python3
"""
Script to fix imports from src.lluminary to lluminary in all Python files.
"""

import os
import re
import sys
from pathlib import Path

def fix_imports_in_file(file_path):
    """Replace src.lluminary imports with lluminary imports."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count original occurrences
    original_count = content.count('src.lluminary')
    
    # Replace imports
    updated_content = content.replace('from src.lluminary', 'from lluminary')
    updated_content = updated_content.replace('import src.lluminary', 'import lluminary')
    
    # Only write file if changes were made
    if content != updated_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        return original_count
    return 0

def main():
    """Fix imports across the project."""
    roots = ["tests", "debug", "examples", "docs"]
    total_files = 0
    total_fixes = 0
    
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"Directory {root} does not exist, skipping")
            continue
            
        for file_path in root_path.glob('**/*.py'):
            fixes = fix_imports_in_file(file_path)
            if fixes > 0:
                total_files += 1
                total_fixes += fixes
                print(f"Fixed {fixes} imports in {file_path}")
    
    print(f"\nSummary: Fixed {total_fixes} imports across {total_files} files")

if __name__ == "__main__":
    main()