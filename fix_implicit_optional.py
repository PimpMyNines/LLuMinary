#!/usr/bin/env python3
"""
Fix implicit Optional parameters in the codebase.
"""

import re
from pathlib import Path
from typing import List

def fix_function_signature(content: str, function_pattern: str, param_name: str, param_type: str) -> str:
    """Fix a function signature to use Optional type for None default values."""
    # Find the function with the specified parameter
    pattern = re.compile(function_pattern)
    
    # Use a function to process each match
    def replace_func(match):
        full_match = match.group(0)
        
        # Replace the parameter type with Optional
        if f"{param_name}: {param_type} = None" in full_match:
            fixed = full_match.replace(
                f"{param_name}: {param_type} = None", 
                f"{param_name}: Optional[{param_type}] = None"
            )
            return fixed
        
        return full_match
    
    # Apply the replacement to the content
    updated_content = pattern.sub(replace_func, content)
    
    # Add Optional import if it's not already there
    if "Optional" not in content.split("\n")[0:10]:
        import_line = "from typing import Optional, "
        if "from typing import " in content:
            # Replace the existing import
            updated_content = re.sub(
                r"from typing import ([^O\n]*?),?\s",
                rf"from typing import \1, Optional, ",
                updated_content
            )
        else:
            # Add a new import line after the existing imports
            updated_content = re.sub(
                r"(import [^\n]+\n)",
                r"\1from typing import Optional\n",
                updated_content,
                count=1
            )
    
    return updated_content

def fix_file(file_path: Path, patterns: List[dict]) -> bool:
    """Fix a file by applying the specified patterns."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply each pattern
        for pattern in patterns:
            content = fix_function_signature(
                content, 
                pattern['function_pattern'], 
                pattern['param_name'], 
                pattern['param_type']
            )
        
        # Only write the file if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Fix implicit Optional parameters in the codebase."""
    # Define the patterns to fix
    patterns_by_file = {
        'src/lluminary/utils/aws.py': [
            {
                'function_pattern': r'def validate_aws_sts.*?\):',
                'param_name': 'required_keys',
                'param_type': 'list[str]'
            },
        ],
        'src/lluminary/models/utils/aws.py': [
            {
                'function_pattern': r'def validate_aws_sts.*?\):',
                'param_name': 'required_keys',
                'param_type': 'list[str]'
            },
        ],
        'src/lluminary/utils/validators.py': [
            {
                'function_pattern': r'def validate_dict.*?\):',
                'param_name': 'required_keys',
                'param_type': 'set[str]'
            },
            {
                'function_pattern': r'def validate_dict.*?\):',
                'param_name': 'optional_keys',
                'param_type': 'set[str]'
            },
        ],
        'src/lluminary/tools/registry.py': [
            {
                'function_pattern': r'def register_tool.*?\):',
                'param_name': 'description',
                'param_type': 'str'
            },
            {
                'function_pattern': r'def _validate_tool_function.*?\):',
                'param_name': 'name',
                'param_type': 'str'
            },
            {
                'function_pattern': r'def _validate_tool_function.*?\):',
                'param_name': 'description',
                'param_type': 'str'
            },
        ],
    }
    
    # Fix each file
    fixed_files = 0
    for file_path, patterns in patterns_by_file.items():
        if fix_file(Path(file_path), patterns):
            fixed_files += 1
            print(f"Fixed {file_path}")
    
    print(f"Fixed {fixed_files} files")

if __name__ == "__main__":
    main()