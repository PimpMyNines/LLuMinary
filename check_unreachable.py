#!/usr/bin/env python3
"""
Script to check for unreachable code in the base LLM class.
"""

import ast
import sys
from typing import List, Tuple


def find_unreachable_code(file_path: str) -> List[Tuple[int, str]]:
    """
    Find unreachable code in a Python file.

    Args:
        file_path: Path to the Python file

    Returns:
        List of tuples containing line number and unreachable code
    """
    with open(file_path) as f:
        source = f.read()

    tree = ast.parse(source)

    unreachable_code = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check for unreachable code after return statements
            returns = []
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    returns.append(child.lineno)

            # Check for code after return statements
            for i, stmt in enumerate(node.body):
                if i > 0 and isinstance(node.body[i - 1], ast.Return):
                    unreachable_code.append(
                        (
                            stmt.lineno,
                            f"Unreachable code after return in function {node.name}",
                        )
                    )

    return unreachable_code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    unreachable_code = find_unreachable_code(file_path)

    if unreachable_code:
        print(f"Found {len(unreachable_code)} instances of unreachable code:")
        for line, message in unreachable_code:
            print(f"Line {line}: {message}")
    else:
        print("No unreachable code found.")
