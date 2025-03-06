#!/usr/bin/env python3
"""
Script to clean up the redundant llmhandler package.
"""

import os
import shutil
from pathlib import Path

def main():
    """Clean up the redundant llmhandler package."""
    # Check if src/llmhandler exists
    llmhandler_path = Path("src/llmhandler")
    if not llmhandler_path.exists():
        print("llmhandler directory not found. Nothing to clean up.")
        return
    
    # Rename egg-info directories
    llmhandler_egg_info = Path("src/llmhandler.egg-info")
    if llmhandler_egg_info.exists():
        print(f"Removing {llmhandler_egg_info}")
        shutil.rmtree(llmhandler_egg_info)
    
    # Remove the old llmhandler package
    print(f"Removing {llmhandler_path}")
    shutil.rmtree(llmhandler_path)
    
    print("Cleanup complete!")

if __name__ == "__main__":
    main()