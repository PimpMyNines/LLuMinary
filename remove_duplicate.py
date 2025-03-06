#!/usr/bin/env python3
"""
Remove duplicate generate method from base.py
"""

with open('src/lluminary/models/base.py', 'r') as f:
    content = f.read()

# Find the second generate method
start_line = "    def generate("
second_start = content.find(start_line, content.find(start_line) + 1)

if second_start != -1:
    # Find a suitable end point (the next method)
    end = content.find("    def _calculate_cost_estimate", second_start)
    
    if end != -1:
        # Replace the duplicate method with a comment
        new_content = content[:second_start] + "    # Second generate method removed to avoid duplication\n\n" + content[end:]
        
        with open('src/lluminary/models/base.py', 'w') as f:
            f.write(new_content)
        
        print('Successfully removed duplicate generate method')
    else:
        print('Could not find end of duplicate generate method')
else:
    print('Could not find duplicate generate method')