#!/usr/bin/env python3
"""
Fix missing return statement in the models/base.py generate method.
"""

import re
from pathlib import Path

def fix_base_lm_generate():
    """Fix the missing return statement in LLM.generate method."""
    file_path = Path('src/lluminary/models/base.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the end of the generate method (look for the next method or class)
    generate_method_pattern = r'def generate\([^)]*\)[^:]*:\s*""".*?""".*?cumulative_usage = \{[^}]*\}'
    
    # Find all occurrences
    match = re.search(generate_method_pattern, content, re.DOTALL)
    
    if match:
        generate_method = match.group(0)
        
        # Add the following content after the method definition
        function_body = """
        # Validate functions if provided
        functions = None  # Default for tools parameter
        thinking_budget = None

        # If tools is provided, validate it
        if tools is not None:
            if not isinstance(tools, list):
                raise LLMValidationError(
                    "tools must be a list",
                    details={"provided_type": type(tools).__name__},
                )
            validate_tools(tools)

        # Generation with retry loop
        while attempts < retry_limit:  # Includes first attempt
            try:
                # Generate raw response
                raw_response, usage = self._raw_generate(
                    event_id=f"{event_id}_attempt_{attempts}",
                    system_prompt=system_prompt,
                    messages=working_messages,
                    max_tokens=max_tokens,
                    temp=temp,
                    tools=tools,
                    thinking_budget=thinking_budget,
                )

                # Update usage statistics
                for key in usage:
                    if key in cumulative_usage:
                        cumulative_usage[key] += usage[key]

                # Process and validate response
                if result_processing_function:
                    try:
                        response = result_processing_function(raw_response)
                    except Exception as proc_error:
                        # Convert processing errors to LLMMistake for retry
                        from ..exceptions import LLMFormatError

                        raise LLMFormatError(
                            f"Response processing failed: {proc_error!s}",
                            provider=self.__class__.__name__.replace("LLM", ""),
                            details={"raw_response": raw_response},
                        )
                else:
                    response = raw_response

                # Prepare response message
                ai_message = {"message_type": "ai", "message": response}

                # Create updated messages list
                updated_messages = working_messages + [ai_message]

                return response, cumulative_usage, updated_messages

            except LLMMistake as e:
                attempts += 1
                cumulative_usage["retry_count"] = attempts

                if attempts >= retry_limit:
                    raise LLMMistake(
                        f"Failed to get valid response after {retry_limit} attempts",
                        error_type="retry_limit_exceeded",
                        provider=self.__class__.__name__.replace("LLM", ""),
                        details={"retry_limit": retry_limit, "last_error": str(e)},
                    )

                # Add failed response to working messages
                working_messages.append(
                    {
                        "message_type": "human",
                        "message": f"Error in previous response: {e!s}. Please try again.",
                    }
                )
        
        # This should never be reached due to the retry loop
        raise Exception("Unexpected end of generation method")
        """
        
        # Replace the generate method with the complete version
        updated_content = content.replace(generate_method, generate_method + function_body)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"Fixed generate method in {file_path}")
        return True
    else:
        print(f"Could not find generate method in {file_path}")
        return False

def remove_duplicate_generate():
    """Remove the duplicate generate function in the LLM class."""
    file_path = Path('src/lluminary/models/base.py')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the duplicate generate method
    duplicate_pattern = r'def generate\(\s*self,\s*event_id: str,.*?retry_limit: int = 3,(?:.|\n)*?return response, cumulative_usage, updated_messages'
    
    # Find all occurrences
    match = re.search(duplicate_pattern, content, re.DOTALL)
    
    if match:
        duplicate_method = match.group(0)
        
        # Remove the duplicate method (and trailing characters)
        updated_content = content.replace(duplicate_method, "# Duplicate generate method removed")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"Removed duplicate generate method in {file_path}")
        return True
    else:
        print(f"Could not find duplicate generate method in {file_path}")
        return False

if __name__ == "__main__":
    fix_base_lm_generate()
    remove_duplicate_generate()