# DOCUMENTATION STANDARDS

## Overview

This document defines the standards for all markdown documentation in the LLuMinary project to ensure consistency across all documentation files.

## File Naming Standards

- Use UPPERCASE_WITH_UNDERSCORES for all markdown documentation files
  - Example: `ERROR_HANDLING.md`, `API_REFERENCE.md`, `OPENAI_PROVIDER.md`
- Use `.md` extension for all markdown files
- Use descriptive names that clearly indicate the content of the file

## Markdown Formatting

### Headers

- Use ATX-style headers with # symbols
- Include a single space after the # symbol
- Use a single blank line after each header
- Use Title Case for all headers (capitalize each word except articles, conjunctions, and prepositions)

Example:
```markdown
# Main Title

## Section Title

### Subsection Title
```

### Lists

- Use - for unordered lists with a space after the dash
- Use 1. for ordered lists with a space after the period
- Indent nested lists with 2 spaces

Example:
```markdown
- Item 1
  - Nested item 1.1
  - Nested item 1.2
- Item 2

1. First step
2. Second step
   1. Substep 2.1
   2. Substep 2.2
```

### Code Blocks

- Always specify the language for syntax highlighting
- Use triple backticks (```) to define code blocks

Example:
```markdown
```python
def example_function():
    return "Hello, world!"
```
```

### Tables

- Use standard markdown table syntax
- Include a header row and separator row
- Align content for readability

Example:
```markdown
| Name | Type | Description |
|------|------|-------------|
| id | string | Unique identifier |
| created | timestamp | Creation timestamp |
```

### Links

- Use descriptive link text
- Use relative links for internal documentation

Example:
```markdown
See the [API Reference](./API_REFERENCE.md) for more details.
```

### Emphasis

- Use *italics* for emphasis or introducing new terms
- Use **bold** for strong emphasis or UI elements
- Use `code` for code references, file names, and commands

## Document Structure

### Required Sections

All documentation files should include:

1. **Title** - Top-level header with the document title
2. **Overview** - Brief description of the document's purpose
3. **Table of Contents** - For documents longer than 3 sections (optional for shorter documents)
4. **Content Sections** - Organized by logical sections with consistent heading levels
5. **Related Documentation** - Links to related documents (if applicable)

### Example Structure

```markdown
# DOCUMENT TITLE

## Overview

Brief description of the document's purpose and scope.

## Table of Contents

- [Section 1](#section-1)
- [Section 2](#section-2)
  - [Subsection 2.1](#subsection-21)
- [Section 3](#section-3)
- [Related Documentation](#related-documentation)

## Section 1

Content for section 1.

## Section 2

Content for section 2.

### Subsection 2.1

Content for subsection 2.1.

## Section 3

Content for section 3.

## Related Documentation

- [Document 1](./DOCUMENT_1.md)
- [Document 2](./DOCUMENT_2.md)
```

## Special Elements

### Status Indicators

- Use consistent symbols for status indicators:
  - ✅ - Complete/Done
  - 🟡 - In Progress/Medium Priority
  - 🔴 - Not Started/High Priority
  - ❌ - Blocked/Issue

### Checkboxes

- Use GitHub-style checkboxes for task lists:
  - `- [ ]` for incomplete tasks
  - `- [x]` for completed tasks

Example:
```markdown
- [x] Completed task
- [ ] Incomplete task
```

### Admonitions

Use a consistent format for notes, warnings, and other admonitions:

```markdown
> **NOTE**: This is an important note.

> **WARNING**: This is a warning message.
```

## Implementation Plan

1. Create this standards document
2. Rename all markdown files to follow the naming convention
3. Update existing documents to follow the formatting standards
4. Add this document to the development guidelines
