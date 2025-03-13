# GitHub API Reference for LLuMinary Project

## Authentication and Basic Commands

When working with GitHub repositories and needing to access the GitHub API:

```bash
# Basic authentication pattern
curl -s -X [METHOD] -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json" [API_ENDPOINT] [DATA]

# Example: List all issues
curl -s -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json" https://api.github.com/repos/PimpMyNines/LLuMinary/issues

# Example: Get a specific issue
curl -s -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json" https://api.github.com/repos/PimpMyNines/LLuMinary/issues/60
```

## Common API Operations

### Managing Issues

```bash
# Close an issue
curl -s -X PATCH -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/PimpMyNines/LLuMinary/issues/[ISSUE_NUMBER] \
  -d '{"state":"closed"}'

# Mark as duplicate (when possible)
curl -s -X PATCH -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/PimpMyNines/LLuMinary/issues/[ISSUE_NUMBER] \
  -d '{"state":"closed", "state_reason":"duplicate"}'

# Add a comment to an issue
curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/PimpMyNines/LLuMinary/issues/[ISSUE_NUMBER]/comments \
  -d '{"body":"Your comment text here"}'
```

## Sub-Issues API (Preview Feature)

> Note: The Sub-issues API is currently in public preview for organizations.

### List Sub-Issues for a Parent Issue

```bash
# List all sub-issues for a parent issue
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/PimpMyNines/LLuMinary/issues/[PARENT_ISSUE_NUMBER]/sub_issues
```

### Add a Sub-Issue to a Parent Issue

```bash
# Add an existing issue as a sub-issue to a parent issue
curl -s -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/PimpMyNines/LLuMinary/issues/[PARENT_ISSUE_NUMBER]/sub_issues \
  -d '{"sub_issue_id":[SUB_ISSUE_NUMBER]}'
```

### Remove a Sub-Issue from a Parent Issue

```bash
# Remove a sub-issue from a parent issue
curl -s -X DELETE \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/PimpMyNines/LLuMinary/issues/[PARENT_ISSUE_NUMBER]/sub_issues/[SUB_ISSUE_NUMBER]
```

### Reprioritize a Sub-Issue

```bash
# Change the priority of a sub-issue
curl -s -X PATCH \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/PimpMyNines/LLuMinary/issues/[PARENT_ISSUE_NUMBER]/sub_issues/priority \
  -d '{"sub_issue_id":[SUB_ISSUE_NUMBER],"after_id":[ISSUE_NUMBER_TO_PLACE_AFTER]}'
```

### Workflow for Managing Tasks as Sub-Issues

To convert tasks in a parent issue into proper sub-issues:

1. Create the parent issue with task descriptions
2. For each task, create a separate issue
3. Add each issue as a sub-issue to the parent
4. Track progress through the sub-issues

Example script to create sub-issues from tasks:

```bash
#!/bin/bash
# Script to create sub-issues from tasks in a parent issue

PARENT_ISSUE=60  # Replace with your parent issue number
REPO="PimpMyNines/LLuMinary"

# Get the parent issue details
parent_data=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/$REPO/issues/$PARENT_ISSUE)

# Extract tasks from the parent issue body
# This example assumes tasks are in the format "- [ ] Task description"
tasks=$(echo "$parent_data" | jq -r '.body' | grep -E '- \[ \] ' | sed 's/- \[ \] //')

# Create sub-issues for each task
for task in $tasks; do
  # Create a new issue for the task
  new_issue=$(curl -s -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/$REPO/issues \
    -d "{\"title\":\"$task\",\"body\":\"Part of #$PARENT_ISSUE: $(echo "$parent_data" | jq -r '.title')\n\n$task\"}")

  # Get the new issue number
  new_issue_number=$(echo "$new_issue" | jq -r '.number')

  # Add as sub-issue to parent
  curl -s -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    https://api.github.com/repos/$REPO/issues/$PARENT_ISSUE/sub_issues \
    -d "{\"sub_issue_id\":$new_issue_number}"

  echo "Created sub-issue #$new_issue_number for task: $task"
done
```

## Finding Duplicate Issues

```bash
# Find titles that appear more than once (potential duplicates)
curl -s https://api.github.com/repos/PimpMyNines/LLuMinary/issues?state=all | \
  grep -E "\"title\":" | sort | uniq -c | sort -nr | grep -v "^ *1 "

# Check specific issues by number and title
curl -s https://api.github.com/repos/PimpMyNines/LLuMinary/issues?state=all | \
  grep -E "\"number\":|\"title\":|\"state\":" | grep -A 2 "\"title\": \"[SEARCH_TERM]\""

# Compare two specific issues
curl -s https://api.github.com/repos/PimpMyNines/LLuMinary/issues/[ISSUE_NUMBER_1] | \
  grep -E "\"number\":|\"title\":|\"state\":" && echo -e "\n---\n" && \
  curl -s https://api.github.com/repos/PimpMyNines/LLuMinary/issues/[ISSUE_NUMBER_2] | \
  grep -E "\"number\":|\"title\":|\"state\":"
```

## Using GitHub CLI (if available)

```bash
# Close an issue with a comment
gh issue close [ISSUE_NUMBER] --repo PimpMyNines/LLuMinary --comment "Closing as duplicate of #[ORIGINAL_ISSUE]"

# List issues
gh issue list --repo PimpMyNines/LLuMinary
```

## Best Practices for Handling Duplicates

1. Always close the newer issue (higher number)
2. Add a comment referencing the original issue
3. Use `{"state":"closed", "state_reason":"duplicate"}` when possible
4. Verify closure with a follow-up API call

## Environment Setup

Ensure your GitHub token is set in your environment:

```bash
# Set token for current session
export GITHUB_TOKEN="your_github_token"

# Or add to your .bashrc/.zshrc for persistence
echo 'export GITHUB_TOKEN="your_github_token"' >> ~/.bashrc
```

## Security Notes

- Never commit your GitHub token to version control
- Consider using GitHub CLI with authenticated sessions for better security
- Use repository-specific tokens with minimal permissions when possible
