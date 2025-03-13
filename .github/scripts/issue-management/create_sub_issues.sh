#!/bin/bash
# Script to create sub-issues from tasks in a parent issue

# Check if GITHUB_TOKEN is set
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN environment variable is not set."
  echo "Please set it with: export GITHUB_TOKEN='your_github_token'"
  exit 1
fi

# Get repository information from git config
REPO_OWNER=$(git config --get remote.origin.url | sed -n 's/.*github.com[:/]\([^/]*\)\/\([^.]*\).*/\1/p')
REPO_NAME=$(git config --get remote.origin.url | sed -n 's/.*github.com[:/]\([^/]*\)\/\([^.]*\).*/\2/p')

# If we couldn't get the repo info from git, use default
if [ -z "$REPO_OWNER" ] || [ -z "$REPO_NAME" ]; then
  echo "Warning: Could not determine repository from git config."
  echo "Using default repository: PimpMyNines/LLuMinary"
  REPO="PimpMyNines/LLuMinary"
else
  REPO="$REPO_OWNER/$REPO_NAME"
  echo "Using repository: $REPO"
fi

# Function to create sub-issues for a parent issue
create_sub_issues() {
  local parent_issue_number=$1

  echo "Processing issue #$parent_issue_number..."

  # Get the parent issue details
  parent_data=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues/$parent_issue_number")

  # Check if issue exists
  if [ "$(echo "$parent_data" | jq -r '.message // empty')" = "Not Found" ]; then
    echo "Error: Issue #$parent_issue_number not found"
    return 1
  fi

  # Extract issue title and body
  parent_title=$(echo "$parent_data" | jq -r '.title')
  parent_body=$(echo "$parent_data" | jq -r '.body')
  parent_labels=$(echo "$parent_data" | jq -c '.labels | map(.name)')

  echo "Parent issue: $parent_title"

  # Extract tasks from the parent issue body
  # This example assumes tasks are in the format "- [ ] Task description"
  tasks=$(echo "$parent_body" | grep -e '- \[ \] ' | sed 's/- \[ \] //')

  # Check if tasks were found
  if [ -z "$tasks" ]; then
    echo "No tasks found in issue #$parent_issue_number"
    return 0
  fi

  # Count tasks
  task_count=$(echo "$tasks" | wc -l)
  echo "Found $task_count tasks in issue #$parent_issue_number"

  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would create $task_count sub-issues for issue #$parent_issue_number"
    return 0
  fi

  # Create a temporary file to store sub-issue numbers
  sub_numbers_file=$(mktemp)

  # Create sub-issues for each task
  echo "$tasks" | while read -r task; do
    # Skip empty tasks
    if [ -z "$task" ]; then
      continue
    fi

    echo "Creating sub-issue for task: $task"

    # Create a new issue for the task
    sub_issue_data=$(curl -s -X POST \
      -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues" \
      -d "{\"title\":\"$task\",\"body\":\"Part of #$parent_issue_number: $parent_title\n\n$task\",\"labels\":$parent_labels}")

    # Get the new issue number
    sub_issue_number=$(echo "$sub_issue_data" | jq -r '.number')

    if [ "$sub_issue_number" = "null" ]; then
      echo "Error creating sub-issue: $(echo "$sub_issue_data" | jq -r '.message')"
      continue
    fi

    echo "Created sub-issue #$sub_issue_number"

    # Add to temporary file with parent number for reference
    echo "$sub_issue_number:$parent_issue_number" >> "$sub_numbers_file"
  done

  # Wait a moment to ensure all issues are created
  sleep 2

  # Now try to add all sub-issues using the sub-issues API
  echo "Attempting to link sub-issues to parent #$parent_issue_number using the Sub-issues API..."

  # First, check if the Sub-issues API is available by trying to list existing sub-issues
  api_check=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues/$parent_issue_number/sub_issues")

  if [ "$api_check" = "404" ] || [ "$api_check" = "403" ]; then
    echo "Warning: Sub-issues API is not available (HTTP $api_check). This feature requires:"
    echo "  - GitHub organization account"
    echo "  - Sub-issues feature to be enabled for the organization"
    echo "  - Token with appropriate permissions"
    echo "Using body references instead. Sub-issues will appear as regular issues with 'Part of #' references."
  else
    echo "Sub-issues API is available. Linking sub-issues..."

    # Check if the file exists and has content
    if [ -s "$sub_numbers_file" ]; then
      # Link each sub-issue to the parent
      while IFS=: read -r sub_number parent_num; do
        echo "Linking sub-issue #$sub_number to parent #$parent_num"

        sub_api_response=$(curl -s -X POST \
          -H "Authorization: token $GITHUB_TOKEN" \
          -H "Accept: application/vnd.github+json" \
          -H "X-GitHub-Api-Version: 2022-11-28" \
          "https://api.github.com/repos/$REPO/issues/$parent_num/sub_issues" \
          -d "{\"sub_issue_id\":$sub_number}")

        # Check for errors
        error_message=$(echo "$sub_api_response" | jq -r '.message // empty')
        if [ ! -z "$error_message" ]; then
          echo "Error linking sub-issue: $error_message"
        else
          echo "Successfully linked sub-issue #$sub_number to parent #$parent_num"
        fi
      done < "$sub_numbers_file"
    else
      echo "No sub-issues were created, skipping linking step."
    fi
  fi

  # Clean up temporary file
  rm -f "$sub_numbers_file"

  echo "Completed processing issue #$parent_issue_number"
  return 0
}

# Check if an issue number was provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <issue_number> [issue_number2 ...]"
  echo "Example: $0 86 85 62"
  exit 1
fi

# Process each provided issue number
for issue in "$@"; do
  create_sub_issues "$issue"
done

echo "Done!"
