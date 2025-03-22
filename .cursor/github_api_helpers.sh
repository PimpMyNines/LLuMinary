#!/bin/bash
# GitHub API Helper Functions for LLuMinary Project

# Check if GITHUB_TOKEN is set
check_github_token() {
  if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable is not set."
    echo "Please set it with: export GITHUB_TOKEN=your_token"
    return 1
  fi
  return 0
}

# List all issues in the repository
list_issues() {
  check_github_token || return 1

  local state=${1:-"open"}  # Default to open issues

  curl -s -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.v3+json" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues?state=$state" | \
       jq -r '.[] | "#\(.number): \(.title) (\(.state))"'
}

# Get details of a specific issue
get_issue() {
  check_github_token || return 1

  local issue_number=$1

  if [ -z "$issue_number" ]; then
    echo "Error: Issue number is required."
    echo "Usage: get_issue ISSUE_NUMBER"
    return 1
  fi

  curl -s -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.v3+json" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$issue_number" | \
       jq '.'
}

# Close an issue
close_issue() {
  check_github_token || return 1

  local issue_number=$1
  local reason=${2:-"completed"}  # Default reason is "completed"

  if [ -z "$issue_number" ]; then
    echo "Error: Issue number is required."
    echo "Usage: close_issue ISSUE_NUMBER [REASON]"
    return 1
  fi

  curl -s -X PATCH \
       -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.v3+json" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$issue_number" \
       -d "{\"state\":\"closed\", \"state_reason\":\"$reason\"}" | \
       jq -r '"\(.number): \(.title) is now \(.state) with reason: \(.state_reason)"'
}

# Add a comment to an issue
add_comment() {
  check_github_token || return 1

  local issue_number=$1
  local comment_text=$2

  if [ -z "$issue_number" ] || [ -z "$comment_text" ]; then
    echo "Error: Issue number and comment text are required."
    echo "Usage: add_comment ISSUE_NUMBER \"COMMENT_TEXT\""
    return 1
  fi

  curl -s -X POST \
       -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.v3+json" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$issue_number/comments" \
       -d "{\"body\":\"$comment_text\"}" | \
       jq -r '"\(.id): Comment added to issue #\(.issue_url | split("/") | .[-1])"'
}

# Find potential duplicate issues
find_duplicates() {
  check_github_token || return 1

  curl -s -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.v3+json" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues?state=all&per_page=100" | \
       jq -r '.[] | "\(.title)"' | sort | uniq -c | sort -nr | grep -v "^ *1 "
}

# Close an issue as a duplicate of another
close_as_duplicate() {
  check_github_token || return 1

  local duplicate_issue=$1
  local original_issue=$2

  if [ -z "$duplicate_issue" ] || [ -z "$original_issue" ]; then
    echo "Error: Both duplicate issue number and original issue number are required."
    echo "Usage: close_as_duplicate DUPLICATE_ISSUE_NUMBER ORIGINAL_ISSUE_NUMBER"
    return 1
  fi

  # Add a comment
  add_comment "$duplicate_issue" "Closing as duplicate of #$original_issue"

  # Close the issue
  close_issue "$duplicate_issue" "duplicate"

  echo "Issue #$duplicate_issue closed as duplicate of #$original_issue"
}

# Compare two issues to check for duplicates
compare_issues() {
  check_github_token || return 1

  local issue1=$1
  local issue2=$2

  if [ -z "$issue1" ] || [ -z "$issue2" ]; then
    echo "Error: Two issue numbers are required."
    echo "Usage: compare_issues ISSUE_NUMBER_1 ISSUE_NUMBER_2"
    return 1
  fi

  echo "=== Issue #$issue1 ==="
  curl -s -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.v3+json" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$issue1" | \
       jq -r '"\(.number): \(.title) (\(.state))\n\(.body)"'

  echo -e "\n=== Issue #$issue2 ==="
  curl -s -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.v3+json" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$issue2" | \
       jq -r '"\(.number): \(.title) (\(.state))\n\(.body)"'
}

# List all sub-issues for a parent issue
list_sub_issues() {
  check_github_token || return 1

  local parent_issue=$1

  if [ -z "$parent_issue" ]; then
    echo "Error: Parent issue number is required."
    echo "Usage: list_sub_issues PARENT_ISSUE_NUMBER"
    return 1
  fi

  echo "Sub-issues for parent issue #$parent_issue:"
  curl -s -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github+json" \
       -H "X-GitHub-Api-Version: 2022-11-28" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$parent_issue/sub_issues" | \
       jq -r '.[] | "#\(.number): \(.title) (\(.state))"'
}

# Add an existing issue as a sub-issue to a parent issue
add_sub_issue() {
  check_github_token || return 1

  local parent_issue=$1
  local sub_issue=$2

  if [ -z "$parent_issue" ] || [ -z "$sub_issue" ]; then
    echo "Error: Both parent issue number and sub-issue number are required."
    echo "Usage: add_sub_issue PARENT_ISSUE_NUMBER SUB_ISSUE_NUMBER"
    return 1
  fi

  curl -s -X POST \
       -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github+json" \
       -H "X-GitHub-Api-Version: 2022-11-28" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$parent_issue/sub_issues" \
       -d "{\"sub_issue_id\":$sub_issue}" | \
       jq -r '"Added issue #\(.number) as sub-issue to parent #'"$parent_issue"'"'
}

# Remove a sub-issue from a parent issue
remove_sub_issue() {
  check_github_token || return 1

  local parent_issue=$1
  local sub_issue=$2

  if [ -z "$parent_issue" ] || [ -z "$sub_issue" ]; then
    echo "Error: Both parent issue number and sub-issue number are required."
    echo "Usage: remove_sub_issue PARENT_ISSUE_NUMBER SUB_ISSUE_NUMBER"
    return 1
  fi

  curl -s -X DELETE \
       -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github+json" \
       -H "X-GitHub-Api-Version: 2022-11-28" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$parent_issue/sub_issues/$sub_issue"

  echo "Removed issue #$sub_issue as sub-issue from parent #$parent_issue"
}

# Change the priority of a sub-issue
reprioritize_sub_issue() {
  check_github_token || return 1

  local parent_issue=$1
  local sub_issue=$2
  local after_issue=$3

  if [ -z "$parent_issue" ] || [ -z "$sub_issue" ]; then
    echo "Error: Parent issue number and sub-issue number are required."
    echo "Usage: reprioritize_sub_issue PARENT_ISSUE_NUMBER SUB_ISSUE_NUMBER [AFTER_ISSUE_NUMBER]"
    return 1
  fi

  local data="{\"sub_issue_id\":$sub_issue"
  if [ -n "$after_issue" ]; then
    data="$data,\"after_id\":$after_issue"
  fi
  data="$data}"

  curl -s -X PATCH \
       -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github+json" \
       -H "X-GitHub-Api-Version: 2022-11-28" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$parent_issue/sub_issues/priority" \
       -d "$data"

  echo "Reprioritized issue #$sub_issue in parent #$parent_issue"
}

# Create sub-issues from tasks in a parent issue
create_sub_issues_from_tasks() {
  check_github_token || return 1

  local parent_issue=$1

  if [ -z "$parent_issue" ]; then
    echo "Error: Parent issue number is required."
    echo "Usage: create_sub_issues_from_tasks PARENT_ISSUE_NUMBER"
    return 1
  fi

  # Get the parent issue details
  local parent_data=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github+json" \
       "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$parent_issue")

  local parent_title=$(echo "$parent_data" | jq -r '.title')

  # Extract tasks from the parent issue body
  # This example assumes tasks are in the format "- [ ] Task description"
  local tasks=$(echo "$parent_data" | jq -r '.body' | grep -E '- \[ \] ' | sed 's/- \[ \] //')

  if [ -z "$tasks" ]; then
    echo "No tasks found in issue #$parent_issue. Tasks should be in the format '- [ ] Task description'"
    return 1
  fi

  echo "Creating sub-issues for tasks in parent issue #$parent_issue: $parent_title"

  # Create sub-issues for each task
  echo "$tasks" | while read -r task; do
    # Create a new issue for the task
    local new_issue=$(curl -s -X POST \
         -H "Authorization: token $GITHUB_TOKEN" \
         -H "Accept: application/vnd.github+json" \
         "https://api.github.com/repos/PimpMyNines/LLuMinary/issues" \
         -d "{\"title\":\"$task\",\"body\":\"Part of #$parent_issue: $parent_title\n\n$task\"}")

    # Get the new issue number
    local new_issue_number=$(echo "$new_issue" | jq -r '.number')

    # Add as sub-issue to parent
    curl -s -X POST \
         -H "Authorization: token $GITHUB_TOKEN" \
         -H "Accept: application/vnd.github+json" \
         -H "X-GitHub-Api-Version: 2022-11-28" \
         "https://api.github.com/repos/PimpMyNines/LLuMinary/issues/$parent_issue/sub_issues" \
         -d "{\"sub_issue_id\":$new_issue_number}" > /dev/null

    echo "Created sub-issue #$new_issue_number for task: $task"
  done
}

echo "GitHub API helper functions loaded. Available commands:"
echo "  - list_issues [state]"
echo "  - get_issue ISSUE_NUMBER"
echo "  - close_issue ISSUE_NUMBER [REASON]"
echo "  - add_comment ISSUE_NUMBER \"COMMENT_TEXT\""
echo "  - find_duplicates"
echo "  - close_as_duplicate DUPLICATE_ISSUE_NUMBER ORIGINAL_ISSUE_NUMBER"
echo "  - compare_issues ISSUE_NUMBER_1 ISSUE_NUMBER_2"
echo "  - list_sub_issues PARENT_ISSUE_NUMBER"
echo "  - add_sub_issue PARENT_ISSUE_NUMBER SUB_ISSUE_NUMBER"
echo "  - remove_sub_issue PARENT_ISSUE_NUMBER SUB_ISSUE_NUMBER"
echo "  - reprioritize_sub_issue PARENT_ISSUE_NUMBER SUB_ISSUE_NUMBER [AFTER_ISSUE_NUMBER]"
echo "  - create_sub_issues_from_tasks PARENT_ISSUE_NUMBER"
echo ""
echo "Usage example: source .cursor/github_api_helpers.sh && find_duplicates"
