#!/bin/bash
# Batch processing script for LLuMinary GitHub Issue Management Tools

# Check if GITHUB_TOKEN is set
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN environment variable is not set."
  echo "Please run './issue-tools.sh setup-token' first."
  exit 1
fi

# Get repository information from git config
REPO_OWNER=$(git config --get remote.origin.url | sed -n 's/.*github.com[:/]\([^/]*\)\/\([^.]*\).*/\1/p')
REPO_NAME=$(git config --get remote.origin.url | sed -n 's/.*github.com[:/]\([^/]*\)\/\([^.]*\).*/\2/p')

# If we couldn't get the repo info from git, use default
if [ -z "$REPO_OWNER" ] || [ -z "$REPO_NAME" ]; then
  REPO="PimpMyNines/LLuMinary"
else
  REPO="$REPO_OWNER/$REPO_NAME"
fi

# Display help if no arguments provided
if [ $# -eq 0 ]; then
    echo "LLuMinary GitHub Issue Management Batch Processing"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -q, --query <query>     Search query to find issues (e.g., 'enhancement')"
    echo "  -l, --limit <number>    Maximum number of issues to process (default: 10)"
    echo "  -d, --dry-run           Show what would be done without making changes"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -q 'enhancement' -l 5"
    echo "  $0 -q 'bug' --dry-run"
    echo ""
    exit 0
fi

# Default values
QUERY=""
LIMIT=10
DRY_RUN=${DRY_RUN:-false}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -q|--query)
      QUERY="$2"
      shift 2
      ;;
    -l|--limit)
      LIMIT="$2"
      shift 2
      ;;
    -d|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      echo "LLuMinary GitHub Issue Management Batch Processing"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  -q, --query <query>     Search query to find issues (e.g., 'enhancement')"
      echo "  -l, --limit <number>    Maximum number of issues to process (default: 10)"
      echo "  -d, --dry-run           Show what would be done without making changes"
      echo "  -h, --help              Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 -q 'enhancement' -l 5"
      echo "  $0 -q 'bug' --dry-run"
      echo ""
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if query is provided
if [ -z "$QUERY" ]; then
  echo "Error: No search query provided. Use -q or --query to specify a search query."
  exit 1
fi

echo "Repository: $REPO"
echo "Search query: $QUERY"
echo "Limit: $LIMIT"
if [ "$DRY_RUN" = true ]; then
  echo "Dry run: Yes (no changes will be made)"
else
  echo "Dry run: No (changes will be applied)"
fi

# Search for issues matching the query
echo "Searching for issues matching query: $QUERY"
search_results=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/search/issues?q=repo:$REPO+$QUERY+is:issue+is:open&per_page=$LIMIT")

# Check if the search was successful
if [ "$(echo "$search_results" | jq -r '.message // empty')" != "" ]; then
  echo "Error: $(echo "$search_results" | jq -r '.message')"
  exit 1
fi

# Get the total count of issues found
total_count=$(echo "$search_results" | jq -r '.total_count')
echo "Found $total_count issues matching the query"

# Get the issues from the search results
issues=$(echo "$search_results" | jq -r '.items[] | .number')

# Check if any issues were found
if [ -z "$issues" ]; then
  echo "No issues found matching the query."
  exit 0
fi

# Count the number of issues to process
issue_count=$(echo "$issues" | wc -l | tr -d ' ')
echo "Processing $issue_count issues (limited to $LIMIT)"

# Process each issue
for issue_number in $issues; do
  echo "Processing issue #$issue_number..."

  # Get the issue details
  issue_data=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues/$issue_number")

  # Extract issue title and body
  issue_title=$(echo "$issue_data" | jq -r '.title')
  issue_body=$(echo "$issue_data" | jq -r '.body')

  echo "Issue: $issue_title"

  # Extract tasks from the issue body
  tasks=$(echo "$issue_body" | grep -e '- \[ \] ' | sed 's/- \[ \] //')

  # Check if tasks were found
  if [ -z "$tasks" ]; then
    echo "No tasks found in issue #$issue_number, skipping..."
    continue
  fi

  # Count tasks
  task_count=$(echo "$tasks" | wc -l)
  echo "Found $task_count tasks in issue #$issue_number"

  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would create $task_count sub-issues for issue #$issue_number"
    continue
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

    # Get labels from parent issue
    parent_labels=$(echo "$issue_data" | jq -c '.labels | map(.name)')

    # Create a new issue for the task
    sub_issue_data=$(curl -s -X POST \
      -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues" \
      -d "{\"title\":\"$task\",\"body\":\"Part of #$issue_number: $issue_title\n\n$task\",\"labels\":$parent_labels}")

    # Get the new issue number
    sub_issue_number=$(echo "$sub_issue_data" | jq -r '.number')

    if [ "$sub_issue_number" = "null" ]; then
      echo "Error creating sub-issue: $(echo "$sub_issue_data" | jq -r '.message')"
      continue
    fi

    echo "Created sub-issue #$sub_issue_number"

    # Add to temporary file
    echo "$sub_issue_number:$issue_number" >> "$sub_numbers_file"
  done

  # Wait a moment to ensure all issues are created
  sleep 2

  # Now try to add all sub-issues using the sub-issues API
  echo "Attempting to link sub-issues to parent #$issue_number using the Sub-issues API..."

  # First, check if the Sub-issues API is available by trying to list existing sub-issues
  api_check=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues/$issue_number/sub_issues")

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

  echo "Completed processing issue #$issue_number"
done

echo "Batch processing complete!"
