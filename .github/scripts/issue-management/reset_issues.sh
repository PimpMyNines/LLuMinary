#!/bin/bash
# Script to delete all existing issues and start fresh

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

# Default values
DRY_RUN=false
BACKUP_FILE="issues_backup_$(date +%Y%m%d_%H%M%S).json"
ISSUE_STRUCTURE_FILE=""
SKIP_CONFIRMATION=false

# Function to display help message
show_help() {
  echo "Usage: $0 [options]"
  echo "Delete all existing issues and start fresh with correctly created issues and subtasks."
  echo ""
  echo "Options:"
  echo "  -d, --dry-run              Show what would be done without making changes"
  echo "  -b, --backup <file>        Backup file to save existing issues (default: $BACKUP_FILE)"
  echo "  -s, --structure <file>     JSON file with new issue structure to create"
  echo "  -y, --yes                  Skip confirmation prompt"
  echo "  -h, --help                 Display this help message"
  echo ""
  echo "Examples:"
  echo "  $0 -d                      Dry run to see what would be deleted"
  echo "  $0 -b my_backup.json       Backup issues to my_backup.json before deleting"
  echo "  $0 -s new_structure.json   Create new issues from structure in new_structure.json"
  echo "  $0 -y                      Skip confirmation prompt and proceed with deletion"
  exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -d|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -b|--backup)
      BACKUP_FILE="$2"
      shift
      shift
      ;;
    -s|--structure)
      ISSUE_STRUCTURE_FILE="$2"
      shift
      shift
      ;;
    -y|--yes)
      SKIP_CONFIRMATION=true
      shift
      ;;
    -h|--help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      ;;
  esac
done

# Function to backup all issues
backup_issues() {
  echo "Backing up all issues to $BACKUP_FILE..."

  # Get all open issues
  open_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues?state=open&per_page=100")

  # Get all closed issues
  closed_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues?state=closed&per_page=100")

  # Combine open and closed issues
  all_issues=$(echo '[]' | jq --argjson open "$open_issues" --argjson closed "$closed_issues" '. + $open + $closed')

  # Save to backup file
  echo "$all_issues" > "$BACKUP_FILE"

  # Count issues
  issue_count=$(echo "$all_issues" | jq 'length')
  echo "Backed up $issue_count issues to $BACKUP_FILE"
}

# Function to close all issues
close_all_issues() {
  echo "Closing all issues..."

  # Get all open issues
  open_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues?state=open&per_page=100")

  # Count issues
  issue_count=$(echo "$open_issues" | jq 'length')
  echo "Found $issue_count open issues"

  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would close $issue_count issues"
    return
  fi

  # Close each issue
  for i in $(seq 0 $(($issue_count - 1))); do
    issue_number=$(echo "$open_issues" | jq -r ".[$i].number")
    issue_title=$(echo "$open_issues" | jq -r ".[$i].title")

    echo "Closing issue #$issue_number: $issue_title"

    curl -s -X PATCH -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues/$issue_number" \
      -d '{"state":"closed", "state_reason":"not_planned"}'
  done

  echo "Closed $issue_count issues"
}

# Function to create new issues from structure file
create_new_issues() {
  if [ -z "$ISSUE_STRUCTURE_FILE" ]; then
    echo "No issue structure file provided. Skipping issue creation."
    return
  fi

  if [ ! -f "$ISSUE_STRUCTURE_FILE" ]; then
    echo "Error: Issue structure file '$ISSUE_STRUCTURE_FILE' not found."
    exit 1
  fi

  echo "Creating new issues from $ISSUE_STRUCTURE_FILE..."

  # Read issue structure file
  issue_structure=$(cat "$ISSUE_STRUCTURE_FILE")

  # Count parent issues
  parent_count=$(echo "$issue_structure" | jq '.parent_issues | length')
  echo "Found $parent_count parent issues to create"

  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would create $parent_count parent issues and their sub-issues"
    return
  fi

  # Create each parent issue and its sub-issues
  for i in $(seq 0 $(($parent_count - 1))); do
    parent=$(echo "$issue_structure" | jq -r ".parent_issues[$i]")
    parent_title=$(echo "$parent" | jq -r '.title')
    parent_body=$(echo "$parent" | jq -r '.body')
    parent_labels=$(echo "$parent" | jq -c '.labels')

    echo "Creating parent issue: $parent_title"

    # Create parent issue
    parent_response=$(curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues" \
      -d "{\"title\":\"$parent_title\",\"body\":\"$parent_body\",\"labels\":$parent_labels}")

    parent_number=$(echo "$parent_response" | jq -r '.number')

    # Check if parent issue was created successfully
    if [ "$parent_number" = "null" ]; then
      echo "Error creating parent issue: $(echo "$parent_response" | jq -r '.message')"
      continue
    fi

    echo "Created parent issue #$parent_number"

    # Create sub-issues
    sub_issues=$(echo "$parent" | jq -c '.sub_issues[]')
    sub_count=$(echo "$parent" | jq '.sub_issues | length')
    echo "Found $sub_count sub-issues to create for parent #$parent_number"

    # Create a temporary file to store sub-issue numbers
    sub_numbers_file=$(mktemp)

    # Process each sub-issue
    echo "$sub_issues" | while read -r sub_issue; do
      sub_title=$(echo "$sub_issue" | jq -r '.title')
      sub_body=$(echo "$sub_issue" | jq -r '.body')
      sub_labels=$(echo "$sub_issue" | jq -c '.labels')

      # Add parent reference to body
      sub_body="Part of #$parent_number: $parent_title\n\n$sub_body"

      echo "Creating sub-issue: $sub_title"

      # Create sub-issue
      sub_response=$(curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "https://api.github.com/repos/$REPO/issues" \
        -d "{\"title\":\"$sub_title\",\"body\":\"$sub_body\",\"labels\":$sub_labels}")

      sub_number=$(echo "$sub_response" | jq -r '.number')

      # Check if sub-issue was created successfully
      if [ "$sub_number" = "null" ]; then
        echo "Error creating sub-issue: $(echo "$sub_response" | jq -r '.message')"
        continue
      fi

      echo "Created sub-issue #$sub_number"

      # Add to temporary file with parent number for reference
      echo "$sub_number:$parent_number" >> "$sub_numbers_file"
    done

    # Wait a moment to ensure all issues are created
    sleep 2

    # Now try to add all sub-issues using the sub-issues API
    echo "Attempting to link sub-issues to parent #$parent_number using the Sub-issues API..."

    # First, check if the Sub-issues API is available by trying to list existing sub-issues
    api_check=$(curl -s -o /dev/null -w "%{http_code}" \
      -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues/$parent_number/sub_issues")

    if [ "$api_check" = "404" ] || [ "$api_check" = "403" ]; then
      echo "Warning: Sub-issues API is not available (HTTP $api_check). This feature requires:"
      echo "  - GitHub organization account"
      echo "  - Sub-issues feature to be enabled for the organization"
      echo "  - Token with appropriate permissions"
      echo "Using body references instead. Sub-issues will appear as regular issues with 'Part of #' references."
    else
      echo "Sub-issues API is available. Linking sub-issues..."

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
    fi

    # Clean up temporary file
    rm -f "$sub_numbers_file"
  done

  echo "Created $parent_count parent issues and their sub-issues"
}

# Main execution
echo "=== LLuMinary GitHub Issue Reset Tool ==="
echo "This script will delete all existing issues and start fresh."
echo "Repository: $REPO"

# Backup issues
backup_issues

# Confirm before proceeding
if [ "$SKIP_CONFIRMATION" = false ] && [ "$DRY_RUN" = false ]; then
  echo ""
  echo "WARNING: This will close ALL issues in the repository."
  echo "A backup has been created at: $BACKUP_FILE"
  echo ""
  read -p "Are you sure you want to continue? (y/N) " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 1
  fi
fi

# Close all issues
close_all_issues

# Create new issues if structure file provided
create_new_issues

echo ""
if [ "$DRY_RUN" = true ]; then
  echo "Dry run completed. No changes were made."
else
  echo "All issues have been closed."
  if [ ! -z "$ISSUE_STRUCTURE_FILE" ]; then
    echo "New issues have been created from $ISSUE_STRUCTURE_FILE."
  else
    echo "To create new issues, run this script with the -s option and a JSON structure file."
  fi
fi

echo ""
echo "Done!"
