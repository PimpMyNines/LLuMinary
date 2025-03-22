#!/bin/bash
# Script to archive closed issues by updating their titles and bodies

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
DAYS=0
PREFIX="[ARCHIVED]"
SKIP_CONFIRMATION=false

# Function to display help message
show_help() {
  echo "Usage: $0 [options]"
  echo "Archive closed issues by updating their titles and bodies."
  echo ""
  echo "Options:"
  echo "  -d, --dry-run              Show what would be done without making changes"
  echo "  -p, --prefix <text>        Prefix to add to issue titles (default: $PREFIX)"
  echo "  -a, --age <days>           Only archive issues closed more than X days ago (default: all)"
  echo "  -y, --yes                  Skip confirmation prompt"
  echo "  -h, --help                 Display this help message"
  echo ""
  echo "Examples:"
  echo "  $0 -d                      Dry run to see what would be archived"
  echo "  $0 -p '[OLD]'              Use '[OLD]' as the prefix instead of '[ARCHIVED]'"
  echo "  $0 -a 30                   Only archive issues closed more than 30 days ago"
  echo "  $0 -y                      Skip confirmation prompt and proceed with archiving"
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
    -p|--prefix)
      PREFIX="$2"
      shift
      shift
      ;;
    -a|--age)
      DAYS="$2"
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

# Function to get closed issues
get_closed_issues() {
  echo "Fetching closed issues..."

  # Get closed issues
  closed_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues?state=closed&per_page=100")

  # Count issues
  issue_count=$(echo "$closed_issues" | jq 'length')
  echo "Found $issue_count closed issues"

  # Filter issues that don't already have the prefix
  filtered_issues=$(echo "$closed_issues" | jq --arg prefix "$PREFIX" '[.[] | select(.title | startswith($prefix) | not)]')
  filtered_count=$(echo "$filtered_issues" | jq 'length')
  echo "Found $filtered_count closed issues without the '$PREFIX' prefix"

  # If age filter is specified, apply it
  if [ "$DAYS" -gt 0 ]; then
    # Calculate the cutoff date in seconds since epoch
    cutoff_date=$(date -v-${DAYS}d +%s)

    # Filter issues closed before the cutoff date
    age_filtered_issues=$(echo "$filtered_issues" | jq --arg cutoff "$cutoff_date" '[.[] | select(.closed_at | fromdateiso8601 | todate | strptime("%Y-%m-%dT%H:%M:%SZ") | mktime < ($cutoff | tonumber))]')
    age_filtered_count=$(echo "$age_filtered_issues" | jq 'length')
    echo "Found $age_filtered_count closed issues older than $DAYS days"

    filtered_issues="$age_filtered_issues"
    filtered_count="$age_filtered_count"
  fi

  echo "$filtered_issues"
}

# Function to archive issues
archive_issues() {
  local issues="$1"
  local count=$(echo "$issues" | jq 'length')

  echo "Preparing to archive $count issues..."

  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would archive $count issues"

    # Show sample of what would be done
    for i in $(seq 0 $(($count > 5 ? 4 : $count - 1))); do
      issue_number=$(echo "$issues" | jq -r ".[$i].number")
      issue_title=$(echo "$issues" | jq -r ".[$i].title")
      echo "[DRY RUN] Would update #$issue_number: '$issue_title' to '$PREFIX $issue_title'"
    done

    if [ $count -gt 5 ]; then
      echo "[DRY RUN] ... and $(($count - 5)) more issues"
    fi

    return
  fi

  # Confirm before proceeding
  if [ "$SKIP_CONFIRMATION" = false ]; then
    echo ""
    echo "WARNING: This will update the titles and bodies of $count closed issues."
    echo "This action cannot be easily undone."
    echo ""
    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Operation cancelled."
      exit 1
    fi
  fi

  # Archive each issue
  for i in $(seq 0 $(($count - 1))); do
    issue_number=$(echo "$issues" | jq -r ".[$i].number")
    issue_title=$(echo "$issues" | jq -r ".[$i].title")
    issue_body=$(echo "$issues" | jq -r ".[$i].body")

    # Skip if title already has prefix
    if [[ "$issue_title" == "$PREFIX"* ]]; then
      echo "Issue #$issue_number already has prefix, skipping..."
      continue
    fi

    echo "Archiving issue #$issue_number: '$issue_title'"

    # Add archive notice to body
    archive_notice="---\n\n**Note:** This issue has been archived on $(date '+%Y-%m-%d') and is kept for historical reference only.\n"
    new_body="$archive_notice\n\n$issue_body"

    # Update the issue
    update_response=$(curl -s -X PATCH -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues/$issue_number" \
      -d "{\"title\":\"$PREFIX $issue_title\",\"body\":$(echo -n "$new_body" | jq -Rs .)}")

    # Check for errors
    error_message=$(echo "$update_response" | jq -r '.message // empty')
    if [ ! -z "$error_message" ]; then
      echo "Error updating issue #$issue_number: $error_message"
    else
      echo "Successfully archived issue #$issue_number"
    fi
  done

  echo "Archived $count issues"
}

# Main execution
echo "=== LLuMinary GitHub Issue Archiver ==="
echo "This script will archive closed issues by updating their titles and bodies."
echo "Repository: $REPO"

# Get closed issues
closed_issues=$(get_closed_issues)

# Archive issues
archive_issues "$closed_issues"

echo ""
if [ "$DRY_RUN" = true ]; then
  echo "Dry run completed. No changes were made."
else
  echo "All specified issues have been archived."
fi

echo ""
echo "Done!"
