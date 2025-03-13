#!/bin/bash
# Wrapper script for LLuMinary GitHub Issue Management Tools

# Determine the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR"

# Function to display help message
show_help() {
  echo "LLuMinary GitHub Issue Management Tools"
  echo ""
  echo "Usage: $0 <command> [options]"
  echo ""
  echo "Commands:"
  echo "  setup-token           Set up GitHub token for API access"
  echo "  test-api              Test GitHub API access"
  echo "  find-issues           Find issues with tasks"
  echo "  create-sub-issues     Create sub-issues from tasks in an issue"
  echo "  batch-process         Process multiple issues based on a search query"
  echo "  find-duplicates       Find potential duplicate issues"
  echo "  reset-issues          Reset all issues (close existing and create new ones)"
  echo "  archive-issues        Archive closed issues by updating their titles and bodies"
  echo ""
  echo "Options:"
  echo "  -h, --help            Show this help message"
  echo "  -d, --dry-run         Dry run (don't make any changes)"
  echo "  -q, --query           Search query for batch processing"
  echo "  -l, --limit           Limit number of issues to process"
  echo ""
  echo "Examples:"
  echo "  $0 setup-token"
  echo "  $0 test-api"
  echo "  $0 find-issues"
  echo "  $0 create-sub-issues 86 85 62"
  echo "  $0 create-sub-issues -d 86 85 62"
  echo "  $0 batch-process -q 'Dockerfile' -l 5"
  echo "  $0 find-duplicates -s"
  echo "  $0 reset-issues -d"
  echo "  $0 reset-issues -f issue_structure.json"
  echo "  $0 archive-issues -d"
  echo "  $0 archive-issues -p '[OLD]' -a 30"
}

# Check if a command was provided
if [ $# -eq 0 ]; then
  show_help
  exit 1
fi

# Parse command
COMMAND="$1"
shift

# Execute the appropriate script based on the command
case "$COMMAND" in
  setup-token)
    "$TOOLS_DIR/setup_token.sh"
    ;;

  test-api)
    "$TOOLS_DIR/test_github_api.sh"
    ;;

  find-issues)
    "$TOOLS_DIR/find_issues_with_tasks.sh"
    ;;

  create-sub-issues)
    # Check for dry run flag
    DRY_RUN=false
    if [[ "$1" == "-d" || "$1" == "--dry-run" ]]; then
      DRY_RUN=true
      shift
    fi

    # Check if issue numbers are provided
    if [ $# -eq 0 ]; then
      echo "Error: No issue numbers provided"
      show_help
      exit 1
    fi

    # Export the dry run flag for the script
    export DRY_RUN

    # Process each issue number
    for issue_number in "$@"; do
      "$TOOLS_DIR/create_sub_issues.sh" "$issue_number"
    done
    ;;

  batch-process)
    # Pass all arguments to the batch process script
    "$TOOLS_DIR/batch_process.sh" "$@"
    ;;

  find-duplicates)
    # Pass all arguments to the find duplicates script
    "$TOOLS_DIR/find_duplicate_issues.sh" "$@"
    ;;

  reset-issues)
    # Pass all arguments to the reset issues script
    "$TOOLS_DIR/reset_issues.sh" "$@"
    ;;

  archive-issues)
    # Pass all arguments to the archive issues script
    "$TOOLS_DIR/archive_issues.sh" "$@"
    ;;

  -h|--help)
    show_help
    ;;

  *)
    echo "Error: Unknown command '$COMMAND'"
    show_help
    exit 1
    ;;
esac

exit 0
