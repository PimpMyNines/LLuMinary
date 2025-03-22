#!/bin/bash
# Wrapper script for LLuMinary GitHub Issue Management Tools

# Display help message
show_help() {
  echo "LLuMinary GitHub Issue Management Tools"
  echo ""
  echo "Usage: $0 <command> [options]"
  echo ""
  echo "Commands:"
  echo "  setup-token           Set up GitHub token for authentication"
  echo "  test-api              Test GitHub API access"
  echo "  find-issues           Find issues with tasks"
  echo "  create-sub-issues     Create sub-issues for tasks in parent issues"
  echo "  batch-process         Process multiple issues based on a search query"
  echo "  find-duplicates       Find potential duplicate issues"
  echo "  reset-issues          Delete all existing issues and start fresh"
  echo "  help                  Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 setup-token"
  echo "  $0 test-api"
  echo "  $0 find-issues"
  echo "  $0 create-sub-issues 86 85 62"
  echo "  $0 batch-process -q 'Dockerfile' -l 5"
  echo "  $0 find-duplicates -s"
  echo "  $0 reset-issues -d"
  echo ""
  echo "For more information on a specific command, run: $0 <command> --help"
}

# Main execution
if [ $# -eq 0 ]; then
  show_help
  exit 0
fi

# Get the command
COMMAND="$1"
shift

# Find the tools directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$SCRIPT_DIR/.github/scripts/issue-management"

# If tools directory doesn't exist in the script directory, try to find it
if [ ! -d "$TOOLS_DIR" ]; then
  # Try to find the tools directory in the current directory
  if [ -d ".github/scripts/issue-management" ]; then
    TOOLS_DIR="$(pwd)/.github/scripts/issue-management"
  else
    echo "Error: Could not find the tools directory."
    echo "Please run this script from the root of the repository."
    exit 1
  fi
fi

echo "Using tools directory: $TOOLS_DIR"

# Execute the command
case "$COMMAND" in
  setup-token)
    "$TOOLS_DIR/setup_token.sh" "$@"
    ;;
  test-api)
    "$TOOLS_DIR/test_github_api.sh" "$@"
    ;;
  find-issues)
    "$TOOLS_DIR/find_issues_with_tasks.sh" "$@"
    ;;
  create-sub-issues)
    "$TOOLS_DIR/create_sub_issues.sh" "$@"
    ;;
  batch-process)
    "$TOOLS_DIR/batch_process.sh" "$@"
    ;;
  find-duplicates)
    "$TOOLS_DIR/find_duplicate_issues.sh" "$@"
    ;;
  reset-issues)
    "$TOOLS_DIR/reset_issues.sh" "$@"
    ;;
  help)
    show_help
    ;;
  *)
    echo "Error: Unknown command '$COMMAND'"
    show_help
    exit 1
    ;;
esac
