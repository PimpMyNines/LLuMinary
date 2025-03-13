#!/bin/bash
# Installation script for LLuMinary GitHub Issue Management Tools

echo "Installing LLuMinary GitHub Issue Management Tools..."

# Make scripts executable
chmod +x *.sh

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "❌ jq is not installed. Please install it before using these tools."
    echo "   macOS: brew install jq"
    echo "   Ubuntu/Debian: sudo apt-get install jq"
    echo "   CentOS/RHEL: sudo yum install jq"
    HAS_PREREQS=false
else
    echo "✅ jq is installed."
fi

# Check for bc
if ! command -v bc &> /dev/null; then
    echo "❌ bc is not installed. Please install it before using these tools."
    echo "   macOS: brew install bc"
    echo "   Ubuntu/Debian: sudo apt-get install bc"
    echo "   CentOS/RHEL: sudo yum install bc"
    HAS_PREREQS=false
else
    echo "✅ bc is installed."
fi

# Check for GitHub token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GITHUB_TOKEN environment variable is not set."
    echo "   Please set it with: export GITHUB_TOKEN='your_github_token'"
    HAS_PREREQS=false
else
    echo "✅ GITHUB_TOKEN is set."
fi

# Create symbolic links in user's bin directory if it exists and is in PATH
if [ -d "$HOME/bin" ] && [[ ":$PATH:" == *":$HOME/bin:"* ]]; then
    echo "Creating symbolic links in $HOME/bin..."
    ln -sf "$(pwd)/test_github_api.sh" "$HOME/bin/lluminary-test-github-api"
    ln -sf "$(pwd)/find_issues_with_tasks.sh" "$HOME/bin/lluminary-find-issues-with-tasks"
    ln -sf "$(pwd)/create_sub_issues.sh" "$HOME/bin/lluminary-create-sub-issues"
    echo "✅ Symbolic links created. You can now use the following commands from anywhere:"
    echo "   lluminary-test-github-api"
    echo "   lluminary-find-issues-with-tasks"
    echo "   lluminary-create-sub-issues <issue_number> [issue_number2 ...]"
else
    echo "ℹ️ $HOME/bin directory not found in PATH. Skipping symbolic link creation."
    echo "   You can still use the scripts from this directory."
fi

# Test GitHub API access if prerequisites are met
if [ "$HAS_PREREQS" != "false" ]; then
    echo "Testing GitHub API access..."
    ./test_github_api.sh
fi

echo "Installation complete!"
echo "For more information, see README.md"
