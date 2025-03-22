#!/bin/bash
# Script to find issues with tasks that need to be converted to sub-issues

# Check if GITHUB_TOKEN is set
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN environment variable is not set."
  echo "Please set it with: export GITHUB_TOKEN='your_github_token'"
  exit 1
fi

REPO="PimpMyNines/LLuMinary"

echo "Finding issues with tasks..."

# Get all open issues
issues=$(curl -s \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/$REPO/issues?state=open&per_page=100")

# Check each issue for tasks
echo "$issues" | jq -r '.[] | {number: .number, title: .title, body: .body}' | jq -s '.' | \
jq -r '.[] | select(.body | contains("- [ ]")) | {number: .number, title: .title, tasks: (.body | split("\n") | map(select(contains("- [ ]"))) | length)}' | \
jq -r '.number, .title, "Tasks: \(.tasks)", "---"'

echo "Done!"
