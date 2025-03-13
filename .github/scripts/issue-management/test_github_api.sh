#!/bin/bash
# Script to test GitHub API access

# Check if GITHUB_TOKEN is set
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN environment variable is not set."
  echo "Please set it with: export GITHUB_TOKEN='your_github_token'"
  exit 1
fi

REPO="PimpMyNines/LLuMinary"

echo "Testing GitHub API access..."

# Test API access by listing issues
response=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/$REPO/issues?per_page=1)

if [ "$response" -eq 200 ]; then
  echo "✅ Successfully connected to GitHub API!"

  # Test sub-issues API
  echo "Testing sub-issues API..."
  response=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    https://api.github.com/repos/$REPO/issues/60/sub_issues)

  if [ "$response" -eq 200 ] || [ "$response" -eq 404 ]; then
    echo "✅ Sub-issues API is accessible!"
  else
    echo "❌ Error accessing sub-issues API. HTTP status: $response"
    echo "Note: The sub-issues API is currently in public preview for organizations."
    echo "Make sure your token has the necessary permissions."
  fi
else
  echo "❌ Error connecting to GitHub API. HTTP status: $response"
  echo "Please check your token and permissions."
fi

# Test repository write access by creating a test issue label
echo "Testing repository write access..."
response=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -X POST \
  https://api.github.com/repos/$REPO/labels \
  -d '{"name":"test-api-access","color":"f29513","description":"Test label for API access verification"}')

if [ "$response" -eq 201 ] || [ "$response" -eq 422 ]; then
  # 422 means the label already exists, which is fine for our test
  echo "✅ Repository write access confirmed!"

  # Clean up by deleting the test label
  curl -s -o /dev/null \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -X DELETE \
    https://api.github.com/repos/$REPO/labels/test-api-access
else
  echo "❌ Error: No write access to repository. HTTP status: $response"
  echo "Please check your token permissions. The token needs 'repo' scope."
fi

echo "Test completed."
