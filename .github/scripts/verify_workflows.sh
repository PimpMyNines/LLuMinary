#!/bin/bash
# Script to verify GitHub Actions workflows are working correctly

echo "GitHub Actions Workflow Verification"
echo "==================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Docker is running
if command_exists docker; then
    if docker info >/dev/null 2>&1; then
        echo "✅ Docker is running and available"
    else
        echo "❌ Docker is installed but not running"
        echo "Please start Docker before continuing"
        exit 1
    fi
else
    echo "❌ Docker is not installed"
    echo "Please install Docker to continue"
    exit 1
fi

# Create a test branch
current_branch=$(git branch --show-current)
test_branch="test-github-actions-$(date +%Y%m%d%H%M%S)"

echo ""
echo "Creating test branch: $test_branch"
git checkout -b "$test_branch"

# Make a small change to trigger workflows
echo ""
echo "Making a small change to trigger workflows"

# Update version.py with a timestamp to simulate a code change
touch src/lluminary/version.py
echo "# Workflow test timestamp: $(date)" >> src/lluminary/version.py

# Commit the change
echo ""
echo "Committing the change"
git add src/lluminary/version.py
git commit -m "ci: Test GitHub Actions workflows"

# Check if the user wants to push the branch
echo ""
echo "Would you like to push this branch and create a PR to test the workflows? (y/n)"
read -r push_choice

if [[ $push_choice == "y" || $push_choice == "Y" ]]; then
    # Push the branch
    echo ""
    echo "Pushing branch $test_branch"
    git push -u origin "$test_branch"
    
    # Check if gh CLI is available for PR creation
    if command_exists gh; then
        echo ""
        echo "Creating PR with gh CLI"
        gh pr create --title "ci: Test GitHub Actions workflows" \
            --body "This PR is a test for the GitHub Actions workflows." \
            --base "$current_branch" \
            --head "$test_branch"
        
        # Provide instructions for checking workflow results
        echo ""
        echo "PR created successfully. Please check the following:"
        echo "1. Go to the Actions tab in GitHub to see the workflow runs"
        echo "2. Verify that matrix-docker-tests workflow runs correctly"
        echo "3. Check that coverage reports are uploaded to Codecov"
        echo "4. Verify that the test summary is added as a PR comment"
    else
        echo ""
        echo "gh CLI not found. Please create a PR manually:"
        echo "1. Go to your repository on GitHub"
        echo "2. Click 'Compare & pull request' for branch $test_branch"
        echo "3. Set the title to 'ci: Test GitHub Actions workflows'"
        echo "4. Set the base branch to $current_branch"
        echo "5. Create the PR and check the Actions tab for workflow runs"
    fi
else
    # Return to the original branch
    echo ""
    echo "Not pushing changes. Returning to original branch $current_branch"
    git checkout "$current_branch"
    echo "You can push the test branch later with:"
    echo "git checkout $test_branch"
    echo "git push -u origin $test_branch"
fi

echo ""
echo "Done!"