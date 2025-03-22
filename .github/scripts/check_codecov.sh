#!/bin/bash
# Script to check if CODECOV_TOKEN is configured correctly in GitHub repository secrets

echo "Codecov Configuration Checker"
echo "============================"
echo ""
echo "This script checks if the CODECOV_TOKEN is properly configured in your GitHub repository."
echo ""

# Check if we're running in GitHub Actions
if [ -n "$GITHUB_ACTIONS" ]; then
  echo "Running in GitHub Actions environment."
  
  # Check if CODECOV_TOKEN is set
  if [ -n "$CODECOV_TOKEN" ]; then
    echo "✅ CODECOV_TOKEN is set in the GitHub Actions environment."
    echo "👍 Coverage reports should upload correctly to Codecov."
  else
    echo "❌ CODECOV_TOKEN is NOT set in the GitHub Actions environment."
    echo "⚠️ Follow these steps to configure the token:"
    echo ""
    echo "1. Sign up or log in to codecov.io using your GitHub account"
    echo "2. Add your repository to Codecov by selecting it from the list"
    echo "3. Navigate to Repository Settings > General > Repository Upload Token"
    echo "4. Copy the generated token"
    echo "5. In your GitHub repository:"
    echo "   - Go to Settings > Secrets and variables > Actions"
    echo "   - Click on 'New repository secret'"
    echo "   - Name: CODECOV_TOKEN"
    echo "   - Value: [paste the token copied from Codecov]"
    echo "   - Click 'Add secret'"
    echo ""
    echo "For more information, see .github/workflows/README.md"
  fi
else
  echo "Running in local environment."
  echo ""
  echo "To verify your Codecov configuration:"
  echo ""
  echo "1. Check if you can authenticate with Codecov:"
  echo "   curl -s https://codecov.io/bash | bash -s -- -t YOUR_CODECOV_TOKEN -f ./coverage.xml"
  echo ""
  echo "2. Or run a test GitHub Action workflow with CODECOV_TOKEN configured"
  echo ""
  echo "For more information, see .github/workflows/README.md"
fi

# Provide guidance on CI workflow verification
echo ""
echo "To verify the matrix-docker-tests workflow:"
echo "1. Make a small change to a Python file"
echo "2. Commit the change"
echo "3. Push to a branch and create a PR"
echo "4. Check if the workflow runs correctly"
echo "5. Verify that coverage reports are uploaded to Codecov"