#!/bin/bash
# Script to help users set up their GitHub token

echo "GitHub Token Setup for LLuMinary Issue Management Tools"
echo "======================================================="
echo ""
echo "This script will help you set up your GitHub token for use with the issue management tools."
echo ""

# Check if token is already set
if [ ! -z "$GITHUB_TOKEN" ]; then
    echo "A GitHub token is already set in your environment."
    echo "Current token: ${GITHUB_TOKEN:0:4}...${GITHUB_TOKEN: -4}"
    echo ""
    read -p "Do you want to replace it? (y/n): " replace_token
    if [[ "$replace_token" != "y" && "$replace_token" != "Y" ]]; then
        echo "Keeping existing token."
        exit 0
    fi
fi

echo "To use these tools, you need a GitHub Personal Access Token with 'repo' scope."
echo "If you don't have one yet, you can create one at:"
echo "https://github.com/settings/tokens/new"
echo ""
echo "Make sure to select the 'repo' scope when creating the token."
echo ""

# Get token from user
read -p "Enter your GitHub token: " github_token

if [ -z "$github_token" ]; then
    echo "Error: No token provided."
    exit 1
fi

# Set token for current session
export GITHUB_TOKEN="$github_token"
echo "Token set for current session."

# Ask if user wants to add to shell profile
echo ""
echo "To make this token available in all terminal sessions, you can add it to your shell profile."
read -p "Would you like to add it to your shell profile? (y/n): " add_to_profile

if [[ "$add_to_profile" == "y" || "$add_to_profile" == "Y" ]]; then
    # Determine shell profile file
    if [ -n "$ZSH_VERSION" ]; then
        profile_file="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            profile_file="$HOME/.bash_profile"
        else
            profile_file="$HOME/.bashrc"
        fi
    else
        echo "Could not determine shell profile file."
        echo "Please manually add the following line to your shell profile:"
        echo "export GITHUB_TOKEN='$github_token'"
        exit 0
    fi

    # Check if token is already in profile
    if grep -q "export GITHUB_TOKEN=" "$profile_file"; then
        # Replace existing token
        sed -i.bak "s/export GITHUB_TOKEN=.*/export GITHUB_TOKEN='$github_token'/" "$profile_file"
        echo "Updated existing token in $profile_file"
    else
        # Add token to profile
        echo "" >> "$profile_file"
        echo "# GitHub token for LLuMinary Issue Management Tools" >> "$profile_file"
        echo "export GITHUB_TOKEN='$github_token'" >> "$profile_file"
        echo "Added token to $profile_file"
    fi

    echo ""
    echo "To apply changes in current session, run:"
    echo "source $profile_file"
fi

# Test token
echo ""
echo "Testing token..."
response=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/user)

if [ "$response" -eq 200 ]; then
    echo "✅ Token is valid and working correctly!"
else
    echo "❌ Error: Token test failed with HTTP status $response"
    echo "Please check that your token is correct and has the necessary permissions."
fi

echo ""
echo "Setup complete!"
