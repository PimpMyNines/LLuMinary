# LLuMinary GitHub Issue Management Tools

This document describes tools to help manage GitHub issues for the LLuMinary project, specifically focusing on converting tasks in parent issues into proper sub-issues.

## Overview

The GitHub sub-issues feature allows for better organization and tracking of complex issues by breaking them down into smaller, manageable pieces. These tools help automate the process of:

1. Identifying issues with tasks that need to be converted to sub-issues
2. Creating sub-issues for each task
3. Linking sub-issues to parent issues
4. Updating parent issues to reflect the changes

## Prerequisites

- GitHub Personal Access Token with `repo` scope
- `jq` command-line tool installed
- `bc` command-line calculator installed

## Setup

1. Set your GitHub token as an environment variable:

```bash
export GITHUB_TOKEN='your_github_token'
```

2. Make sure all scripts are executable:

```bash
chmod +x *.sh
```

## Available Scripts

### 1. Test GitHub API Access

Before using the main scripts, verify that your GitHub token has the necessary permissions:

```bash
./test_github_api.sh
```

This script will:
- Test basic API access
- Test access to the sub-issues API
- Verify write permissions by creating and deleting a test label

### 2. Find Issues with Tasks

To identify which issues have tasks that need to be converted to sub-issues:

```bash
./find_issues_with_tasks.sh
```

This script will output a list of issue numbers, titles, and the number of tasks in each issue.

### 3. Create Sub-Issues

Once you've identified issues with tasks, use this script to create sub-issues:

```bash
./create_sub_issues.sh <issue_number> [issue_number2 ...]
```

Example:

```bash
./create_sub_issues.sh 86 85 62
```

This script will:
1. Get the issue details from GitHub
2. Extract tasks in the format `- [ ] Task description`
3. Create a new issue for each task
4. Add each new issue as a sub-issue to the parent
5. Update the parent issue description

## Story Points Distribution

The script distributes story points from the parent issue evenly among the sub-issues. For example, if a parent issue has 10 story points and 5 tasks, each sub-issue will get 2 story points.

## Workflow

The recommended workflow is:

1. Run `./test_github_api.sh` to verify API access
2. Run `./find_issues_with_tasks.sh` to identify issues with tasks
3. Run `./create_sub_issues.sh` with the issue numbers you want to process

For bulk processing, you can use the batch processing script:

```bash
./batch_process.sh -q 'is:open label:enhancement' -l 10
```

This will search for open issues with the 'enhancement' label and process up to 10 of them.

## Batch Processing

The batch processing script allows you to process multiple issues at once based on a search query. This is useful for converting tasks to sub-issues in bulk.

### Usage

```bash
./batch_process.sh [options]
```

### Options

- `-q, --query <query>` - Search query to find issues (e.g., 'label:enhancement')
- `-l, --limit <number>` - Maximum number of issues to process (default: 10)
- `-d, --dry-run` - Show what would be done without making changes
- `-h, --help` - Show help message

### Examples

```bash
# Process up to 5 enhancement issues
./batch_process.sh -q 'label:enhancement' -l 5

# Dry run for issues with tasks
./batch_process.sh -q 'is:open has:tasks' --dry-run

# Process issues with a specific milestone
./batch_process.sh -q 'milestone:"Version 1.0"'
```

### Search Query Syntax

The search query uses GitHub's search syntax. Some useful query parameters:

- `is:open` - Only open issues
- `is:closed` - Only closed issues
- `label:X` - Issues with label X
- `milestone:X` - Issues with milestone X
- `author:X` - Issues created by user X
- `assignee:X` - Issues assigned to user X
- `no:assignee` - Issues with no assignee
- `created:>YYYY-MM-DD` - Issues created after date
- `updated:>YYYY-MM-DD` - Issues updated after date

You can combine these with AND (space) and OR (|) operators.

## Troubleshooting

- If you get authentication errors, check that your GitHub token is set correctly and has the necessary permissions.
- If tasks aren't being extracted, ensure they follow the format `- [ ] Task description` in the issue body.
- If you encounter JSON parsing errors, check that the issue body doesn't contain special characters that might break the JSON.

## Notes

- These scripts require the GitHub API's sub-issues feature, which is currently in public preview for organizations.
- The scripts assume the repository is "PimpMyNines/LLuMinary". If you need to use them for a different repository, edit the `REPO` variable in each script.
