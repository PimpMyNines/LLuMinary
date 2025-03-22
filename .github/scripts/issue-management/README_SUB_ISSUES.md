# Sub-Issue Creation Tool

This tool helps convert tasks in GitHub issues into proper sub-issues, making them easier to track and manage.

## Prerequisites

- GitHub Personal Access Token with `repo` scope
- `jq` command-line tool installed
- `bc` command-line calculator installed

## Setup

1. Set your GitHub token as an environment variable:

```bash
export GITHUB_TOKEN='your_github_token'
```

2. Make sure the script is executable:

```bash
chmod +x create_sub_issues.sh
```

## Usage

Run the script with one or more issue numbers:

```bash
./create_sub_issues.sh <issue_number> [issue_number2 ...]
```

Example:

```bash
./create_sub_issues.sh 86 85 62
```

## What the Script Does

For each parent issue:

1. Gets the issue details from GitHub
2. Extracts tasks in the format `- [ ] Task description`
3. Creates a new issue for each task with:
   - Title: The task description
   - Body: Formatted with issue type, story points, and reference to parent issue
4. Adds each new issue as a sub-issue to the parent
5. Updates the parent issue description to remove the tasks and add a note about sub-issues

## Story Points Distribution

The script distributes story points from the parent issue evenly among the sub-issues. For example, if a parent issue has 10 story points and 5 tasks, each sub-issue will get 2 story points.

## Troubleshooting

- If you get authentication errors, check that your GitHub token is set correctly and has the necessary permissions.
- If tasks aren't being extracted, ensure they follow the format `- [ ] Task description` in the issue body.
- If you encounter JSON parsing errors, check that the issue body doesn't contain special characters that might break the JSON.

## Notes

- This script requires the GitHub API's sub-issues feature, which is currently in public preview for organizations.
- The script assumes the repository is "PimpMyNines/LLuMinary". If you need to use it for a different repository, edit the `REPO` variable in the script.
