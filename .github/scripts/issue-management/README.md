# GitHub Issue Management Tools

This directory contains scripts for managing GitHub issues for the LLuMinary project.

## Scripts

- `batch_process.sh`: Process multiple issues based on a search query
- `create_sub_issues.sh`: Create sub-issues for tasks in parent issues
- `find_issues_with_tasks.sh`: Find issues with tasks
- `find_duplicate_issues.sh`: Find potential duplicate issues
- `setup_token.sh`: Set up GitHub token for authentication
- `test_github_api.sh`: Test GitHub API access
- `reset_issues.sh`: Reset all issues (close existing and create new ones)
- `archive_issues.sh`: Archive closed issues by updating their titles and bodies

## Usage

The scripts can be run directly from this directory, but it's recommended to use the wrapper script `issue-tools.sh` in the root directory of the project.

```bash
# Set up GitHub token
./issue-tools.sh setup-token

# Test GitHub API access
./issue-tools.sh test-api

# Find issues with tasks
./issue-tools.sh find-issues

# Create sub-issues for tasks in parent issues
./issue-tools.sh create-sub-issues 86 85 62

# Process multiple issues based on a search query
./issue-tools.sh batch-process -q 'Dockerfile' -l 5
```

## Batch Processing

The `batch_process.sh` script allows you to process multiple issues based on a search query. It will find issues matching the query and create sub-issues for any tasks found in those issues.

```bash
./issue-tools.sh batch-process -q "label:enhancement" -l 5
```

Options:
- `-q, --query`: Search query to find issues (required)
- `-l, --limit`: Limit the number of issues to process (default: 10)
- `-d, --dry-run`: Perform a dry run without making any changes

Example queries:
- `label:enhancement`: Issues with the "enhancement" label
- `Dockerfile`: Issues containing "Dockerfile" in the title or body
- `is:open`: All open issues
- `created:>2023-01-01`: Issues created after January 1, 2023

You can combine multiple criteria:
```bash
./issue-tools.sh batch-process -q "label:bug is:open" -l 10
```

To perform a dry run:
```bash
./issue-tools.sh batch-process -d -q "label:enhancement" -l 5
```

## Creating Sub-Issues

The `create_sub_issues.sh` script creates sub-issues from tasks in a parent issue. Tasks are identified by the "- [ ] " format in the issue body.

```bash
./issue-tools.sh create-sub-issues 123 456 789
```

This will:
1. Fetch each specified issue
2. Extract tasks from the issue body
3. Create a new sub-issue for each task
4. Link the sub-issues to the parent issue using the GitHub Sub-issues API (if available)
5. If the Sub-issues API is not available, it will add references in the issue bodies

You can also perform a dry run to see what would happen without making any changes:

```bash
./issue-tools.sh create-sub-issues -d 123
```

#### Sub-issues API Requirements

The GitHub Sub-issues API is currently in public preview and has the following requirements:
- Only available for GitHub organization accounts (not personal accounts)
- The Sub-issues feature must be enabled for the organization
- The GitHub token must have appropriate permissions

If the Sub-issues API is not available, the script will fall back to using body references, and sub-issues will appear as regular issues in the GitHub UI with "Part of #" references.

To check if the Sub-issues API is available for your repository, you can run:

```bash
curl -s -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/OWNER/REPO/issues/ISSUE_NUMBER/sub_issues
```

Replace `YOUR_GITHUB_TOKEN`, `OWNER`, `REPO`, and `ISSUE_NUMBER` with your values.

#### Troubleshooting Sub-issues API

If you encounter errors like "Not Found" when trying to link sub-issues to their parent, it could be due to one of the following reasons:

1. **API Not Available**: The Sub-issues API might not be available for your repository or organization.
2. **Permission Issues**: Your GitHub token might not have the necessary permissions.
3. **Feature Not Enabled**: The Sub-issues feature might not be enabled for your organization.

You can check if the Sub-issues API is available for your repository by running:

```bash
curl -s -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/OWNER/REPO/issues/ISSUE_NUMBER/sub_issues"
```

Replace `YOUR_GITHUB_TOKEN`, `OWNER`, `REPO`, and `ISSUE_NUMBER` with appropriate values.

If the API is available, you should get a JSON response with sub-issues or an empty array. If the API is not available, you'll get a 404 or 403 error.

#### Using Sub-issues Without the API

Even if the Sub-issues API is not available, our scripts will still create sub-issues with references to their parent issues in the body. This provides a way to organize your issues hierarchically, even without the official Sub-issues feature.

The main difference is that without the API:
- Sub-issues will appear as regular issues in the GitHub UI
- They won't be visually nested under their parent issue
- They will have a "Part of #X" reference in their body

## Finding Issues with Tasks

The `find_issues_with_tasks.sh` script can be used to find issues with tasks. It will list all issues that have tasks in the format "- [ ] Task description".

```bash
./issue-tools.sh find-issues
```

## Finding Duplicate Issues

The `find_duplicate_issues.sh` script helps identify potential duplicate issues in your repository. It uses a similarity algorithm to compare issue titles and bodies.

### Usage

```bash
# Check recently created sub-issues for duplicates
./issue-tools.sh find-duplicates -s

# Check a specific issue for duplicates
./issue-tools.sh find-duplicates -i 123

# Check for issues with a similar title
./issue-tools.sh find-duplicates -t "Fix Docker"

# Check all issues for potential duplicates
./issue-tools.sh find-duplicates -a
```

### Options

- `-i ISSUE_NUMBER`: Check for duplicates of a specific issue
- `-t TITLE`: Check for issues with a similar title
- `-a`: Check all issues for potential duplicates
- `-s`: Check recently created sub-issues for duplicates
- `-d DAYS`: Filter by number of days since creation (default: 7)
- `-p THRESHOLD`: Set the similarity threshold percentage (default: 70)

### Examples

```bash
# Check sub-issues with a higher similarity threshold (80%)
./issue-tools.sh find-duplicates -s -p 80

# Check for issues similar to a title with a lower threshold (50%)
./issue-tools.sh find-duplicates -t "Docker setup" -p 50

# Check for duplicates of issue #123 with a custom threshold
./issue-tools.sh find-duplicates -i 123 -p 60
```

### How It Works

The script uses a Jaccard similarity algorithm to compare issue titles and bodies. Issues with a title similarity higher than the threshold (default: 70%) are considered potential duplicates. The script also calculates body similarity for additional context.

The similarity calculation:
1. Normalizes text (lowercase, remove punctuation)
2. Compares word sets between issues
3. Calculates similarity as: (common words / total unique words) * 100

This helps identify issues that might be duplicates even if they're not exact matches. You can adjust the threshold to be more or less strict in finding potential duplicates.

## Resetting Issues

The `reset_issues.sh` script allows you to delete all existing issues and start fresh with correctly created issues and subtasks.

### Usage

```bash
# Dry run to see what would be deleted
./issue-tools.sh reset-issues -d

# Backup issues to a specific file before deleting
./issue-tools.sh reset-issues -b my_backup.json

# Create new issues from a structure file
./issue-tools.sh reset-issues -s issue_structure.json

# Skip confirmation prompt and proceed with deletion
./issue-tools.sh reset-issues -y
```

### Options

- `-d, --dry-run`: Show what would be done without making changes
- `-b, --backup <file>`: Backup file to save existing issues (default: issues_backup_YYYYMMDD_HHMMSS.json)
- `-s, --structure <file>`: JSON file with new issue structure to create
- `-y, --yes`: Skip confirmation prompt

### Issue Structure File

The script can create new issues from a JSON structure file. A template is provided at `.github/scripts/issue-management/issue_structure_template.json`. The structure file should have the following format:

```json
{
  "parent_issues": [
    {
      "title": "Parent Issue Title",
      "body": "Parent issue description with tasks",
      "labels": ["label1", "label2"],
      "sub_issues": [
        {
          "title": "Sub-Issue Title",
          "body": "Sub-issue description",
          "labels": ["label1", "label3"]
        },
        // More sub-issues...
      ]
    },
    // More parent issues...
  ]
}
```

### GitHub Sub-issues API

The script attempts to use the GitHub Sub-issues API to create proper parent-child relationships between issues. However, this API has some requirements:

1. **Organization Account**: The Sub-issues API is only available for GitHub organization accounts, not personal accounts.
2. **Feature Enablement**: The Sub-issues feature must be enabled for your organization.
3. **Token Permissions**: Your GitHub token must have appropriate permissions.

If the Sub-issues API is not available, the script will fall back to using body references (adding "Part of #X" to the issue body). In this case, sub-issues will appear as regular issues in the GitHub UI, but they will have a reference to their parent issue.

You can check if the Sub-issues API is available for your repository by running:

```bash
curl -s -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/OWNER/REPO/issues/ISSUE_NUMBER/sub_issues"
```

Replace `YOUR_GITHUB_TOKEN`, `OWNER`, `REPO`, and `ISSUE_NUMBER` with appropriate values.

#### Troubleshooting Sub-issues API

If you encounter errors like "Not Found" when trying to link sub-issues to their parent, it could be due to one of the following reasons:

1. **API Not Available**: The Sub-issues API might not be available for your repository or organization.
2. **Permission Issues**: Your GitHub token might not have the necessary permissions.
3. **Feature Not Enabled**: The Sub-issues feature might not be enabled for your organization.

You can check if the Sub-issues API is available for your repository by running:

```bash
curl -s -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/OWNER/REPO/issues/ISSUE_NUMBER/sub_issues"
```

Replace `YOUR_GITHUB_TOKEN`, `OWNER`, `REPO`, and `ISSUE_NUMBER` with appropriate values.

If the API is available, you should get a JSON response with sub-issues or an empty array. If the API is not available, you'll get a 404 or 403 error.

#### Using Sub-issues Without the API

Even if the Sub-issues API is not available, our scripts will still create sub-issues with references to their parent issues in the body. This provides a way to organize your issues hierarchically, even without the official Sub-issues feature.

The main difference is that without the API:
- Sub-issues will appear as regular issues in the GitHub UI
- They won't be visually nested under their parent issue
- They will have a "Part of #X" reference in their body

### Workflow for Resetting Issues

1. First, run a dry run to see what would be deleted:
   ```bash
   ./issue-tools.sh reset-issues -d
   ```

2. Create a backup of existing issues:
   ```bash
   ./issue-tools.sh reset-issues -b my_backup.json
   ```

3. Create or modify the issue structure file based on the template.

4. Reset issues and create new ones from the structure file:
   ```bash
   ./issue-tools.sh reset-issues -s my_structure.json
   ```

## GitHub Token

The scripts require a GitHub token with the appropriate permissions to access the GitHub API. You can set up the token using the `setup_token.sh` script.

```bash
./issue-tools.sh setup-token
```

## Testing API Access

You can test your GitHub API access using the `test_github_api.sh` script.

```bash
./issue-tools.sh test-api
```

## Available Scripts

- `setup_token.sh` - Set up GitHub token for authentication
- `test_github_api.sh` - Test GitHub API access and permissions
- `find_issues_with_tasks.sh` - Find issues with tasks that need to be converted to sub-issues
- `create_sub_issues.sh` - Create sub-issues for tasks in parent issues
- `batch_process.sh` - Process multiple issues based on a search query
- `install.sh` - Install and set up the tools

## Batch Processing

The `batch_process.sh` script allows you to process multiple issues at once based on a search query. This is useful for converting tasks to sub-issues in bulk.

```bash
./batch_process.sh -q 'is:open label:enhancement' -l 10
```

Options:
- `-q, --query <query>` - Search query to find issues (e.g., 'label:enhancement')
- `-l, --limit <number>` - Maximum number of issues to process (default: 10)
- `-d, --dry-run` - Show what would be done without making changes
- `-h, --help` - Show help message

## Documentation

- [Issue Management Guide](./ISSUE_MANAGEMENT.md) - Detailed documentation on using these tools
- [Sub-Issues Creation Tool](./README_SUB_ISSUES.md) - Specific documentation for the sub-issues creation tool

## Prerequisites

- GitHub Personal Access Token with `repo` scope
- `jq` command-line tool installed
- `bc` command-line calculator installed

## Archiving Closed Issues

If you have many closed issues that you want to clearly mark as archived without completely deleting them (which GitHub doesn't allow), you can use the `archive_issues.sh` script to update their titles and bodies.

```bash
# Run a dry run to see what would be archived
./issue-tools.sh archive-issues -d

# Archive all closed issues
./issue-tools.sh archive-issues

# Archive issues closed more than 30 days ago
./issue-tools.sh archive-issues -a 30

# Use a custom prefix instead of [ARCHIVED]
./issue-tools.sh archive-issues -p "[OLD]"

# Skip confirmation prompt
./issue-tools.sh archive-issues -y
```

This script will:
1. Add a prefix (default: `[ARCHIVED]`) to the issue title
2. Add an archive notice to the issue body
3. Leave the issue in the closed state

Options:
- `-d, --dry-run`: Show what would be done without making changes
- `-p, --prefix <text>`: Prefix to add to issue titles (default: `[ARCHIVED]`)
- `-a, --age <days>`: Only archive issues closed more than X days ago
- `-y, --yes`: Skip confirmation prompt

Note: GitHub does not provide an API to permanently delete issues. This script helps manage closed issues by clearly marking them as archived.
