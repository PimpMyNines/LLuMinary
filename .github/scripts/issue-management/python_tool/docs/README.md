# GitHub Project Management Tool Documentation

## Introduction

This tool provides a comprehensive Python-based approach to managing GitHub projects, issues, sub-issues, and roadmaps. It replaces the bash scripts with a more robust solution that provides better error handling, cross-platform support, and access to the full GitHub API.

## Installation

```bash
# Clone the repository
git clone https://github.com/PimpMyNines/github-project-management.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## Configuration

Create a configuration file at one of these locations:

1. Current directory: `config.yaml`
2. User home directory: `~/.github-pm/config.yaml`
3. Custom location, set with `GITHUB_PM_CONFIG` environment variable

See the example configuration in `config/config.yaml.example`.

## Authentication

Authenticate with GitHub using one of the following methods:

1. Set the `GITHUB_TOKEN` environment variable
2. Add your token to the configuration file
3. Use GitHub App authentication (for more advanced scenarios)

```bash
# Set token in environment
export GITHUB_TOKEN=your-github-token
```

## Command-Line Usage

### Managing Issues

```bash
# Create an issue
github-pm issues create --repo owner/repo --title "Issue title" --body "Issue description"

# Create a sub-issue
github-pm issues create --repo owner/repo --title "Sub-issue title" --parent 123

# Convert tasks to sub-issues
github-pm issues convert-tasks --repo owner/repo --issue 123
```

### Managing Projects

```bash
# Create a project
github-pm projects create --name "Project Name" --repo owner/repo

# Add issue to project
github-pm projects add-issue --project-id 12345 --repo owner/repo --issue 123 --column "To Do"
```

### Managing Roadmaps

```bash
# Create a milestone
github-pm roadmap create --repo owner/repo --title "v1.0 Release" --due-date 2023-12-31

# Generate roadmap report
github-pm roadmap report --repo owner/repo
```

## Python API Usage

```python
from github_project_management.auth.github_auth import GitHubAuth
from github_project_management.issues.issue_manager import IssueManager
from github_project_management.utils.config import Config

# Initialize configuration and authentication
config = Config()
auth = GitHubAuth(config)

# Create an issue manager
issue_manager = IssueManager(auth)

# Create an issue
issue = issue_manager.create_issue(
    repo="owner/repo",
    title="Issue title",
    body="Issue description",
    labels=["bug", "high-priority"],
    assignees=["username"]
)

# Print result
print(f"Created issue #{issue['number']}")
```

## Migration from Bash Scripts

Replace existing bash scripts with the equivalent Python commands:

### Old Bash Method
```bash
./issue-tools.sh create-issue "Issue title" "Description" owner/repo
```

### New Python Method
```bash
github-pm issues create --title "Issue title" --body "Description" --repo owner/repo
```

## Troubleshooting

If you encounter issues:

1. Check your GitHub token permissions
2. Verify configuration file format
3. Enable debug logging with `--verbose` flag
4. Look for error messages in the logs

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](../CONTRIBUTING.md) file for details.
