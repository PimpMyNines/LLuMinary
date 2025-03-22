# GitHub Scripts

This directory contains various scripts and tools for managing GitHub-related tasks for the LLuMinary project.

## Directory Structure

- **issue-management/**: Tools for managing GitHub issues, specifically for converting tasks in parent issues into proper sub-issues.
  - See [issue-management/README.md](./issue-management/README.md) for more details.

- **issue-export/**: Scripts for exporting GitHub issues to JSON files.
  - `export_github_issues.py`: Script to export GitHub issues to JSON files.
  - JSON files containing exported issues.

- **issue-cleanup/**: Scripts for cleaning up GitHub issues.
  - `cleanup_github_issues.py`: Script to clean up GitHub issues.

- **automations/**: Scripts for setting up GitHub automations.
  - `setup_github_automations.sh`: Script to set up GitHub automations.

## Root Scripts

- `rebuild_github_project.py`: Script to rebuild GitHub projects.
- `refactor_github_issues.sh`: Script to refactor GitHub issues.
- `setup_github_issues.sh`: Script to set up GitHub issues.

## Usage

Most scripts can be run directly from their respective directories. For the issue management tools, you can use the wrapper script `issue-tools.sh` at the root of the repository:

```bash
./issue-tools.sh <command> [arguments]
```

See [issue-management/README.md](./issue-management/README.md) for more details on the issue management tools.
