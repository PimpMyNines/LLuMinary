#!/usr/bin/env python3
"""
GitHub Project Rebuilder

This script rebuilds a complete GitHub project structure including:
- Repository configuration files
- GitHub Actions workflows
- Issue templates and PR templates
- Project board setup
- Issues with proper relationships, fields, and metadata
- Labels, milestones, and other project settings

It can be used to recreate a project from scratch or update an existing project.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Optional imports for GitHub API interactions
try:
    import requests
    import yaml
    from github import Github, GithubException, Project, ProjectColumn, Repository
    from github.Issue import Issue

    HAS_GITHUB_DEPS = True
except ImportError:
    HAS_GITHUB_DEPS = False

    # Define dummy class for type hints when dependencies aren't installed
    class Issue:
        pass


class GitHubProjectRebuilder:
    """Main class for rebuilding a GitHub project."""

    def __init__(
        self,
        config_file: str,
        token: Optional[str] = None,
        repo_path: Optional[str] = None,
        dry_run: bool = False,
    ):
        """Initialize the rebuilder with configuration and credentials.

        Args:
            config_file: Path to the JSON config file
            token: GitHub API token (optional if using local git)
            repo_path: Path to local repository (optional if using API only)
            dry_run: If True, don't make actual changes
        """
        self.config_file = config_file
        self.token = token
        self.repo_path = repo_path
        self.dry_run = dry_run
        self.config = self._load_config()

        # Initialize GitHub API client if token provided and dependencies installed
        self.gh_client = None
        self.repo = None
        if token and HAS_GITHUB_DEPS:
            self.gh_client = Github(token)
            repo_name = self.config.get("repository", {}).get("full_name")
            if repo_name:
                self.repo = self.gh_client.get_repo(repo_name)

        # Map to keep track of created issues and their original IDs
        self.issue_map = {}

    def _load_config(self) -> Dict:
        """Load the configuration file."""
        with open(self.config_file) as f:
            return json.load(f)

    def run(self):
        """Execute the rebuild process."""
        print(
            f"Starting GitHub project rebuild for {self.config.get('repository', {}).get('name', 'Unknown')}"
        )

        # Create repo config files
        self._create_repo_config_files()

        # Set up labels, milestones, project board
        if self.gh_client and self.repo:
            self._setup_labels()
            self._setup_milestones()
            self._setup_project_board()

        # Create issues with proper relationships
        self._create_issues()

        print("GitHub project rebuild complete!")

    def _create_repo_config_files(self):
        """Create repository configuration files."""
        if not self.repo_path:
            print("Skipping repository file creation (no repo path provided)")
            return

        repo_path = Path(self.repo_path)

        # Create .github directory structure
        (repo_path / ".github" / "ISSUE_TEMPLATE").mkdir(parents=True, exist_ok=True)
        (repo_path / ".github" / "workflows").mkdir(parents=True, exist_ok=True)

        # Process config files from the config
        config_files = self.config.get("config_files", {})
        for file_path, content in config_files.items():
            full_path = repo_path / file_path

            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)

            if self.dry_run:
                print(f"Would create file: {file_path}")
            else:
                # Handle different content formats (string vs dict)
                if isinstance(content, dict):
                    if file_path.endswith((".yml", ".yaml")):
                        if not HAS_GITHUB_DEPS:  # yaml is required for this operation
                            print(
                                f"Skipping YAML file {file_path} (yaml module not available)"
                            )
                            continue
                        with open(full_path, "w") as f:
                            yaml.dump(content, f, default_flow_style=False)
                    else:
                        with open(full_path, "w") as f:
                            json.dump(content, f, indent=2)
                else:
                    with open(full_path, "w") as f:
                        f.write(content)
                print(f"Created file: {file_path}")

    def _setup_labels(self):
        """Set up repository labels."""
        if self.dry_run:
            print("Would create labels (dry run)")
            return

        # Delete existing labels if configured to do so
        if self.config.get("delete_existing_labels", False):
            print("Deleting existing labels...")
            for label in self.repo.get_labels():
                try:
                    label.delete()
                except GithubException as e:
                    print(f"Error deleting label {label.name}: {e}")

        # Create new labels
        labels = self.config.get("labels", [])
        for label_data in labels:
            name = label_data.get("name")
            color = label_data.get("color", "ededed")  # Default to light gray
            description = label_data.get("description", "")

            try:
                # Check if label exists first
                try:
                    label = self.repo.get_label(name)
                    # Update the label if it exists
                    label.edit(name, color, description)
                    print(f"Updated label: {name}")
                except GithubException:
                    # Create the label if it doesn't exist
                    self.repo.create_label(name, color, description)
                    print(f"Created label: {name}")
            except GithubException as e:
                print(f"Error creating/updating label {name}: {e}")

    def _setup_milestones(self):
        """Set up repository milestones."""
        if self.dry_run:
            print("Would create milestones (dry run)")
            return

        # Delete existing milestones if configured to do so
        if self.config.get("delete_existing_milestones", False):
            print("Deleting existing milestones...")
            for milestone in self.repo.get_milestones():
                try:
                    milestone.delete()
                except GithubException as e:
                    print(f"Error deleting milestone {milestone.title}: {e}")

        # Create new milestones
        milestones = self.config.get("milestones", [])
        for milestone_data in milestones:
            title = milestone_data.get("title")
            description = milestone_data.get("description", "")
            due_on = milestone_data.get("due_on", None)
            state = milestone_data.get("state", "open")

            # Convert due_on to datetime if provided
            due_date = None
            if due_on:
                try:
                    due_date = datetime.fromisoformat(due_on)
                except ValueError:
                    print(f"Invalid date format for milestone {title}: {due_on}")

            try:
                # Check if milestone exists first
                existing_milestone = None
                for ms in self.repo.get_milestones():
                    if ms.title == title:
                        existing_milestone = ms
                        break

                if existing_milestone:
                    # Update the milestone if it exists
                    existing_milestone.edit(title, state, description, due_date)
                    print(f"Updated milestone: {title}")
                else:
                    # Create the milestone if it doesn't exist
                    self.repo.create_milestone(title, state, description, due_date)
                    print(f"Created milestone: {title}")
            except GithubException as e:
                print(f"Error creating/updating milestone {title}: {e}")

    def _setup_project_board(self):
        """Set up project board and columns."""
        if self.dry_run:
            print("Would create project board (dry run)")
            return

        # Get project configuration
        project_config = self.config.get("project_board", {})
        if not project_config:
            return

        project_name = project_config.get("name")
        project_desc = project_config.get("description", "")
        columns = project_config.get("columns", [])

        # Don't recreate if configured to use existing
        use_existing = project_config.get("use_existing", False)

        # Find existing project or create new one
        project = None
        if use_existing:
            for p in self.repo.get_projects():
                if p.name == project_name:
                    project = p
                    print(f"Using existing project board: {project_name}")
                    break

        if not project:
            try:
                project = self.repo.create_project(project_name, project_desc)
                print(f"Created project board: {project_name}")
            except GithubException as e:
                print(f"Error creating project board {project_name}: {e}")
                return

        # Create columns
        existing_columns = {column.name: column for column in project.get_columns()}

        for column_data in columns:
            column_name = column_data.get("name")

            if column_name in existing_columns:
                print(f"Column already exists: {column_name}")
                continue

            try:
                project.create_column(column_name)
                print(f"Created column: {column_name}")
            except GithubException as e:
                print(f"Error creating column {column_name}: {e}")

    def _get_parent_issue_number(self, original_parent_id: int) -> Optional[int]:
        """Get the new issue number for a parent based on the original ID."""
        return self.issue_map.get(original_parent_id)

    def _create_issues(self):
        """Create all issues with their relationships and fields."""
        issues = self.config.get("issues", [])

        # First pass: Create all parent issues
        print("Creating parent issues...")
        for issue_data in issues:
            if not issue_data.get("parent_id"):  # This is a parent issue
                self._create_single_issue(issue_data)

        # Second pass: Create all child issues (sub-issues)
        print("Creating child issues...")
        for issue_data in issues:
            if issue_data.get("parent_id"):  # This is a child issue
                self._create_single_issue(issue_data)

        # Third pass: Update issue relationships
        if self.gh_client and self.repo:
            print("Updating issue relationships...")
            for issue_data in issues:
                original_id = issue_data.get("id")
                new_id = self.issue_map.get(original_id)

                if not new_id:
                    continue

                # Update with additional content for relationships
                self._update_issue_relationships(issue_data, new_id)

    def _create_single_issue(
        self, issue_data: Dict[str, Any]
    ) -> Optional[Union[int, Issue]]:
        """Create a single issue based on the provided data."""
        title = issue_data.get("title")
        original_id = issue_data.get("id")

        # Format the issue body with metadata
        body = self._format_issue_body(issue_data)

        # Handle labels
        labels = issue_data.get("labels", [])

        # Handle assignees
        assignees = issue_data.get("assignees", [])

        # Handle milestone
        milestone_title = issue_data.get("milestone")
        milestone = None

        if self.gh_client and self.repo:
            # GitHub API approach
            if milestone_title and not self.dry_run:
                for ms in self.repo.get_milestones():
                    if ms.title == milestone_title:
                        milestone = ms
                        break

            if self.dry_run:
                print(f"Would create issue: {title}")
                self.issue_map[original_id] = 999  # Placeholder for dry run
                return 999

            try:
                new_issue = self.repo.create_issue(
                    title=title,
                    body=body,
                    labels=labels,
                    assignees=assignees,
                    milestone=milestone,
                )
                self.issue_map[original_id] = new_issue.number
                print(f"Created issue #{new_issue.number}: {title}")

                # Sleep briefly to avoid rate limiting
                time.sleep(1)
                return new_issue
            except GithubException as e:
                print(f"Error creating issue {title}: {e}")
                return None
        else:
            # Local git/CLI approach (simplified)
            try:
                # Write body to temp file to avoid command line length issues
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".md", delete=False
                ) as tmp:
                    tmp.write(body)
                    body_file = tmp.name

                # Use the body file instead of passing body directly
                cmd = [
                    "gh",
                    "issue",
                    "create",
                    "--title",
                    title,
                    "--body-file",
                    body_file,
                ]

                for label in labels:
                    cmd.extend(["--label", label])

                for assignee in assignees:
                    cmd.extend(["--assignee", assignee])

                if milestone_title:
                    cmd.extend(["--milestone", milestone_title])

                if self.dry_run:
                    print(f"Would run: {' '.join(cmd)}")
                    os.unlink(body_file)  # Clean up temp file
                    self.issue_map[original_id] = 999  # Placeholder for dry run
                    return 999

                # Run the command and get the issue number from output
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # Clean up temp file
                os.unlink(body_file)

                # Extract issue number from URL in output
                output = result.stdout.strip()
                if output.startswith("https://github.com"):
                    issue_number = int(output.split("/")[-1])
                    self.issue_map[original_id] = issue_number
                    print(f"Created issue #{issue_number}: {title}")
                    return issue_number
                else:
                    print(f"Created issue but couldn't determine number: {title}")
                    return None
            except subprocess.CalledProcessError as e:
                print(f"Error creating issue {title}: {e}")
                try:
                    os.unlink(body_file)  # Clean up temp file on error
                except:
                    pass
                return None
            except Exception as e:
                print(f"Unexpected error creating issue {title}: {e}")
                try:
                    os.unlink(body_file)  # Clean up temp file on error
                except:
                    pass
                return None

    def _format_issue_body(self, issue_data: Dict[str, Any]) -> str:
        """Format the issue body with metadata and relationships."""
        body = issue_data.get("body", "")

        # Add issue type
        issue_type = issue_data.get("type", "Task")
        body_prefix = f"**Issue Type**: {issue_type}\n\n"

        # Add story points if available
        story_points = issue_data.get("story_points")
        if story_points is not None:
            body_prefix += f"**Story Points**: {story_points}\n\n"

        # Add priority if available
        priority = issue_data.get("priority")
        if priority:
            body_prefix += f"**Priority**: {priority}\n\n"

        # Add parent reference if this is a sub-issue
        parent_id = issue_data.get("parent_id")
        if parent_id:
            new_parent_id = self._get_parent_issue_number(parent_id)
            parent_title = self._get_issue_title_by_id(parent_id)
            if new_parent_id:
                sub_issue_marker = (
                    f"# Sub-issue\nPart of #{new_parent_id}: {parent_title}\n\n---\n\n"
                )
                body_prefix += sub_issue_marker

        # Add sub-issues tracking section if this is a parent issue
        if not parent_id or issue_data.get(
            "parent_issue", False
        ):  # This is a parent issue
            parent_marker = """# Parent Issue
This is a parent issue that tracks multiple sub-issues. Progress on this issue depends on completing all sub-issues.

## Sub-issues
<!-- DO NOT EDIT BELOW THIS LINE - Sub-issues will be listed automatically -->
"""
            body_prefix += parent_marker

        # Combine the prefix with the original body - ensure no double newlines
        combined = f"{body_prefix}{body}"
        return combined

    def _get_issue_title_by_id(self, original_id: int) -> str:
        """Get an issue title by its original ID."""
        for issue_data in self.config.get("issues", []):
            if issue_data.get("id") == original_id:
                return issue_data.get("title", "Unknown Issue")
        return "Unknown Issue"

    def _update_issue_relationships(
        self, issue_data: Dict[str, Any], new_issue_id: int
    ):
        """Update issue relationships and additional fields."""
        if self.dry_run:
            print(f"Would update relationships for issue #{new_issue_id}")
            return

        # Get all child issues for this parent
        if not issue_data.get("parent_id"):  # This is a parent issue
            # Find all child issues
            child_issues = []
            for potential_child in self.config.get("issues", []):
                if potential_child.get("parent_id") == issue_data.get("id"):
                    new_child_id = self.issue_map.get(potential_child.get("id"))
                    if new_child_id:
                        child_issues.append(new_child_id)

            if child_issues:
                # Update the parent issue to list all child issues
                issue = self.repo.get_issue(new_issue_id)
                body = issue.body

                # Add checklist items for each child
                checklist = "\n"
                for child_id in child_issues:
                    child_issue = self.repo.get_issue(child_id)
                    checklist += f"- [ ] #{child_id}: {child_issue.title}\n"

                # Find the sub-issues section and add the checklist
                if "## Sub-issues" in body:
                    parts = body.split(
                        "<!-- DO NOT EDIT BELOW THIS LINE - Sub-issues will be listed automatically -->"
                    )
                    updated_body = f"{parts[0]}<!-- DO NOT EDIT BELOW THIS LINE - Sub-issues will be listed automatically -->{checklist}"

                    try:
                        issue.edit(body=updated_body)
                        print(
                            f"Updated parent issue #{new_issue_id} with {len(child_issues)} sub-issues"
                        )
                    except GithubException as e:
                        print(f"Error updating parent issue #{new_issue_id}: {e}")

        # Handle dependencies
        dependencies = issue_data.get("depends_on", [])
        if dependencies:
            issue = self.repo.get_issue(new_issue_id)
            body = issue.body

            # Add dependency information
            for dep_id in dependencies:
                new_dep_id = self.issue_map.get(dep_id)
                if new_dep_id:
                    if f"Blocked by #{new_dep_id}" not in body:
                        body += f"\n\nBlocked by #{new_dep_id}"

            try:
                issue.edit(body=body)
                print(f"Updated issue #{new_issue_id} with dependencies")
            except GithubException as e:
                print(f"Error updating dependencies for issue #{new_issue_id}: {e}")

    @staticmethod
    def generate_config_template(output_file: str):
        """Generate a template configuration file."""
        template = {
            "repository": {"name": "lluminary", "full_name": "PimpMyNines/LLuMinary"},
            "delete_existing_labels": False,
            "delete_existing_milestones": False,
            "labels": [
                {
                    "name": "bug",
                    "color": "d73a4a",
                    "description": "Something isn't working",
                },
                {
                    "name": "documentation",
                    "color": "0075ca",
                    "description": "Improvements or additions to documentation",
                },
                {
                    "name": "enhancement",
                    "color": "a2eeef",
                    "description": "New feature or request",
                },
                {
                    "name": "good first issue",
                    "color": "7057ff",
                    "description": "Good for newcomers",
                },
                {
                    "name": "help wanted",
                    "color": "008672",
                    "description": "Extra attention is needed",
                },
                {
                    "name": "priority:p0",
                    "color": "b60205",
                    "description": "Highest priority, must be fixed immediately",
                },
                {
                    "name": "priority:p1",
                    "color": "d93f0b",
                    "description": "High priority, should be addressed soon",
                },
                {
                    "name": "priority:p2",
                    "color": "fbca04",
                    "description": "Medium priority, address when convenient",
                },
                {
                    "name": "priority:p3",
                    "color": "c5def5",
                    "description": "Low priority, nice to have",
                },
                {
                    "name": "parent-issue",
                    "color": "0366d6",
                    "description": "Parent issue with sub-issues",
                },
                {
                    "name": "sub-issue",
                    "color": "5319e7",
                    "description": "Part of a larger parent issue",
                },
                {
                    "name": "blocked",
                    "color": "b60205",
                    "description": "Blocked by another issue",
                },
            ],
            "milestones": [
                {
                    "title": "CI Infrastructure Stabilization",
                    "description": "Fix critical CI infrastructure issues to enable reliable testing",
                    "due_on": (datetime.now() + timedelta(days=14)).isoformat(),
                    "state": "open",
                },
                {
                    "title": "Type System Overhaul",
                    "description": "Implement unified type definitions across providers",
                    "due_on": (datetime.now() + timedelta(days=30)).isoformat(),
                    "state": "open",
                },
            ],
            "project_board": {
                "name": "LLuMinary Development",
                "description": "Project tracking for LLuMinary development",
                "use_existing": False,
                "columns": [
                    {"name": "To Do"},
                    {"name": "In Progress"},
                    {"name": "Review"},
                    {"name": "Done"},
                ],
            },
            "issues": [
                {
                    "id": 1,  # Original ID (for tracking relationships)
                    "title": "Fix Dockerfile.matrix handling in GitHub Actions workflow",
                    "body": "There's an issue with how Dockerfile.matrix is being handled in our GitHub Actions workflow.\n\nWe need to fix the matrix generation and ensure it's being properly referenced.",
                    "type": "Task",
                    "story_points": 8,
                    "priority": "P0",
                    "labels": [
                        "priority:p0",
                        "area:infrastructure",
                        "technical-debt",
                        "parent-issue",
                    ],
                    "milestone": "CI Infrastructure Stabilization",
                    "assignees": [],
                    "depends_on": [],
                },
                {
                    "id": 2,
                    "title": "Analyze current Dockerfile.matrix generation issues",
                    "body": "Analyze the root causes of the current issues with Dockerfile.matrix generation in the CI pipeline.\n\n## Acceptance Criteria\n- Document all failing scenarios and root causes\n- Identify specific workflow steps that are failing\n- Analysis includes recommendations for fixes",
                    "type": "Task",
                    "story_points": 2,
                    "priority": "P0",
                    "labels": ["area:infrastructure", "size:small", "sub-issue"],
                    "milestone": "CI Infrastructure Stabilization",
                    "assignees": [],
                    "parent_id": 1,
                },
                {
                    "id": 3,
                    "title": "Implement fixes for Dockerfile.matrix generation",
                    "body": "Fix the identified issues with Dockerfile.matrix generation to ensure it's created correctly.\n\n## Acceptance Criteria\n- Dockerfile.matrix is generated correctly\n- Generation process is reliable and consistent\n- Failures are properly handled with clear error messages",
                    "type": "Task",
                    "story_points": 4,
                    "priority": "P0",
                    "labels": ["area:infrastructure", "size:medium", "sub-issue"],
                    "milestone": "CI Infrastructure Stabilization",
                    "assignees": [],
                    "parent_id": 1,
                },
            ],
            "config_files": {
                ".github/ISSUE_TEMPLATE/bug_report.yml": """name: Bug Report
description: Report a bug in LLuminary
title: "[Bug]: "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: input
    id: version
    attributes:
      label: LLuminary Version
      description: What version of LLuminary are you using?
      placeholder: "0.5.0"
    validations:
      required: true
  - type: dropdown
    id: provider
    attributes:
      label: LLM Provider
      description: Which provider is affected by this issue?
      options:
        - OpenAI
        - Anthropic
        - Google
        - Bedrock
        - Cohere
        - All/Multiple
    validations:
      required: true
  - type: textarea
    id: description
    attributes:
      label: Description
      description: A clear description of the bug
      placeholder: What happened? What did you expect to happen?
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Reproduction Steps
      description: Steps to reproduce the behavior
      placeholder: |
        1. Initialize client with '...'
        2. Call method '...'
        3. See error
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Relevant Logs
      description: Please copy and paste any relevant logs
      render: shell""",
                ".github/PULL_REQUEST_TEMPLATE.md": """## Description
<!-- Provide a brief description of the changes in this PR -->

## Related Issue
<!-- Link to any related issues using #issue_number -->
Fixes #

## Type of Change
<!-- Mark the appropriate option with an "x" -->
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test improvement
- [ ] CI/CD improvement

## Testing
<!-- Describe the testing you've done -->
- [ ] Added new tests for new functionality
- [ ] All tests pass locally
- [ ] Verified manually that the change works as expected

## Checklist
<!-- Mark the appropriate options with an "x" -->
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have updated the CHANGELOG.md file""",
            },
        }

        with open(output_file, "w") as f:
            json.dump(template, f, indent=2)

        print(f"Template configuration saved to: {output_file}")
        print("Customize this template with your project details and issues.")


def main():
    """Parse command-line arguments and execute the script."""
    parser = argparse.ArgumentParser(description="Rebuild a GitHub project structure")
    parser.add_argument("--config", required=False, help="Path to JSON config file")
    parser.add_argument("--token", help="GitHub API token")
    parser.add_argument("--repo-path", help="Path to local repository")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't make actual changes"
    )
    parser.add_argument(
        "--create-template", help="Generate template config to specified file"
    )

    args = parser.parse_args()

    # Generate template config if requested
    if args.create_template:
        GitHubProjectRebuilder.generate_config_template(args.create_template)
        return 0

    # Check for required dependencies
    if not HAS_GITHUB_DEPS and args.token:
        print("Warning: Required packages for GitHub API are not installed.")
        print("Run: pip install PyGithub pyyaml requests")
        return 1

    # Check for required arguments
    if not args.config:
        parser.print_help()
        print("\nError: --config is required unless --create-template is specified.")
        return 1

    # Create and run the rebuilder
    rebuilder = GitHubProjectRebuilder(
        config_file=args.config,
        token=args.token,
        repo_path=args.repo_path,
        dry_run=args.dry_run,
    )
    rebuilder.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
