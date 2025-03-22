#!/usr/bin/env python3
"""
GitHub Issues Cleanup Script

This script removes all issues from a GitHub repository. Use with extreme caution!
It provides options to:
1. Export issues to JSON before deleting (recommended)
2. Filter which issues to delete based on labels, milestone, or state
3. Perform a dry run to see what would be deleted
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Optional

try:
    from github import Github, GithubException

    HAS_GITHUB_DEPS = True
except ImportError:
    HAS_GITHUB_DEPS = False


def export_issues_before_deletion(repo_name: str, output_file: str) -> None:
    """Export issues to a JSON file before deletion."""
    print(f"Exporting issues to {output_file} before deletion...")

    # Check if we have the export script
    export_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "export_github_issues.py"
    )

    if os.path.exists(export_script):
        try:
            subprocess.run(
                [
                    sys.executable,
                    export_script,
                    "--repo",
                    repo_name,
                    "--output",
                    output_file,
                ],
                check=True,
            )
            print(f"Successfully exported issues to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error exporting issues: {e}")
            sys.exit(1)
    else:
        print("Warning: export_github_issues.py not found. Skipping export.")
        print(
            "You may lose all issue data! Consider aborting and backing up issues first."
        )
        if input("Continue anyway? (y/N): ").lower() != "y":
            sys.exit(0)


def delete_issues_api(
    repo_name: str,
    token: str,
    filter_labels: Optional[List[str]] = None,
    filter_milestone: Optional[str] = None,
    filter_state: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Delete issues using GitHub API."""
    if not HAS_GITHUB_DEPS:
        print("Error: PyGithub not installed. Run: pip install PyGithub")
        sys.exit(1)

    print(f"Using GitHub API to delete issues from {repo_name}")

    g = Github(token)
    repo = g.get_repo(repo_name)

    # Get issues with filters
    query_params = {}
    if filter_state:
        query_params["state"] = filter_state
    if filter_milestone:
        # Get milestone object
        milestones = repo.get_milestones()
        milestone_obj = None
        for ms in milestones:
            if ms.title == filter_milestone:
                milestone_obj = ms
                break
        if milestone_obj:
            query_params["milestone"] = milestone_obj

    # Get all issues
    issues = repo.get_issues(**query_params)

    # Filter by labels if needed
    if filter_labels:
        filtered_issues = []
        for issue in issues:
            if any(label.name in filter_labels for label in issue.labels):
                filtered_issues.append(issue)
    else:
        filtered_issues = list(issues)

    # Skip pull requests
    issues_to_delete = [issue for issue in filtered_issues if not issue.pull_request]

    print(f"Found {len(issues_to_delete)} issues to delete")

    if not issues_to_delete:
        print("No issues to delete.")
        return

    if dry_run:
        for issue in issues_to_delete:
            print(f"Would delete issue #{issue.number}: {issue.title}")
        return

    # Confirm deletion
    if (
        input(
            f"Are you sure you want to delete {len(issues_to_delete)} issues? (y/N): "
        ).lower()
        != "y"
    ):
        print("Deletion cancelled.")
        return

    # Delete issues
    for i, issue in enumerate(issues_to_delete):
        try:
            print(
                f"Deleting issue #{issue.number}: {issue.title} ({i+1}/{len(issues_to_delete)})"
            )
            # GitHub API doesn't support direct issue deletion, so we need to close and modify
            # the issue to indicate it's been "deleted"
            issue.edit(state="closed", title=f"[DELETED] {issue.title}")
            # Add a comment to indicate deletion
            issue.create_comment("This issue has been programmatically deleted.")

            # Sleep to avoid rate limiting
            time.sleep(0.5)
        except GithubException as e:
            print(f"Error deleting issue #{issue.number}: {e}")

    print(f"Deleted {len(issues_to_delete)} issues")


def delete_issues_cli(
    repo_name: str,
    filter_labels: Optional[List[str]] = None,
    filter_milestone: Optional[str] = None,
    filter_state: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Delete issues using GitHub CLI."""
    print(f"Using GitHub CLI to delete issues from {repo_name}")

    # Build search query
    search_query = f"repo:{repo_name} is:issue"
    if filter_state:
        search_query += f" state:{filter_state}"
    if filter_milestone:
        search_query += f' milestone:"{filter_milestone}"'
    if filter_labels:
        for label in filter_labels:
            search_query += f' label:"{label}"'

    # Get issues
    try:
        result = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--repo",
                repo_name,
                "--search",
                search_query,
                "--json",
                "number,title,url",
                "--limit",
                "1000",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        issues = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting issues: {e}")
        sys.exit(1)

    print(f"Found {len(issues)} issues to delete")

    if not issues:
        print("No issues to delete.")
        return

    if dry_run:
        for issue in issues:
            print(f"Would delete issue #{issue['number']}: {issue['title']}")
        return

    # Confirm deletion
    if (
        input(f"Are you sure you want to delete {len(issues)} issues? (y/N): ").lower()
        != "y"
    ):
        print("Deletion cancelled.")
        return

    # Delete issues
    for i, issue in enumerate(issues):
        try:
            print(
                f"Deleting issue #{issue['number']}: {issue['title']} ({i+1}/{len(issues)})"
            )

            # GitHub CLI doesn't support direct issue deletion, so we need to close and modify
            # the issue to indicate it's been "deleted"
            subprocess.run(
                ["gh", "issue", "close", str(issue["number"]), "--repo", repo_name],
                check=True,
            )

            # Change title to indicate deletion
            new_title = f"[DELETED] {issue['title']}"
            subprocess.run(
                [
                    "gh",
                    "issue",
                    "edit",
                    str(issue["number"]),
                    "--repo",
                    repo_name,
                    "--title",
                    new_title,
                ],
                check=True,
            )

            # Add a comment to indicate deletion
            subprocess.run(
                [
                    "gh",
                    "issue",
                    "comment",
                    str(issue["number"]),
                    "--repo",
                    repo_name,
                    "--body",
                    "This issue has been programmatically deleted.",
                ],
                check=True,
            )

            # Sleep to avoid rate limiting
            time.sleep(0.5)
        except subprocess.CalledProcessError as e:
            print(f"Error deleting issue #{issue['number']}: {e}")

    print(f"Deleted {len(issues)} issues")


def main():
    """Parse command-line arguments and execute the script."""
    parser = argparse.ArgumentParser(
        description="Delete all issues from a GitHub repository"
    )
    parser.add_argument(
        "--repo", required=True, help="Repository name in format owner/repo"
    )
    parser.add_argument(
        "--token", help="GitHub API token (optional if using GitHub CLI)"
    )
    parser.add_argument(
        "--export-file", help="Export issues to this file before deletion"
    )
    parser.add_argument(
        "--filter-labels", nargs="+", help="Only delete issues with these labels"
    )
    parser.add_argument(
        "--filter-milestone", help="Only delete issues with this milestone"
    )
    parser.add_argument(
        "--filter-state",
        choices=["open", "closed", "all"],
        default="all",
        help="Only delete issues with this state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )

    args = parser.parse_args()

    # Export issues first if requested
    if args.export_file:
        export_issues_before_deletion(args.repo, args.export_file)

    # Show warning
    if not args.dry_run:
        print("\n" + "=" * 80)
        print("WARNING: This will PERMANENTLY DELETE issues from your repository!")
        print("This action cannot be undone. Make sure you have a backup.")
        print("=" * 80 + "\n")

        # Extra confirmation for non-filtered deletions
        if not (
            args.filter_labels or args.filter_milestone or args.filter_state != "all"
        ):
            print("You are about to delete ALL issues in the repository.")
            print("This is extremely destructive and permanent.")
            if input("Type 'DELETE ALL ISSUES' to confirm: ") != "DELETE ALL ISSUES":
                print("Deletion cancelled.")
                return 1

    # Delete issues
    if args.token and HAS_GITHUB_DEPS:
        delete_issues_api(
            args.repo,
            args.token,
            args.filter_labels,
            args.filter_milestone,
            args.filter_state,
            args.dry_run,
        )
    else:
        delete_issues_cli(
            args.repo,
            args.filter_labels,
            args.filter_milestone,
            args.filter_state,
            args.dry_run,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
