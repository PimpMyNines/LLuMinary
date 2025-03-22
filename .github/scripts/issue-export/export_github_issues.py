#!/usr/bin/env python3
"""
GitHub Issue Exporter

This script exports all GitHub issues from a repository to a JSON format
compatible with the rebuild_github_project.py script. It captures all issue metadata,
relationships, labels, and other fields needed for a complete rebuild.
"""

import argparse
import json
import re
import subprocess
import sys
from typing import Dict, List, Optional

try:
    from github import Github, GithubException

    HAS_GITHUB_DEPS = True
except ImportError:
    HAS_GITHUB_DEPS = False


def extract_issue_type(body: str) -> str:
    """Extract the issue type from the body."""
    match = re.search(r"\*\*Issue Type\*\*:\s*(\w+)", body)
    if match:
        return match.group(1)

    # Default type based on common labels
    if "bug" in body.lower():
        return "Bug"
    elif "feature" in body.lower() or "enhancement" in body.lower():
        return "Feature"
    else:
        return "Task"


def extract_story_points(body: str) -> Optional[int]:
    """Extract story points from the body."""
    match = re.search(r"\*\*Story Points\*\*:\s*(\d+)", body)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    # Try to find story points in text
    match = re.search(r"story points:?\s*(\d+)", body, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass

    return None


def extract_priority(body: str) -> Optional[str]:
    """Extract priority from the body."""
    match = re.search(r"\*\*Priority\*\*:\s*(P\d)", body)
    if match:
        return match.group(1)

    return None


def extract_parent_id(body: str, issues_map: Dict[int, Dict]) -> Optional[int]:
    """Extract parent issue id from the body."""
    match = re.search(r"Part of #(\d+)", body)
    if match:
        issue_num = int(match.group(1))
        # Map real GitHub issue number to our sequential internal ID
        for internal_id, issue_data in issues_map.items():
            if issue_data.get("github_number") == issue_num:
                return internal_id

    return None


def extract_depends_on(body: str, issues_map: Dict[int, Dict]) -> List[int]:
    """Extract dependencies from the body."""
    depends_on = []

    # Match all instances of "Blocked by #X" or "Depends on #X"
    matches = re.finditer(r"(?:Blocked|Depends) (?:by|on) #(\d+)", body, re.IGNORECASE)
    for match in matches:
        try:
            issue_num = int(match.group(1))
            # Map real GitHub issue number to our sequential internal ID
            for internal_id, issue_data in issues_map.items():
                if issue_data.get("github_number") == issue_num:
                    depends_on.append(internal_id)
        except ValueError:
            continue

    return depends_on


def export_github_issues(
    repo_name: str, token: Optional[str] = None, output_file: str = "github_issues.json"
) -> None:
    """
    Export GitHub issues to a JSON format compatible with rebuild_github_project.py.

    Args:
        repo_name: Repository name in format "owner/repo"
        token: GitHub API token (optional if using gh CLI)
        output_file: Path to output JSON file
    """
    issues_data = {
        "repository": {"name": repo_name.split("/")[-1], "full_name": repo_name},
        "delete_existing_labels": False,
        "delete_existing_milestones": False,
        "labels": [],
        "milestones": [],
        "issues": [],
    }

    # Keep track of all issues with GitHub issue number as key
    issues_map = {}

    # Get all issues - try GitHub API first, fall back to gh CLI
    if token and HAS_GITHUB_DEPS:
        print(f"Using GitHub API to fetch issues from {repo_name}")

        # Initialize GitHub client
        g = Github(token)
        repo = g.get_repo(repo_name)

        # Get labels
        print("Fetching labels...")
        for label in repo.get_labels():
            label_data = {
                "name": label.name,
                "color": label.color,
                "description": label.description or "",
            }
            issues_data["labels"].append(label_data)

        # Get milestones
        print("Fetching milestones...")
        for milestone in repo.get_milestones(state="all"):
            milestone_data = {
                "title": milestone.title,
                "description": milestone.description or "",
                "state": milestone.state,
            }
            if milestone.due_on:
                milestone_data["due_on"] = milestone.due_on.isoformat()

            issues_data["milestones"].append(milestone_data)

        # Get issues
        print("Fetching issues...")
        for issue in repo.get_issues(state="all"):
            # Skip pull requests
            if issue.pull_request:
                continue

            # Basic issue data
            internal_id = len(issues_map) + 1
            issues_map[internal_id] = {
                "github_number": issue.number,
                "title": issue.title,
            }

            issue_data = {
                "id": internal_id,
                "github_number": issue.number,  # Keep original GitHub number for reference
                "title": issue.title,
                "body": issue.body or "",
                "labels": [label.name for label in issue.labels],
                "state": issue.state,
            }

            # Handle assignees
            if issue.assignees:
                issue_data["assignees"] = [
                    assignee.login for assignee in issue.assignees
                ]

            # Handle milestone
            if issue.milestone:
                issue_data["milestone"] = issue.milestone.title

            # Add to issues list
            issues_data["issues"].append(issue_data)
    else:
        # Fall back to using GitHub CLI
        print(f"Using GitHub CLI to fetch issues from {repo_name}")

        # Get labels
        print("Fetching labels...")
        try:
            result = subprocess.run(
                [
                    "gh",
                    "label",
                    "list",
                    "--repo",
                    repo_name,
                    "--json",
                    "name,color,description",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            labels = json.loads(result.stdout)
            issues_data["labels"] = labels
        except subprocess.CalledProcessError as e:
            print(f"Error fetching labels: {e}")

        # Get milestones
        print("Fetching milestones...")
        try:
            result = subprocess.run(
                [
                    "gh",
                    "milestone",
                    "list",
                    "--repo",
                    repo_name,
                    "--json",
                    "title,description,state,dueOn",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            milestones = json.loads(result.stdout)
            issues_data["milestones"] = [
                {
                    "title": m["title"],
                    "description": m["description"] or "",
                    "state": m["state"],
                    "due_on": m["dueOn"] if "dueOn" in m else None,
                }
                for m in milestones
            ]
        except subprocess.CalledProcessError as e:
            print(f"Error fetching milestones: {e}")

        # Get issues
        print("Fetching issues...")
        try:
            # First, get list of all issue numbers
            result = subprocess.run(
                [
                    "gh",
                    "issue",
                    "list",
                    "--repo",
                    repo_name,
                    "--state",
                    "all",
                    "--json",
                    "number,title",
                    "--limit",
                    "1000",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            issues_list = json.loads(result.stdout)

            # For each issue, get details
            for idx, issue_summary in enumerate(issues_list):
                print(
                    f"Processing issue {idx+1}/{len(issues_list)}: #{issue_summary['number']} {issue_summary['title']}"
                )

                # Store in issues map for relationship tracking
                internal_id = idx + 1
                issues_map[internal_id] = {
                    "github_number": issue_summary["number"],
                    "title": issue_summary["title"],
                }

                # Get detailed issue info
                try:
                    result = subprocess.run(
                        [
                            "gh",
                            "issue",
                            "view",
                            str(issue_summary["number"]),
                            "--repo",
                            repo_name,
                            "--json",
                            "number,title,body,labels,assignees,milestone,state",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    issue = json.loads(result.stdout)

                    issue_data = {
                        "id": internal_id,
                        "github_number": issue["number"],
                        "title": issue["title"],
                        "body": issue["body"] or "",
                        "labels": [label["name"] for label in issue["labels"]],
                        "state": issue["state"],
                    }

                    # Handle assignees
                    if issue["assignees"]:
                        issue_data["assignees"] = [
                            assignee["login"] for assignee in issue["assignees"]
                        ]

                    # Handle milestone
                    if issue["milestone"]:
                        issue_data["milestone"] = issue["milestone"]["title"]

                    # Add to issues list
                    issues_data["issues"].append(issue_data)

                except subprocess.CalledProcessError as e:
                    print(f"Error fetching issue #{issue_summary['number']}: {e}")

        except subprocess.CalledProcessError as e:
            print(f"Error fetching issues: {e}")

    # Second pass: extract additional data and relationships
    for issue in issues_data["issues"]:
        # Extract issue type
        issue["type"] = extract_issue_type(issue["body"])

        # Extract story points
        story_points = extract_story_points(issue["body"])
        if story_points is not None:
            issue["story_points"] = story_points

        # Extract priority
        priority = extract_priority(issue["body"])
        if priority:
            issue["priority"] = priority

        # Check for parent
        parent_id = extract_parent_id(issue["body"], issues_map)
        if parent_id:
            issue["parent_id"] = parent_id

        # Extract dependencies
        depends_on = extract_depends_on(issue["body"], issues_map)
        if depends_on:
            issue["depends_on"] = depends_on
        else:
            issue["depends_on"] = []

        # Clean issue body (remove Type, Story Points, etc.)
        body = issue["body"]
        body = re.sub(r"\*\*Issue Type\*\*:.*\n\n", "", body)
        body = re.sub(r"\*\*Story Points\*\*:.*\n\n", "", body)
        body = re.sub(r"\*\*Priority\*\*:.*\n\n", "", body)

        # Remove "Parent Issue" and "Sub-issue" sections
        body = re.sub(
            r"# Parent Issue.*## Sub-issues\s+<!-- DO NOT EDIT BELOW THIS LINE.*?-->",
            "",
            body,
            flags=re.DOTALL,
        )
        body = re.sub(
            r"# Sub-issue\s+Part of #\d+:.*?\n\n---\s+\n\n", "", body, flags=re.DOTALL
        )

        # Remove dependency markers (they'll be recreated)
        body = re.sub(
            r"(?:Blocked|Depends) (?:by|on) #\d+\s*", "", body, flags=re.IGNORECASE
        )

        # Clean up extra newlines
        body = re.sub(r"\n{3,}", "\n\n", body)
        issue["body"] = body.strip()

    # Set parent_issue=True on all parent issues and remove github_number
    for issue in issues_data["issues"]:
        # Check if any issue has this as a parent
        is_parent = any(
            sub_issue.get("parent_id") == issue["id"]
            for sub_issue in issues_data["issues"]
        )
        if is_parent:
            issue["parent_issue"] = True

        # Remove github_number (not needed for import)
        github_number = issue.pop("github_number", None)

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(issues_data, f, indent=2)

    print(f"Exported {len(issues_data['issues'])} issues to {output_file}")
    print(
        f"Found {len(issues_data['labels'])} labels and {len(issues_data['milestones'])} milestones"
    )


def main():
    """Parse command-line arguments and execute the script."""
    parser = argparse.ArgumentParser(description="Export GitHub issues to JSON format")
    parser.add_argument(
        "--repo", required=True, help="Repository name in format owner/repo"
    )
    parser.add_argument("--token", help="GitHub API token")
    parser.add_argument(
        "--output", default="github_issues.json", help="Output file path"
    )

    args = parser.parse_args()

    # Check for required dependencies if using token
    if args.token and not HAS_GITHUB_DEPS:
        print("Warning: Required packages for GitHub API are not installed.")
        print("Run: pip install PyGithub")
        print("Falling back to GitHub CLI...")

    export_github_issues(args.repo, args.token, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
