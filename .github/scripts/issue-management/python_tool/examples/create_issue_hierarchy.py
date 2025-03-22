#!/usr/bin/env python3
"""
Example script to create an issue hierarchy (parent and sub-issues).

This example demonstrates how to:
1. Create a parent issue
2. Create multiple sub-issues linked to the parent
3. Update the parent issue with references to sub-issues
"""

import os
import sys

# Add src directory to path for running directly from examples
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.github_project_management.auth.github_auth import GitHubAuth
from src.github_project_management.issues.issue_manager import IssueManager
from src.github_project_management.utils.config import Config

def create_issue_hierarchy(repo, parent_title, sub_issues):
    """Create a parent issue with multiple sub-issues.
    
    Args:
        repo: Repository in format owner/repo
        parent_title: Title of the parent issue
        sub_issues: List of sub-issue titles
    """
    # Initialize configuration and authentication
    config = Config()
    auth = GitHubAuth(config)
    issue_manager = IssueManager(auth)
    
    try:
        # Create parent issue
        parent_body = ("This is a parent issue that will have sub-issues.\n\n"
                      "## Tasks:\n")
        
        for task in sub_issues:
            parent_body += f"- [ ] {task}\n"
            
        print(f"Creating parent issue: {parent_title}")
        parent = issue_manager.create_issue(
            repo=repo,
            title=parent_title,
            body=parent_body,
            labels=["enhancement"]
        )
        print(f"Created parent issue #{parent['number']}")
        
        # Convert tasks to sub-issues
        print("Converting tasks to sub-issues...")
        created_sub_issues = issue_manager.convert_tasks_to_issues(
            repo=repo,
            issue_number=parent['number'],
            labels=["task"]
        )
        
        print(f"Created {len(created_sub_issues)} sub-issues:")
        for issue in created_sub_issues:
            print(f"  - #{issue['number']}: {issue['title']}")
            
        return parent, created_sub_issues
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Repository to create issues in
    REPO = "owner/repo"  # Replace with your repository
    
    # Issue details
    PARENT_TITLE = "Implement new authentication system"
    SUB_ISSUES = [
        "Design authentication database schema",
        "Implement OAuth2 provider integration",
        "Create user registration workflow",
        "Add password reset functionality",
        "Write authentication documentation"
    ]
    
    # Create the issue hierarchy
    create_issue_hierarchy(REPO, PARENT_TITLE, SUB_ISSUES)
