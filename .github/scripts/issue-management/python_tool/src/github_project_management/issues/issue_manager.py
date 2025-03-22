"""GitHub issue management module."""

from typing import Dict, List, Any, Optional, Union
import github
from github import Github
from github_project_management.auth.github_auth import GitHubAuth

class IssueManager:
    """Manage GitHub issues and sub-issues."""

    def __init__(self, auth: GitHubAuth):
        """Initialize issue manager.

        Args:
            auth: GitHub authentication instance
        """
        self.auth = auth
        self.github = auth.client

    def create_issue(
        self,
        repo: str,
        title: str,
        body: str = "",
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        milestone: Optional[int] = None,
        parent_issue: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new GitHub issue.

        Args:
            repo: Repository name in format "owner/repo"
            title: Issue title
            body: Issue body/description
            labels: List of labels to apply
            assignees: List of users to assign
            milestone: Milestone ID
            parent_issue: Parent issue number for creating sub-issues

        Returns:
            Dictionary with issue information
        """
        repository = self.github.get_repo(repo)
        
        # Create the issue
        milestone_obj = repository.get_milestone(milestone) if milestone else None
        issue = repository.create_issue(
            title=title,
            body=body,
            labels=labels or [],
            assignees=assignees or [],
            milestone=milestone_obj
        )
        
        # If parent_issue is provided, create sub-issue relationship
        if parent_issue:
            self._create_sub_issue_relationship(repository, parent_issue, issue.number)
            
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "labels": [label.name for label in issue.labels],
            "assignees": [assignee.login for assignee in issue.assignees],
            "milestone": issue.milestone.title if issue.milestone else None,
            "html_url": issue.html_url,
            "parent_issue": parent_issue
        }

    def _create_sub_issue_relationship(self, repo: github.Repository.Repository, parent_id: int, child_id: int) -> None:
        """Create a parent-child relationship between issues using the sub-issues API.

        Args:
            repo: GitHub repository object
            parent_id: Parent issue number
            child_id: Child issue number
        """
        # This is a placeholder for the GitHub Sub-issues API
        # When GitHub provides a public API for sub-issues, this method would implement it
        # For now, we'll use a workaround by updating the parent issue description
        # to include a reference to the child issue
        
        parent_issue = repo.get_issue(parent_id)
        child_issue = repo.get_issue(child_id)
        
        # Update parent issue body to include reference to child
        new_body = parent_issue.body or ""
        if "## Sub-issues" not in new_body:
            new_body += "\n\n## Sub-issues\n"
            
        new_body += f"\n- #{child_id}: {child_issue.title}"
        parent_issue.edit(body=new_body)
        
        # Update child issue to reference parent
        child_body = child_issue.body or ""
        if not child_body.startswith(f"Parent: #{parent_id}"):
            child_body = f"Parent: #{parent_id}\n\n{child_body}"
            child_issue.edit(body=child_body)

    def convert_tasks_to_issues(
        self, 
        repo: str,
        issue_number: int,
        labels: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert tasks in an issue's description to proper sub-issues.

        Args:
            repo: Repository name in format "owner/repo"
            issue_number: The issue number containing tasks
            labels: Optional labels to apply to created sub-issues

        Returns:
            List of created sub-issues
        """
        repository = self.github.get_repo(repo)
        issue = repository.get_issue(issue_number)
        body = issue.body or ""
        
        # Find task items in the format: - [ ] Task description
        import re
        task_pattern = r'- \[ \] (.+)$'
        tasks = re.findall(task_pattern, body, re.MULTILINE)
        
        created_issues = []
        for task in tasks:
            # Create sub-issue for each task
            sub_issue = self.create_issue(
                repo=repo,
                title=task,
                body=f"Created from task in #{issue_number}",
                labels=labels,
                parent_issue=issue_number
            )
            created_issues.append(sub_issue)
            
            # Replace task with link to issue
            issue_link = f"- [ ] #{sub_issue['number']} {task}"
            body = body.replace(f"- [ ] {task}", issue_link)
            
        # Update the parent issue with links to sub-issues
        issue.edit(body=body)
        
        return created_issues
