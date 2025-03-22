"""GitHub roadmap management module."""

from typing import Dict, List, Any, Optional, Union
import datetime
import github
from github import Github
from github_project_management.auth.github_auth import GitHubAuth

class RoadmapManager:
    """Manage GitHub roadmaps via milestones."""

    def __init__(self, auth: GitHubAuth):
        """Initialize roadmap manager.

        Args:
            auth: GitHub authentication instance
        """
        self.auth = auth
        self.github = auth.client

    def create_milestone(self, 
                       repo: str, 
                       title: str, 
                       due_date: Optional[str] = None,
                       description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new milestone for roadmap.

        Args:
            repo: Repository name in format "owner/repo"
            title: Milestone title
            due_date: Due date in YYYY-MM-DD format
            description: Milestone description

        Returns:
            Dictionary with milestone information
        """
        repository = self.github.get_repo(repo)
        
        # Parse due date if provided
        due_on = None
        if due_date:
            try:
                due_on = datetime.datetime.strptime(due_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Invalid due date format: {due_date}. Use YYYY-MM-DD.")
        
        # Create the milestone
        milestone = repository.create_milestone(
            title=title,
            state="open",
            description=description or "",
            due_on=due_on
        )
        
        return {
            "number": milestone.number,
            "title": milestone.title,
            "description": milestone.description,
            "state": milestone.state,
            "due_on": milestone.due_on.isoformat() if milestone.due_on else None,
            "html_url": milestone.html_url
        }

    def get_roadmap(self, repo: str) -> List[Dict[str, Any]]:
        """Get the current roadmap (all milestones).

        Args:
            repo: Repository name in format "owner/repo"

        Returns:
            List of milestones with their information
        """
        repository = self.github.get_repo(repo)
        
        # Get all milestones
        milestones = []
        for milestone in repository.get_milestones(state="all"):
            # Get progress stats
            open_issues = milestone.open_issues
            closed_issues = milestone.closed_issues
            total_issues = open_issues + closed_issues
            completion_percentage = 0
            if total_issues > 0:
                completion_percentage = round((closed_issues / total_issues) * 100)
                
            # Format due date
            due_date = None
            if milestone.due_on:
                due_date = milestone.due_on.strftime("%Y-%m-%d")
                
            milestones.append({
                "number": milestone.number,
                "title": milestone.title,
                "description": milestone.description,
                "state": milestone.state,
                "due_on": due_date,
                "html_url": milestone.html_url,
                "open_issues": open_issues,
                "closed_issues": closed_issues,
                "completion_percentage": completion_percentage
            })
            
        # Sort by due date (None values at the end)
        milestones.sort(key=lambda x: x["due_on"] or "9999-12-31")
        
        return milestones

    def generate_roadmap_report(self, repo: str) -> str:
        """Generate a markdown report of the roadmap progress.

        Args:
            repo: Repository name in format "owner/repo"

        Returns:
            Markdown formatted roadmap report
        """
        roadmap = self.get_roadmap(repo)
        owner, repo_name = repo.split('/')
        
        # Build report
        report = f"# Roadmap Report for {owner}/{repo_name}\n\n"
        
        # Current milestone
        current_milestone = None
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        for milestone in roadmap:
            if milestone["state"] == "open":
                if not milestone["due_on"] or milestone["due_on"] >= today:
                    current_milestone = milestone
                    break
                    
        if current_milestone:
            report += "## Current Milestone\n\n"
            report += f"### {current_milestone['title']}\n\n"
            report += f"Due: {current_milestone['due_on'] or 'No due date'}\n\n"
            report += f"Progress: {current_milestone['completion_percentage']}% complete "  
            report += f"({current_milestone['closed_issues']}/{current_milestone['closed_issues'] + current_milestone['open_issues']} issues closed)\n\n"
            
        # Upcoming milestones
        upcoming = [m for m in roadmap if m["state"] == "open" and m != current_milestone]
        if upcoming:
            report += "## Upcoming Milestones\n\n"
            for milestone in upcoming:
                report += f"### {milestone['title']}\n\n"
                report += f"Due: {milestone['due_on'] or 'No due date'}\n\n"
                report += f"Progress: {milestone['completion_percentage']}% complete "  
                report += f"({milestone['closed_issues']}/{milestone['closed_issues'] + milestone['open_issues']} issues closed)\n\n"
                
        # Completed milestones
        completed = [m for m in roadmap if m["state"] == "closed"]
        if completed:
            report += "## Completed Milestones\n\n"
            for milestone in completed[:3]:  # Show only the 3 most recent
                report += f"### {milestone['title']}\n\n"
                report += f"Completed on: {milestone['due_on'] or 'Unknown date'}\n\n"
                
        return report
