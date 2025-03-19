#!/usr/bin/env python3
"""Command-line interface for GitHub Project Management Tool."""

import click
import os
import sys

from github_project_management.auth.github_auth import GitHubAuth
from github_project_management.issues.issue_manager import IssueManager
from github_project_management.projects.project_manager import ProjectManager
from github_project_management.roadmap.roadmap_manager import RoadmapManager
from github_project_management.utils.config import Config

@click.group()
def main():
    """GitHub Project Management Tool.
    
    A Python-based tool for managing GitHub projects, issues, sub-issues, and roadmaps.
    """
    pass

@main.group()
def issues():
    """Manage GitHub issues and sub-issues."""
    pass

@issues.command('create')
@click.option('--title', '-t', required=True, help='Issue title')
@click.option('--body', '-b', default='', help='Issue body/description')
@click.option('--repo', '-r', required=True, help='Repository in format owner/repo')
@click.option('--labels', '-l', multiple=True, help='Labels to apply to the issue')
@click.option('--assignees', '-a', multiple=True, help='Users to assign to the issue')
@click.option('--milestone', '-m', type=int, help='Milestone ID')
@click.option('--parent', '-p', type=int, help='Parent issue number for creating sub-issues')
def create_issue(title, body, repo, labels, assignees, milestone, parent):
    """Create a new GitHub issue."""
    config = Config()
    auth = GitHubAuth(config)
    issue_manager = IssueManager(auth)
    
    try:
        result = issue_manager.create_issue(
            repo=repo,
            title=title,
            body=body,
            labels=labels,
            assignees=assignees,
            milestone=milestone,
            parent_issue=parent
        )
        click.echo(f"Created issue #{result['number']}: {result['title']}")
        click.echo(f"URL: {result['html_url']}")
    except Exception as e:
        click.echo(f"Error creating issue: {str(e)}", err=True)
        sys.exit(1)

@main.group()
def projects():
    """Manage GitHub projects."""
    pass

@projects.command('create')
@click.option('--name', '-n', required=True, help='Project name')
@click.option('--body', '-b', default='', help='Project description')
@click.option('--org', '-o', help='Organization (if not user project)')
@click.option('--repo', '-r', help='Repository (if repo project)')
@click.option('--template', '-t', type=click.Choice(['basic', 'advanced']), default='basic',
              help='Project template type')
def create_project(name, body, org, repo, template):
    """Create a new GitHub project."""
    config = Config()
    auth = GitHubAuth(config)
    project_manager = ProjectManager(auth)
    
    try:
        result = project_manager.create_project(
            name=name,
            body=body,
            org=org,
            repo=repo,
            template=template
        )
        click.echo(f"Created project: {result['name']}")
        click.echo(f"URL: {result['html_url']}")
    except Exception as e:
        click.echo(f"Error creating project: {str(e)}", err=True)
        sys.exit(1)

@main.group()
def roadmap():
    """Manage roadmap and milestones."""
    pass

@roadmap.command('create')
@click.option('--title', '-t', required=True, help='Milestone title')
@click.option('--due-date', '-d', help='Due date (YYYY-MM-DD format)')
@click.option('--description', help='Milestone description')
@click.option('--repo', '-r', required=True, help='Repository in format owner/repo')
def create_milestone(title, due_date, description, repo):
    """Create a new milestone for roadmap."""
    config = Config()
    auth = GitHubAuth(config)
    roadmap_manager = RoadmapManager(auth)
    
    try:
        result = roadmap_manager.create_milestone(
            repo=repo,
            title=title,
            due_date=due_date,
            description=description
        )
        click.echo(f"Created milestone: {result['title']}")
        click.echo(f"Due date: {result['due_on']}")
        click.echo(f"URL: {result['html_url']}")
    except Exception as e:
        click.echo(f"Error creating milestone: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
