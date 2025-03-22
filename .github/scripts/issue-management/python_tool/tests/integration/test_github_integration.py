"""Integration tests for GitHub interactions.

Note: These tests require a valid GitHub token and internet connection.
""" 

import os
import pytest
from unittest.mock import patch

from src.github_project_management.auth.github_auth import GitHubAuth
from src.github_project_management.issues.issue_manager import IssueManager
from src.github_project_management.projects.project_manager import ProjectManager
from src.github_project_management.utils.config import Config

# Skip tests if no GitHub token is available
pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_TOKEN") is None,
    reason="GITHUB_TOKEN environment variable not set"
)

@pytest.fixture
def auth():
    """Create real GitHub authentication."""
    config = Config()
    return GitHubAuth(config)

def test_authentication(auth):
    """Test authentication to GitHub API."""
    # Simply verifying that get_user doesn't raise an exception
    user = auth.get_user()
    assert user is not None
    assert user.login is not None

def test_create_and_resolve_issue(auth):
    """Test creating and then closing an issue.
    
    Note: This test creates a real issue in the configured repository.
    """
    issue_manager = IssueManager(auth)
    
    # Create a unique test issue title
    import uuid
    test_title = f"Test Issue {uuid.uuid4()}"
    
    # Get repository name from config or default to a test repo
    config = Config()
    test_repo = config.get("defaults.repository", "owner/test-repo")
    
    # Skip test if no repo is configured
    if test_repo == "owner/test-repo":
        pytest.skip("No test repository configured")
        
    # Create the issue
    issue = issue_manager.create_issue(
        repo=test_repo,
        title=test_title,
        body="This is a test issue created by automated tests. It will be closed immediately.",
        labels=["test"]
    )
    
    # Verify issue was created
    assert issue["number"] > 0
    assert issue["title"] == test_title
    
    # Clean up - close the issue
    repo = auth.client.get_repo(test_repo)
    gh_issue = repo.get_issue(issue["number"])
    gh_issue.edit(state="closed")
