"""Test configuration and fixtures."""

import os
import pytest
from unittest.mock import patch, MagicMock

from src.github_project_management.auth.github_auth import GitHubAuth
from src.github_project_management.utils.config import Config

@pytest.fixture
def mock_config():
    """Mock configuration."""
    config = MagicMock(spec=Config)
    config.get.return_value = "test-value"
    config.has.return_value = True
    return config

@pytest.fixture
def mock_github():
    """Mock GitHub client."""
    github_mock = MagicMock()
    return github_mock

@pytest.fixture
def mock_auth(mock_github):
    """Mock GitHub authentication."""
    auth = MagicMock(spec=GitHubAuth)
    auth.client = mock_github
    auth.get_token.return_value = "mock-token"
    return auth

@pytest.fixture
def mock_repo():
    """Mock GitHub repository."""
    repo = MagicMock()
    issue = MagicMock()
    issue.number = 123
    issue.title = "Test Issue"
    issue.body = "Test issue body"
    issue.html_url = "https://github.com/owner/repo/issues/123"
    issue.labels = []
    issue.assignees = []
    issue.milestone = None
    
    repo.create_issue.return_value = issue
    repo.get_issue.return_value = issue
    
    return repo
