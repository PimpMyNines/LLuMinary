"""Unit tests for issue management module."""

import pytest
from unittest.mock import patch, MagicMock

from src.github_project_management.issues.issue_manager import IssueManager

def test_create_issue(mock_auth, mock_repo):
    """Test creating an issue."""
    # Arrange
    issue_manager = IssueManager(mock_auth)
    mock_auth.client.get_repo.return_value = mock_repo
    
    # Act
    result = issue_manager.create_issue(
        repo="owner/repo",
        title="Test Issue",
        body="Test issue body"
    )
    
    # Assert
    assert result["number"] == 123
    assert result["title"] == "Test Issue"
    mock_repo.create_issue.assert_called_once_with(
        title="Test Issue",
        body="Test issue body",
        labels=[],
        assignees=[],
        milestone=None
    )

def test_create_issue_with_parent(mock_auth, mock_repo):
    """Test creating a sub-issue with parent relationship."""
    # Arrange
    issue_manager = IssueManager(mock_auth)
    mock_auth.client.get_repo.return_value = mock_repo
    
    # Mock _create_sub_issue_relationship method
    issue_manager._create_sub_issue_relationship = MagicMock()
    
    # Act
    result = issue_manager.create_issue(
        repo="owner/repo",
        title="Test Sub-Issue",
        body="Test sub-issue body",
        parent_issue=100
    )
    
    # Assert
    assert result["number"] == 123
    assert result["parent_issue"] == 100
    issue_manager._create_sub_issue_relationship.assert_called_once_with(
        mock_repo, 100, 123
    )

def test_convert_tasks_to_issues(mock_auth, mock_repo):
    """Test converting tasks to sub-issues."""
    # Arrange
    issue_manager = IssueManager(mock_auth)
    mock_auth.client.get_repo.return_value = mock_repo
    
    # Set up issue with tasks
    task_body = "Test issue\n\n- [ ] Task 1\n- [ ] Task 2\n"
    mock_issue = MagicMock()
    mock_issue.body = task_body
    mock_repo.get_issue.return_value = mock_issue
    
    # Mock create_issue to return predictable results
    issue_manager.create_issue = MagicMock()
    issue_manager.create_issue.side_effect = [
        {"number": 124, "title": "Task 1"},
        {"number": 125, "title": "Task 2"}
    ]
    
    # Act
    result = issue_manager.convert_tasks_to_issues(
        repo="owner/repo",
        issue_number=123
    )
    
    # Assert
    assert len(result) == 2
    assert result[0]["number"] == 124
    assert result[1]["number"] == 125
    assert issue_manager.create_issue.call_count == 2
