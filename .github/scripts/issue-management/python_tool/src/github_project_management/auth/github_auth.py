"""GitHub authentication module."""

import os
from typing import Optional, Dict, Any
import github
from github import Github
from github_project_management.utils.config import Config

class GitHubAuth:
    """GitHub authentication handler.

    This class handles authentication with GitHub using either
    personal access tokens or GitHub Apps.
    """

    def __init__(self, config: Config):
        """Initialize GitHub authentication.

        Args:
            config: Configuration object containing auth settings
        """
        self.config = config
        self._github_client = None
        self._token = None
        self._initialize_auth()

    def _initialize_auth(self) -> None:
        """Initialize GitHub authentication using config settings."""
        # Try environment variable first
        token = os.environ.get('GITHUB_TOKEN')
        
        # If not in env, try from config
        if not token and self.config.has('auth.token'):
            token = self.config.get('auth.token')
            
        # GitHub App authentication
        if not token and self.config.has('auth.app'):
            # GitHub App authentication is more complex and would be
            # implemented here using JWT and installation tokens
            pass
            
        if not token:
            raise ValueError(
                "GitHub token not found. Set GITHUB_TOKEN environment "
                "variable or configure in the auth.token config setting."
            )
            
        self._token = token
        self._github_client = Github(token)

    @property
    def client(self) -> Github:
        """Get the GitHub client.

        Returns:
            Authenticated GitHub client instance
        """
        if not self._github_client:
            raise ValueError("GitHub client has not been initialized")
        return self._github_client

    def get_token(self) -> str:
        """Get the GitHub token.

        Returns:
            The GitHub authentication token
        """
        if not self._token:
            raise ValueError("GitHub token has not been initialized")
        return self._token

    def get_user(self) -> github.NamedUser.NamedUser:
        """Get the authenticated user.

        Returns:
            Authenticated GitHub user
        """
        return self.client.get_user()

    def get_repo(self, repo_name: str) -> github.Repository.Repository:
        """Get a GitHub repository.

        Args:
            repo_name: Repository name in format "owner/repo"

        Returns:
            GitHub repository object
        """
        return self.client.get_repo(repo_name)

    def get_organization(self, org_name: str) -> github.Organization.Organization:
        """Get a GitHub organization.

        Args:
            org_name: Organization name

        Returns:
            GitHub organization object
        """
        return self.client.get_organization(org_name)
