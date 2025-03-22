# LLuMinary CI/CD Workflows

This directory contains GitHub Actions workflows for continuous integration and deployment of the LLuMinary project.

## Available Workflows

### CI Workflows

1. **ci.yml**: Standard CI pipeline
   - Runs on pushes to main/develop and PRs
   - Matrix testing across Python 3.8-3.11
   - Linting with ruff, black, isort, and mypy
   - Unit testing and coverage reporting

2. **docker-tests.yml**: Containerized test environment
   - Runs tests in a consistent Docker environment
   - Parallelized test execution with pytest-xdist
   - Includes both unit and integration tests
   - Optimized with Docker layer caching

3. **matrix-docker-tests.yml**: Enhanced matrix testing (NEW)
   - Tests across multiple Python versions in Docker containers
   - Specialized provider-specific test jobs
   - Smart conditional execution based on changed files
   - Performance benchmarking with PR comments
   - Comprehensive test summary generation

### Release Workflows

4. **publish.yml**: Package publishing
   - Triggered on releases or manually
   - Builds and validates distribution package
   - Publishes to PyPI with trusted publishing

5. **release.yml**: Release automation
   - Generates release notes
   - Creates GitHub releases
   - Handles versioning

### Supporting Workflows

6. **pr.yml**: PR validation
   - Validates PR title format
   - Checks code formatting
   - Ensures version is bumped for main branch PRs

7. **docs.yml**: Documentation
   - Builds and validates documentation
   - Publishes documentation to GitHub Pages

8. **version.yml**: Version management
   - Checks version consistency
   - Validates version changes

## Using the Matrix Docker Tests Workflow

The enhanced `matrix-docker-tests.yml` workflow provides comprehensive testing capabilities:

### Key Features

- **Multi-Python Version Testing**: Tests across Python 3.8, 3.9, 3.10, and 3.11 in separate jobs
- **Provider-Specific Testing**: Dedicated test jobs for each LLM provider (OpenAI, Anthropic, etc.)
- **Conditional Execution**: Provider tests only run when relevant files change
- **Performance Benchmarking**: Compares performance metrics between base branch and PR
- **Comprehensive Reports**: Adds detailed test results as PR comments

### Recent Improvements

The following issues have been addressed in the matrix-docker-tests workflow:

1. **Fixed Dockerfile.matrix Handling**
   - Added `PYTHON_VERSION` build arg to properly handle Python version selection
   - Updated `docker-build-matrix-cached` Makefile target to use the build arg
   - Modified Dockerfile.matrix to use the build arg, defaulting to Python 3.10 if not specified

2. **Improved Provider Test Conditional Logic**
   - Fixed conditional execution for provider-specific tests
   - Added support for running provider tests based on commit messages
   - Enhanced provider test discovery based on file changes

3. **Enhanced Test Coverage Reporting**
   - Added pytest-cov and pytest-asyncio to all test containers
   - Added documentation about CODECOV_TOKEN requirement
   - Improved coverage reporting with appropriate test flags

### Manual Triggering Options

The workflow can be manually triggered with these options:

1. **Python Versions**: Specify which Python versions to test against (comma-separated list)
2. **Provider Tests**: Choose whether to run provider-specific test suites

To manually trigger:
1. Go to the "Actions" tab in GitHub
2. Select "Matrix Docker Tests"
3. Click "Run workflow"
4. Configure the options and click "Run workflow"

### PR Test Results

When run on a PR, the workflow adds:
1. A test summary comment showing pass/fail status for each category
2. A performance benchmark report comparing test execution times
3. Detailed coverage information in Codecov

## Required Repository Secrets

For the CI workflows to function correctly, the following GitHub repository secrets must be configured:

### CODECOV_TOKEN Configuration (REQUIRED)

The CODECOV_TOKEN is required for uploading test coverage data to Codecov. Without this token, coverage reporting will fail.

To configure CODECOV_TOKEN:

1. Sign up or log in to [codecov.io](https://codecov.io) using your GitHub account
2. Add your repository to Codecov by selecting it from the list
3. Navigate to Repository Settings > General > Repository Upload Token
4. Copy the generated token
5. In your GitHub repository:
   - Go to Settings > Secrets and variables > Actions
   - Click on "New repository secret"
   - Name: `CODECOV_TOKEN`
   - Value: [paste the token copied from Codecov]
   - Click "Add secret"

⚠️ **IMPORTANT**: This token must be added before running the matrix-docker-tests workflow, as both the main test jobs and provider-specific test jobs depend on it for coverage reporting.

### Verification Script

You can verify that your CODECOV_TOKEN is properly configured by running:

```bash
# Check if CODECOV_TOKEN exists in GitHub repository secrets
gh secret list | grep CODECOV_TOKEN

# Or when using GitHub CLI
gh api /repos/OWNER/REPO/actions/secrets | jq '.secrets[] | select(.name=="CODECOV_TOKEN")'
```

If the token is not present, the coverage reporting steps in the workflow will fail with an error message indicating that the CODECOV_TOKEN secret is missing.

## Using CI Workflows Locally

To simulate the CI environment locally:

```bash
# Run basic CI checks (similar to ci.yml)
make check

# Run tests in Docker (similar to docker-tests.yml)
make docker-build
make test-docker-unit
make test-docker-integration

# Run matrix tests for a specific Python version
make docker-create-matrix-file PYTHON_VERSION=3.9
make docker-build-matrix
make test-matrix-python

# Run provider-specific tests
make test-docker-file FILE="tests/unit/test_openai_*.py"
```

## Adding New Workflows

When adding a new workflow:

1. Follow the naming convention of existing workflows
2. Include comprehensive comments for each job and step
3. Add path exclusions for documentation and non-code files
4. Implement proper caching for performance
5. Update this README with details of the new workflow
