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

The new `matrix-docker-tests.yml` workflow provides enhanced testing capabilities:

### Key Features

- **Multi-Python Version Testing**: Tests across Python 3.8, 3.9, 3.10, and 3.11 in separate jobs
- **Provider-Specific Testing**: Dedicated test jobs for each LLM provider (OpenAI, Anthropic, etc.)
- **Conditional Execution**: Provider tests only run when relevant files change
- **Performance Benchmarking**: Compares performance metrics between base branch and PR
- **Comprehensive Reports**: Adds detailed test results as PR comments

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

## Adding New Workflows

When adding a new workflow:

1. Follow the naming convention of existing workflows
2. Include comprehensive comments for each job and step
3. Add path exclusions for documentation and non-code files
4. Implement proper caching for performance
5. Update this README with details of the new workflow
