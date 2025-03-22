#!/bin/bash
# Script to set up GitHub automations and best practices for the LLuminary repository

echo "Setting up GitHub best practices and automations for LLuminary..."

# Create directories if they don't exist
mkdir -p .github/ISSUE_TEMPLATE
mkdir -p .github/workflows

# 1. Create Issue Templates

# Bug Report Template
cat > .github/ISSUE_TEMPLATE/bug_report.yml << 'EOL'
name: Bug Report
description: Report a bug in LLuminary
title: "[Bug]: "
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!
  - type: input
    id: version
    attributes:
      label: LLuminary Version
      description: What version of LLuminary are you using?
      placeholder: "0.5.0"
    validations:
      required: true
  - type: dropdown
    id: provider
    attributes:
      label: LLM Provider
      description: Which provider is affected by this issue?
      options:
        - OpenAI
        - Anthropic
        - Google
        - Bedrock
        - Cohere
        - All/Multiple
    validations:
      required: true
  - type: textarea
    id: description
    attributes:
      label: Description
      description: A clear description of the bug
      placeholder: What happened? What did you expect to happen?
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Reproduction Steps
      description: Steps to reproduce the behavior
      placeholder: |
        1. Initialize client with '...'
        2. Call method '...'
        3. See error
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Relevant Logs
      description: Please copy and paste any relevant logs
      render: shell
EOL

# Feature Request Template
cat > .github/ISSUE_TEMPLATE/feature_request.yml << 'EOL'
name: Feature Request
description: Suggest a feature for LLuminary
title: "[Feature]: "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!
  - type: dropdown
    id: category
    attributes:
      label: Feature Category
      description: What area would this feature affect?
      options:
        - Core Functionality
        - Provider Support
        - New Provider Integration
        - Documentation
        - Examples
        - Testing
        - Other
    validations:
      required: true
  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: What problem would this feature solve?
      placeholder: Describe the challenge you're facing
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: How would you like to see this addressed?
      placeholder: Describe your ideal solution
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Are there any alternatives or workarounds?
      placeholder: Describe any alternatives you've considered
EOL

# Documentation Improvement Template
cat > .github/ISSUE_TEMPLATE/documentation.yml << 'EOL'
name: Documentation Improvement
description: Suggest improvements to LLuminary documentation
title: "[Docs]: "
labels: ["documentation"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for helping improve our documentation!
  - type: dropdown
    id: doc_type
    attributes:
      label: Documentation Type
      description: What type of documentation needs improvement?
      options:
        - API Reference
        - Tutorials
        - Examples
        - README
        - Code Comments
        - Other
    validations:
      required: true
  - type: input
    id: location
    attributes:
      label: Documentation Location
      description: Where is the documentation that needs improvement?
      placeholder: "docs/API_REFERENCE.md or src/lluminary/models/base.py"
  - type: textarea
    id: issue
    attributes:
      label: Issue Description
      description: What's wrong or missing in the documentation?
      placeholder: Describe what needs to be improved
    validations:
      required: true
  - type: textarea
    id: suggestion
    attributes:
      label: Suggested Improvement
      description: How would you improve the documentation?
      placeholder: Provide your suggestion for improvement
EOL

# 2. Create Pull Request Template
cat > .github/PULL_REQUEST_TEMPLATE.md << 'EOL'
## Description
<!-- Provide a brief description of the changes in this PR -->

## Related Issue
<!-- Link to any related issues using #issue_number -->
Fixes #

## Type of Change
<!-- Mark the appropriate option with an "x" -->
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Performance improvement
- [ ] Test improvement
- [ ] CI/CD improvement

## Testing
<!-- Describe the testing you've done -->
- [ ] Added new tests for new functionality
- [ ] All tests pass locally
- [ ] Verified manually that the change works as expected

## Checklist
<!-- Mark the appropriate options with an "x" -->
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have updated the CHANGELOG.md file
EOL

# 3. Create Security Policy
cat > SECURITY.md << 'EOL'
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5.0 | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in LLuminary, please follow these steps:

1. **Do Not** disclose the vulnerability publicly until it has been addressed
2. Email the vulnerability details to [security@pimpmy9s.com](mailto:security@pimpmy9s.com)
3. Include detailed steps to reproduce the issue
4. Allow time for the vulnerability to be addressed before any public disclosure

We take all security vulnerabilities seriously and will respond to your report within 48 hours.
EOL

# 4. Create Dependabot Configuration
cat > .github/dependabot.yml << 'EOL'
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "automated"
    ignore:
      - dependency-name: "*"
        update-types: ["version-update:semver-patch"]

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "automated"
      - "github-actions"
EOL

# 5. Create CodeQL Workflow
cat > .github/workflows/codeql.yml << 'EOL'
name: "CodeQL"

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    - cron: '0 14 * * 3'  # Run weekly on Wednesdays

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      security-events: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: python

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
EOL

# 6. Create Stale Issue Management Workflow
cat > .github/workflows/stale.yml << 'EOL'
name: Mark stale issues and pull requests

on:
  schedule:
    - cron: '0 0 * * *'  # Run daily at midnight

jobs:
  stale:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/stale@v8
      with:
        repo-token: ${{ secrets.GITHUB_TOKEN }}
        stale-issue-message: 'This issue has been automatically marked as stale due to inactivity. It will be closed in 14 days unless there is new activity.'
        stale-pr-message: 'This pull request has been automatically marked as stale due to inactivity. It will be closed in 14 days unless there is new activity.'
        stale-issue-label: 'stale'
        stale-pr-label: 'stale'
        days-before-stale: 60
        days-before-close: 14
        exempt-issue-labels: 'pinned,security,enhancement,bug'
        exempt-pr-labels: 'pinned,security,enhancement,bug'
EOL

# 7. Create PR Labeler Configuration
cat > .github/labeler.yml << 'EOL'
documentation:
  - docs/**/*
  - '**/*.md'

dependencies:
  - requirements.txt
  - setup.py
  - pyproject.toml

core:
  - src/lluminary/handler.py
  - src/lluminary/models/base.py

providers:
  - src/lluminary/models/providers/**/*

tests:
  - tests/**/*

ci-cd:
  - .github/workflows/**/*
  - Dockerfile*
  - Makefile
EOL

# 8. Create PR Labeler Workflow
cat > .github/workflows/labeler.yml << 'EOL'
name: "Pull Request Labeler"
on:
  pull_request_target:
    types: [opened, synchronize, reopened]

jobs:
  triage:
    permissions:
      contents: read
      pull-requests: write
    runs-on: ubuntu-latest
    steps:
    - uses: actions/labeler@v4
      with:
        repo-token: "${{ secrets.GITHUB_TOKEN }}"
EOL

# 9. Create Release Drafter Configuration
cat > .github/release-drafter.yml << 'EOL'
name-template: 'v$RESOLVED_VERSION'
tag-template: 'v$RESOLVED_VERSION'
categories:
  - title: '🚀 Features'
    labels:
      - 'feature'
      - 'enhancement'
  - title: '🐛 Bug Fixes'
    labels:
      - 'fix'
      - 'bugfix'
      - 'bug'
  - title: '🧰 Maintenance'
    labels:
      - 'chore'
      - 'maintenance'
  - title: '📚 Documentation'
    labels:
      - 'documentation'
      - 'docs'
change-template: '- $TITLE @$AUTHOR (#$NUMBER)'
change-title-escapes: '\<*_&'
version-resolver:
  major:
    labels:
      - 'major'
  minor:
    labels:
      - 'minor'
  patch:
    labels:
      - 'patch'
  default: patch
template: |
  ## What's Changed
  $CHANGES
EOL

# 10. Create Release Drafter Workflow
cat > .github/workflows/release-drafter.yml << 'EOL'
name: Release Drafter

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  update_release_draft:
    runs-on: ubuntu-latest
    steps:
      - uses: release-drafter/release-drafter@v5
        with:
          config-name: release-drafter.yml
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
EOL

# 11. Create API Documentation Generation Workflow
cat > .github/workflows/api-docs.yml << 'EOL'
name: API Documentation

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'src/**/*.py'
  workflow_dispatch:

jobs:
  generate-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[docs]"
        pip install pdoc
    - name: Generate API documentation
      run: |
        pdoc --html --output-dir api-docs src/lluminary
    - name: Deploy to GitHub Pages
      uses: JamesIves/github-pages-deploy-action@v4
      with:
        folder: api-docs
        target-folder: api
        branch: gh-pages
EOL

# 12. Create Code of Conduct
cat > CODE_OF_CONDUCT.md << 'EOL'
# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, religion, or sexual identity
and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming,
diverse, inclusive, and healthy community.

## Our Standards

Examples of behavior that contributes to a positive environment for our
community include:

* Demonstrating empathy and kindness toward other people
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Accepting responsibility and apologizing to those affected by our mistakes,
  and learning from the experience
* Focusing on what is best not just for us as individuals, but for the
  overall community

Examples of unacceptable behavior include:

* The use of sexualized language or imagery, and sexual attention or
  advances of any kind
* Trolling, insulting or derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or email
  address, without their explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

## Enforcement Responsibilities

Project maintainers are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in
response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

## Scope

This Code of Conduct applies within all community spaces, and also applies when
an individual is officially representing the community in public spaces.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the project maintainers. All complaints will be reviewed and
investigated promptly and fairly.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant](https://www.contributor-covenant.org),
version 2.0, available at
https://www.contributor-covenant.org/version/2/0/code_of_conduct.html.
EOL

# Create an implementation report
cat > GITHUB_SETUP_REPORT.md << 'EOL'
# GitHub Setup Implementation Report

## Overview

This report documents the implementation of GitHub best practices and automations for the LLuminary repository. The following files and configurations have been created to improve repository management, developer productivity, and community engagement.

## Files Created

### Issue & PR Templates
- `.github/ISSUE_TEMPLATE/bug_report.yml` - Template for bug reports
- `.github/ISSUE_TEMPLATE/feature_request.yml` - Template for feature requests
- `.github/ISSUE_TEMPLATE/documentation.yml` - Template for documentation improvements
- `.github/PULL_REQUEST_TEMPLATE.md` - Template for pull requests

### Security & Dependency Management
- `SECURITY.md` - Security policy and vulnerability reporting
- `.github/dependabot.yml` - Automated dependency updates
- `.github/workflows/codeql.yml` - Security vulnerability scanning

### Repository Maintenance Automations
- `.github/workflows/stale.yml` - Automated stale issue management
- `.github/labeler.yml` - Configuration for automated PR labeling
- `.github/workflows/labeler.yml` - Workflow for automated PR labeling
- `.github/release-drafter.yml` - Configuration for automated release notes
- `.github/workflows/release-drafter.yml` - Workflow for release drafting
- `CODE_OF_CONDUCT.md` - Project code of conduct

### CI/CD Enhancements
- `.github/workflows/api-docs.yml` - Automated API documentation generation

## Next Steps

1. **GitHub Repository Settings**:
   - Enable branch protection rules for `main` and `develop`
   - Set up required status checks for PRs
   - Configure required reviews for PRs

2. **Project Board**:
   - Create a GitHub project board with appropriate columns
   - Configure automation rules for issue/PR movement
   - Set up field tracking for story points and priorities

3. **Labels**:
   - Create standardized labels for issues and PRs
   - Use consistent naming conventions for labels

4. **Documentation**:
   - Update CONTRIBUTING.md to reference new templates and workflows
   - Add information about project board and issue tracking

5. **Team Onboarding**:
   - Inform team members about new processes
   - Provide training on using issue templates and PR workflows

## Benefits

The implemented automations and best practices provide several benefits:

1. **Improved Issue Quality** - Standardized templates ensure complete information
2. **Enhanced Security** - Automated vulnerability scanning and dependency updates
3. **Reduced Maintenance Overhead** - Automated issue management and labeling
4. **Better Release Management** - Automated release notes and versioning
5. **Improved Documentation** - Automated API documentation generation
6. **Clearer Contribution Guidelines** - Code of conduct and PR templates

These improvements will help the LLuminary project maintain high code quality, improve contributor experience, and ensure long-term sustainability.
EOL

echo "GitHub setup complete! Please review GITHUB_SETUP_REPORT.md for details."
echo "Remember to commit these changes to your repository."
