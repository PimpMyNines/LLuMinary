# GitHub Repository Best Practices Implementation Plan for LLuminary

This document outlines a comprehensive implementation plan for GitHub best practices based on an analysis of the LLuminary repository. The plan addresses missing elements and recommends automations for repository maintenance.

## 1. Repository Configuration

### Issue & PR Templates

#### Action Items:
- Create `.github/ISSUE_TEMPLATE/` directory with templates for:
  - Bug reports
  - Feature requests
  - Documentation improvements
- Create `.github/PULL_REQUEST_TEMPLATE.md` for standardized PR submissions

#### Implementation Timeline:
- Priority: High
- Estimated Completion: 1 day

#### Sample Templates:

**Bug Report Template** (`.github/ISSUE_TEMPLATE/bug_report.yml`):
```yaml
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
```

**PR Template** (`.github/PULL_REQUEST_TEMPLATE.md`):
```markdown
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
```

## 2. Security & Dependency Management

### Action Items:
- Configure Dependabot for automated dependency updates
- Add SECURITY.md for vulnerability reporting
- Implement CodeQL scanning for security vulnerabilities
- Configure branch protection rules

### Implementation Timeline:
- Priority: High
- Estimated Completion: 2 days

### Sample Configurations:

**Dependabot Configuration** (`.github/dependabot.yml`):
```yaml
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
```

**Security Policy** (`SECURITY.md`):
```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5.0 | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in LLuminary, please follow these steps:

1. **Do Not** disclose the vulnerability publicly until it has been addressed
2. Email the vulnerability details to [security@example.com](mailto:security@example.com)
3. Include detailed steps to reproduce the issue
4. Allow time for the vulnerability to be addressed before any public disclosure

We take all security vulnerabilities seriously and will respond to your report within 48 hours.
```

**CodeQL Scanning** (`.github/workflows/codeql.yml`):
```yaml
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
```

## 3. Repository Maintenance Automations

### Action Items:
- Implement stale issue/PR management
- Add automated PR labeling
- Create a code of conduct
- Configure PR size checker
- Set up release drafting automation

### Implementation Timeline:
- Priority: Medium
- Estimated Completion: 3 days

### Sample Configurations:

**Stale Issue Management** (`.github/workflows/stale.yml`):
```yaml
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
```

**PR Auto-Labeler** (`.github/labeler.yml`):
```yaml
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
```

**PR Labeler Workflow** (`.github/workflows/labeler.yml`):
```yaml
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
```

**Release Drafter** (`.github/workflows/release-drafter.yml`):
```yaml
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
```

**Release Drafter Config** (`.github/release-drafter.yml`):
```yaml
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
```

## 4. CI/CD Enhancements

### Action Items:
- Add scheduled test runs
- Implement API documentation generation
- Create integration environment deployment workflow
- Add test coverage reporting integration

### Implementation Timeline:
- Priority: Medium
- Estimated Completion: 4 days

### Sample Configurations:

**API Documentation Generation** (`.github/workflows/api-docs.yml`):
```yaml
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
```

**Scheduled Test Runs** (Add to existing CI workflow):
```yaml
on:
  schedule:
    - cron: '0 0 * * MON'  # Run every Monday at midnight
```

## 5. Project Board Automation

### Action Items:
- Create GitHub project board
- Configure automated issue/PR movement
- Implement field tracking for story points and priorities
- Set up project views for different stakeholders

### Implementation Timeline:
- Priority: Low
- Estimated Completion: 2 days

### Project Board Configuration:
- **Columns**:
  - To Do
  - In Progress
  - Review
  - Done

- **Automation Rules**:
  - New issues → To Do
  - Issues assigned → In Progress
  - PRs opened → In Progress
  - PRs ready for review → Review
  - PRs/Issues closed → Done

- **Field Tracking**:
  - Story Points (1, 2, 3, 5, 8, 13)
  - Priority (P0, P1, P2, P3)
  - Type (Bug, Feature, Task)

## Implementation Sequence and Timeline

1. **Week 1**:
   - Issue & PR Templates
   - Security & Dependency Management
   - Basic PR automation

2. **Week 2**:
   - Repository Maintenance Automations
   - CI/CD Enhancements

3. **Week 3**:
   - Project Board Automation
   - Documentation Updates

## Conclusion

Implementing these GitHub best practices will significantly improve the LLuminary repository's management, developer productivity, and community engagement. The structured approach ensures all critical areas are addressed while maintaining compatibility with existing workflows.

Key benefits:
1. Improved issue and PR quality through standardized templates
2. Enhanced security through automated vulnerability scanning
3. Reduced maintenance overhead through automated workflows
4. Better project tracking and visibility
5. Streamlined contribution process for new contributors

This implementation plan provides a comprehensive roadmap for aligning the LLuminary project with GitHub best practices, ensuring long-term sustainability and community growth.
