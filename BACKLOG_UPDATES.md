# LLuMinary Project Backlog Updates

This document provides comprehensive updates for the project backlog at https://github.com/orgs/PimpMyNines/projects/9/views/1

## Recommended Issue Order With Dependencies

### Phase 1: Fix Critical CI Infrastructure Issues

1. **[Issue #13: Fix Dockerfile.matrix handling in GitHub Actions workflow](https://github.com/PimpMyNines/LLuMinary/issues/13)**
   - **Status**: Ready for Development
   - **Priority**: Highest (P0)
   - **Size**: Medium
   - **Effort Estimate**: 8-12 hours (1-2 days)
   - **Sprint**: Current
   - **Assignee**: TBD
   - **Dependencies**: None
   - **Technical Debt**: Yes
   - **GitHub Fields to Update**:
     - Labels: `infrastructure`, `ci/cd`, `highest-priority`, `technical-debt`
     - Milestone: "CI Infrastructure Stabilization"
     - Project: Add to current sprint
     - Estimate: 8 story points
   - **Acceptance Criteria**:
     ```
     - Dockerfile.matrix is correctly generated and used in CI
     - CI workflow passes Docker build steps consistently
     - Changes are well-documented for future maintenance
     - PR includes before/after screenshots of successful CI runs
     ```
   - **Implementation Notes**:
     ```
     - Focus on .github/workflows/matrix-docker-tests.yml
     - Review Makefile targets that generate Dockerfile.matrix
     - Consider adding validation step to ensure file is created correctly
     ```

2. **[Issue #15: Improve provider test execution logic in CI](https://github.com/PimpMyNines/LLuMinary/issues/15)**
   - **Status**: Blocked (by #13)
   - **Priority**: Highest (P0)
   - **Size**: Medium
   - **Effort Estimate**: 8-12 hours (1-2 days)
   - **Sprint**: Current
   - **Assignee**: TBD
   - **Dependencies**: Issue #13 (blocks implementation)
   - **Technical Debt**: Yes
   - **GitHub Fields to Update**:
     - Labels: `infrastructure`, `ci/cd`, `highest-priority`, `technical-debt`, `blocked`
     - Milestone: "CI Infrastructure Stabilization"
     - Project: Add to current sprint
     - Estimate: 8 story points
   - **Acceptance Criteria**:
     ```
     - Provider-specific tests execute correctly in CI
     - FILE parameter is correctly passed to test-docker-file command
     - Conditional logic for provider tests works as expected
     - Test execution is well-documented
     - All provider test suites show as separate jobs in GitHub Actions UI
     ```
   - **Implementation Notes**:
     ```
     - Review matrix job configuration in workflow file
     - Fix issues with provider-specific test execution
     - Consider creating reusable composite actions for test execution
     ```

3. **[Issue #16: Configure CODECOV_TOKEN in GitHub repository secrets](https://github.com/PimpMyNines/LLuMinary/issues/16)**
   - **Status**: Ready for Development
   - **Priority**: Medium (P2)
   - **Size**: Small
   - **Effort Estimate**: 2-4 hours (0.5 days)
   - **Sprint**: Current
   - **Assignee**: TBD
   - **Dependencies**: None
   - **Technical Debt**: No
   - **GitHub Fields to Update**:
     - Labels: `infrastructure`, `ci/cd`, `medium-priority`, `quick-win`
     - Milestone: "CI Infrastructure Stabilization"
     - Project: Add to current sprint
     - Estimate: 3 story points
   - **Acceptance Criteria**:
     ```
     - CODECOV_TOKEN is set in GitHub repository secrets
     - Coverage reports are generated and uploaded correctly
     - Dashboard shows coverage metrics
     - Documentation updated with instructions for maintainers
     ```
   - **Implementation Notes**:
     ```
     - Generate new token from Codecov.io
     - Add to GitHub repository secrets
     - Verify token works by triggering a test CI run
     - Document process for future token rotation
     ```

### Phase 2: Improve Build and Testing Infrastructure

4. **[Issue #14: Fix docker-build-matrix-cached target in Makefile](https://github.com/PimpMyNines/LLuMinary/issues/14)**
   - **Status**: Blocked (by #13)
   - **Priority**: Medium (P2)
   - **Size**: Small
   - **Effort Estimate**: 4-6 hours (0.5-1 day)
   - **Sprint**: Current
   - **Assignee**: TBD
   - **Dependencies**: Issue #13 (requires fixed Dockerfile.matrix)
   - **Technical Debt**: Yes
   - **GitHub Fields to Update**:
     - Labels: `infrastructure`, `build`, `medium-priority`, `technical-debt`, `blocked`
     - Milestone: "CI Infrastructure Stabilization"
     - Project: Add to current sprint
     - Estimate: 5 story points
   - **Acceptance Criteria**:
     ```
     - docker-build-matrix-cached target correctly references Dockerfile.matrix
     - Docker layer caching works properly
     - Build times are improved by at least 25%
     - Documentation is updated with usage examples
     - PR includes build time comparisons before/after
     ```
   - **Implementation Notes**:
     ```
     - Review Makefile targets for Docker builds
     - Fix layer caching configuration
     - Consider adding explicit cache invalidation mechanism
     - Measure and document performance improvements
     ```

5. **[Issue #17: Ensure consistency in Docker-based testing setup](https://github.com/PimpMyNines/LLuMinary/issues/17)**
   - **Status**: Blocked (by #13, #14, #15)
   - **Priority**: Medium (P2)
   - **Size**: Medium
   - **Effort Estimate**: 8-12 hours (1-2 days)
   - **Sprint**: Next
   - **Assignee**: TBD
   - **Dependencies**: Issues #13, #14, #15 (requires fixed Docker infrastructure)
   - **Technical Debt**: Yes
   - **GitHub Fields to Update**:
     - Labels: `infrastructure`, `testing`, `medium-priority`, `technical-debt`, `blocked`
     - Milestone: "Testing Infrastructure Improvements"
     - Project: Add to backlog
     - Estimate: 8 story points
   - **Acceptance Criteria**:
     ```
     - All Docker testing approaches are standardized
     - Local and CI testing environments provide consistent results
     - Documentation clearly explains how to run tests in different environments
     - New developer onboarding guide includes Docker testing instructions
     - Visual diagram of testing infrastructure added to documentation
     ```
   - **Implementation Notes**:
     ```
     - Review and standardize Docker-based testing commands
     - Create helper scripts for common testing scenarios
     - Update README with clear instructions
     - Consider creating docker-compose setup for easier local testing
     ```

### Phase 3: Implement Core Features

6. **[Issue #3: Implement unified type definitions across providers](https://github.com/PimpMyNines/LLuMinary/issues/3)**
   - **Status**: Blocked (by #13, #15)
   - **Priority**: High (P1)
   - **Size**: Large
   - **Effort Estimate**: 16-24 hours (2-3 days)
   - **Sprint**: Next
   - **Assignee**: TBD
   - **Dependencies**: Issues #13, #15 (requires working CI to validate changes)
   - **Technical Debt**: No (Feature implementation)
   - **GitHub Fields to Update**:
     - Labels: `enhancement`, `typing`, `high-priority`, `blocked`, `core-feature`
     - Milestone: "Type System Overhaul"
     - Project: Add to backlog
     - Estimate: 13 story points
   - **Acceptance Criteria**:
     ```
     - All provider implementations use the unified type definitions
     - Type checking passes with mypy --strict flag
     - Type system is well-documented for future extensions
     - Unit tests verify type compatibility across providers
     - Migration guide for extension developers is created
     ```
   - **Implementation Notes**:
     ```
     - Focus on src/lluminary/models/types.py
     - Audit all provider files for type consistency
     - Create strong base interfaces for all providers
     - Consider creating type validation helpers
     - Add comprehensive type checking tests
     ```

7. **[Issue #4: Enhance streaming support for tool/function calling](https://github.com/PimpMyNines/LLuMinary/issues/4)**
   - **Status**: Blocked (by #3)
   - **Priority**: Medium (P2)
   - **Size**: Large
   - **Effort Estimate**: 16-24 hours (2-3 days)
   - **Sprint**: Future
   - **Assignee**: TBD
   - **Dependencies**: Issue #3 (requires unified type definitions)
   - **Technical Debt**: No (Feature implementation)
   - **GitHub Fields to Update**:
     - Labels: `enhancement`, `streaming`, `medium-priority`, `blocked`, `core-feature`
     - Milestone: "Advanced Streaming Support"
     - Project: Add to backlog
     - Estimate: 13 story points
   - **Acceptance Criteria**:
     ```
     - All providers support streaming tool/function calls where supported
     - Fallback mechanisms exist for providers without native support
     - Comprehensive examples added to documentation
     - Performance benchmarks show acceptable overhead
     - Integration tests verify behavior across providers
     ```
   - **Implementation Notes**:
     ```
     - Implementation should leverage the unified type system from Issue #3
     - Consider provider-specific edge cases in streaming implementations
     - Add throttling/backpressure mechanisms
     - Design for extensibility with future provider capabilities
     - Create examples demonstrating real-time tool use cases
     ```

## Additional Upcoming Issues (For Future Planning)

8. **[Issue #X: Implement Mistral AI Provider Support]**
   - **Status**: Not Started
   - **Priority**: Medium (P2)
   - **Size**: Medium
   - **Effort Estimate**: 12-16 hours (1.5-2 days)
   - **Sprint**: Future
   - **Dependencies**: Issues #3, #4 (requires type system and streaming)
   - **GitHub Fields to Create**:
     - Title: "Implement Mistral AI Provider Support"
     - Labels: `enhancement`, `provider`, `medium-priority`
     - Milestone: "Provider Expansion"
     - Estimate: 10 story points
   - **Acceptance Criteria**:
     ```
     - Mistral API integrated with unified interface
     - Authentication and error handling implemented
     - Streaming and tool calling supported
     - Comprehensive unit and integration tests
     - Documentation and examples updated
     ```

9. **[Issue #Y: Add Vector Database Integration]**
   - **Status**: Not Started
   - **Priority**: Medium (P2)
   - **Size**: Large
   - **Effort Estimate**: 24-32 hours (3-4 days)
   - **Sprint**: Future
   - **Dependencies**: Issue #3 (requires type system)
   - **GitHub Fields to Create**:
     - Title: "Add Vector Database Integration Support"
     - Labels: `enhancement`, `vector-db`, `medium-priority`
     - Milestone: "Advanced Features"
     - Estimate: 16 story points
   - **Acceptance Criteria**:
     ```
     - Abstract vector storage interface implemented
     - At least two backends supported (FAISS, Pinecone)
     - Seamless integration with embedding functionality
     - Performance benchmarks for different sizes
     - Comprehensive documentation and examples
     ```

## Justification for Order and Planning Details

This ordering optimizes for:

1. **Unblocking CI/CD First**: Issues #13, #15, and #16 directly address the failing CI pipeline, allowing all subsequent work to be properly tested and validated.

2. **Infrastructure Before Features**: Building a solid foundation (Issues #13-#17) before implementing core features (Issues #3-#4) ensures stable development and testing.

3. **Dependency Satisfaction**: Each issue is placed after its dependencies, ensuring we don't face blockers during implementation.

4. **Balanced Workload**: By interleaving high and low effort tasks, we maintain steady progress while avoiding burnout on complex issues.

5. **Clear Sprint Planning**: Issues are explicitly marked for current, next, or future sprints to provide clear planning guidance.

6. **Accurate Sizing**: Story points follow a modified Fibonacci sequence (3, 5, 8, 13, etc.) to acknowledge the uncertainty in larger estimates.

## GitHub Issue Update Plan

Based on requirements, the following updates will be made to each issue:

### Issue Types (Instead of Labels)
- Use the issue "type" field to categorize issues at the highest level:
  - **Bug**: Issues that represent a defect in existing functionality
  - **Enhancement**: Issues that represent new functionality or improvements
  - **Task**: Issues for maintenance work, documentation, or infrastructure
  - **Epic**: Large issues that contain multiple sub-tasks

### Labels (For Subcategories)
- Replace existing labels with more organized subcategories:
  - **Area**: `area:infrastructure`, `area:typing`, `area:providers`, `area:testing`
  - **Priority**: `priority:p0`, `priority:p1`, `priority:p2`, `priority:p3`
  - **Size**: `size:small`, `size:medium`, `size:large`
  - **Status**: `status:blocked`, `status:ready`
  - **Nature**: `technical-debt`, `performance`, `security`, `documentation`

### Project Assignment
- Set all issues to Project = "LLuMinary"
- Set appropriate Status field in project

### Milestones
- "CI Infrastructure Stabilization" - Issues #13, #14, #15, #16
- "Testing Infrastructure Improvements" - Issue #17
- "Type System Overhaul" - Issue #3
- "Advanced Streaming Support" - Issue #4
- "Provider Expansion" - Future Mistral issue
- "Advanced Features" - Future Vector DB issue

### Relationships
- Set all dependency relationships using GitHub's "Dependency" link type
- Also add "Related to" links for issues that have connections but aren't strict dependencies

### Sub-issues Plan

For each major issue, create the following sub-issues:

#### Issue #13: Fix Dockerfile.matrix handling in GitHub Actions workflow
1. **Sub-issue**: "Analyze current Dockerfile.matrix generation issues"
   - Type: Task
   - Labels: `area:infrastructure`, `size:small`
   - Estimate: 2 story points
   - Acceptance Criteria: Document all failing scenarios and root causes

2. **Sub-issue**: "Implement fixes for Dockerfile.matrix generation"
   - Type: Task
   - Labels: `area:infrastructure`, `size:medium`
   - Estimate: 4 story points
   - Acceptance Criteria: Dockerfile.matrix is generated correctly

3. **Sub-issue**: "Update GitHub Actions to properly use Dockerfile.matrix"
   - Type: Task
   - Labels: `area:infrastructure`, `size:small`
   - Estimate: 2 story points
   - Acceptance Criteria: GitHub Actions workflow uses correct Docker file

4. **Sub-issue**: "Document Dockerfile.matrix process for future maintenance"
   - Type: Task
   - Labels: `area:infrastructure`, `area:documentation`, `size:small`
   - Estimate: 1 story point
   - Acceptance Criteria: Clear documentation with examples and diagrams

#### Issue #15: Improve provider test execution logic in CI
1. **Sub-issue**: "Fix FILE parameter handling in test-docker-file command"
   - Type: Bug
   - Labels: `area:infrastructure`, `area:testing`, `size:small`
   - Estimate: 3 story points
   - Acceptance Criteria: FILE parameter correctly passes to Docker container

2. **Sub-issue**: "Implement conditional logic for provider-specific tests"
   - Type: Task
   - Labels: `area:infrastructure`, `area:testing`, `size:medium`
   - Estimate: 4 story points
   - Acceptance Criteria: Tests only run for changed providers

3. **Sub-issue**: "Create separate GitHub Actions jobs for each provider"
   - Type: Enhancement
   - Labels: `area:infrastructure`, `size:small`
   - Estimate: 2 story points
   - Acceptance Criteria: Each provider has a dedicated workflow job

#### Issue #3: Implement unified type definitions across providers
1. **Sub-issue**: "Design core type interfaces for all providers"
   - Type: Task
   - Labels: `area:typing`, `size:medium`
   - Estimate: 5 story points
   - Acceptance Criteria: Type interfaces defined for all shared functionality

2. **Sub-issue**: "Implement OpenAI provider with unified types"
   - Type: Task
   - Labels: `area:typing`, `area:providers`, `size:medium`
   - Estimate: 3 story points
   - Acceptance Criteria: OpenAI provider updated with new type system

3. **Sub-issue**: "Implement Anthropic provider with unified types"
   - Type: Task
   - Labels: `area:typing`, `area:providers`, `size:medium`
   - Estimate: 3 story points
   - Acceptance Criteria: Anthropic provider updated with new type system

4. **Sub-issue**: "Implement Bedrock provider with unified types"
   - Type: Task
   - Labels: `area:typing`, `area:providers`, `size:medium`
   - Estimate: 3 story points
   - Acceptance Criteria: Bedrock provider updated with new type system

5. **Sub-issue**: "Create comprehensive type checking tests"
   - Type: Task
   - Labels: `area:typing`, `area:testing`, `size:small`
   - Estimate: 2 story points
   - Acceptance Criteria: Tests validate type compatibility across providers

#### Issue #4: Enhance streaming support for tool/function calling
1. **Sub-issue**: "Design unified streaming interface for tools/functions"
   - Type: Task
   - Labels: `area:providers`, `size:medium`
   - Estimate: 5 story points
   - Acceptance Criteria: Interface design document with sequence diagrams

2. **Sub-issue**: "Implement OpenAI streaming tool support"
   - Type: Enhancement
   - Labels: `area:providers`, `size:medium`
   - Estimate: 3 story points
   - Acceptance Criteria: OpenAI provider supports streaming tool calls

3. **Sub-issue**: "Implement Anthropic streaming tool support"
   - Type: Enhancement
   - Labels: `area:providers`, `size:medium`
   - Estimate: 3 story points
   - Acceptance Criteria: Anthropic provider supports streaming tool calls

4. **Sub-issue**: "Implement fallback for providers without native streaming tools"
   - Type: Enhancement
   - Labels: `area:providers`, `size:small`
   - Estimate: 2 story points
   - Acceptance Criteria: All providers support tools via streaming interface

5. **Sub-issue**: "Create integration tests for streaming tool calls"
   - Type: Task
   - Labels: `area:testing`, `size:small`
   - Estimate: 2 story points
   - Acceptance Criteria: Tests verify streaming tool behavior across providers

## GitHub Project Setup Recommendations

1. **Board Columns**:
   - Backlog
   - Ready for Development
   - In Progress
   - Review/QA
   - Done

2. **Required Fields**:
   - Type (Bug/Enhancement/Task/Epic)
   - Status (dropdown)
   - Priority (P0-P3 scale)
   - Size (S/M/L/XL)
   - Estimate (story points)
   - Sprint (Current/Next/Future)
   - Dependencies (issue links)

3. **Automations**:
   - When PR linked, move to "In Progress"
   - When PR merged, move to "Done"
   - When blocked label added, update status to "Blocked"

## Issue Update Process

1. **For Each Main Issue**:
   - Update type field based on nature (mostly Task or Enhancement)
   - Apply appropriate labels from new categorization
   - Set Project = "LLuMinary"
   - Assign to appropriate milestone
   - Set relationships to other issues (dependencies)
   - Create sub-issues as outlined above
   - Link sub-issues to parent using "part of" relationship

2. **For Each Sub-issue**:
   - Set appropriate type field
   - Apply relevant labels
   - Set Project = "LLuMinary"
   - Assign to same milestone as parent
   - Set relationship to parent issue using "part of" relationship
   - Set dependencies between sub-issues where applicable

3. **After Updates**:
   - Verify all issues have proper type, labels, project, milestone, and relationships
   - Ensure all sub-issues are properly linked to parents
   - Validate that the project board shows all issues in appropriate columns
   - Confirm that automations are working as expected

## Implementation Instructions for Next Agent

Follow these step-by-step instructions to implement the GitHub issue structure defined in this document:

### 1. Initial Setup (Prerequisites)

1. Ensure you have write access to the GitHub repository: https://github.com/PimpMyNines/LLuMinary
2. Install GitHub CLI (`gh`) if not already available, and authenticate:
   ```bash
   gh auth login
   ```
3. Clone the repository locally if needed:
   ```bash
   gh repo clone PimpMyNines/LLuMinary
   ```

### 2. Create Label Structure

1. Create all labels with proper categorization using this script:

```bash
#!/bin/bash
# Area labels
gh label create "area:infrastructure" --color "0366d6" --description "Infrastructure related issues"
gh label create "area:typing" --color "0366d6" --description "Type system related issues"
gh label create "area:providers" --color "0366d6" --description "LLM provider implementation issues"
gh label create "area:testing" --color "0366d6" --description "Testing related issues"

# Priority labels
gh label create "priority:p0" --color "b60205" --description "Highest priority, must be fixed immediately"
gh label create "priority:p1" --color "d93f0b" --description "High priority, should be addressed soon"
gh label create "priority:p2" --color "fbca04" --description "Medium priority, address when convenient"
gh label create "priority:p3" --color "c5def5" --description "Low priority, nice to have"

# Size labels
gh label create "size:small" --color "0e8a16" --description "Small task (1-3 story points)"
gh label create "size:medium" --color "fbca04" --description "Medium task (5-8 story points)"
gh label create "size:large" --color "d93f0b" --description "Large task (13+ story points)"

# Status labels
gh label create "status:blocked" --color "b60205" --description "Work is blocked by another issue"
gh label create "status:ready" --color "0e8a16" --description "Ready for implementation"

# Nature labels
gh label create "technical-debt" --color "5319e7" --description "Issues addressing technical debt"
gh label create "performance" --color "5319e7" --description "Performance related issues"
gh label create "security" --color "b60205" --description "Security related issues"
gh label create "documentation" --color "1d76db" --description "Documentation related issues"
```

2. Create milestones for the project:

```bash
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="CI Infrastructure Stabilization" -f description="Fix critical CI infrastructure issues to enable reliable testing"
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="Testing Infrastructure Improvements" -f description="Improve testing infrastructure for better developer experience"
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="Type System Overhaul" -f description="Implement unified type definitions across providers"
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="Advanced Streaming Support" -f description="Enhance streaming support for tool/function calling"
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="Provider Expansion" -f description="Add support for additional LLM providers"
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="Advanced Features" -f description="Implement advanced features like vector database integration"
```

3. Get milestone IDs for reference:

```bash
gh api repos/PimpMyNines/LLuMinary/milestones | jq '.[] | {title: .title, id: .number}'
```

### 3. Update Main Issues

For each main issue (#13, #15, #16, #14, #17, #3, #4), update the issue with appropriate type, labels, and milestone:

Example for Issue #13:

```bash
# Set issue type (this is done via web UI, as GitHub API doesn't directly expose this field)
# Update labels, milestone, and project
gh issue edit 13 \
  --repo PimpMyNines/LLuMinary \
  --add-label "area:infrastructure,priority:p0,size:medium,status:ready,technical-debt" \
  --milestone "CI Infrastructure Stabilization"

# Add to LLuMinary project (this requires GraphQL mutations using gh api)
# For simplicity, do this step via GitHub UI
```

### 4. Create Sub-issues

For each main issue, create the defined sub-issues. Example for Issue #13 sub-issues:

```bash
# Create sub-issue 1
gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Analyze current Dockerfile.matrix generation issues" \
  --body "## Description
Analyze the root causes of the current issues with Dockerfile.matrix generation in the CI pipeline.

## Acceptance Criteria
- Document all failing scenarios and root causes
- Identify specific workflow steps that are failing
- Analysis includes recommendations for fixes

## Parent Issue
Part of #13: Fix Dockerfile.matrix handling in GitHub Actions workflow

## Estimate
2 story points" \
  --label "area:infrastructure,size:small" \
  --milestone "CI Infrastructure Stabilization"

# Repeat for all other sub-issues
```

### 5. Establish Issue Relationships

After creating all issues, set up the relationships using GitHub's UI:

1. For each sub-issue, link to its parent using "part of" relationship
2. For dependent issues, set up "Dependency" links
3. For related issues, set up "Related to" links

Example workflow for setting dependencies:
- Open issue #15 in GitHub UI
- Scroll down to Development section
- Click "Link an issue or pull request"
- Enter "#13" to link to issue #13
- Select "Dependency" as the relationship type

### 6. Update Project Board

1. Ensure all issues are added to the "LLuMinary" project
2. Set appropriate status values for each issue in the project board
3. Configure project view to show relevant fields:
   - Type
   - Status
   - Priority
   - Size
   - Estimate
   - Sprint
   - Dependencies

### 7. Verify Implementation

1. Check that all issues have correct:
   - Type
   - Labels
   - Project assignment
   - Milestone
   - Related issues

2. Verify relationships:
   - All sub-issues are linked to parents
   - All dependencies are properly set
   - All related issues are linked

3. Test automations:
   - Create a test PR to verify "In Progress" automation
   - Close a test issue to verify "Done" automation

### 8. Document Completion

Create a status report on the implementation, including:
1. Screenshot of updated project board
2. List of all issues updated
3. List of all sub-issues created
4. Any challenges encountered and how they were resolved
5. Recommendations for future improvements to the issue structure

This report should be saved as `BACKLOG_IMPLEMENTATION_REPORT.md` in the repository.
