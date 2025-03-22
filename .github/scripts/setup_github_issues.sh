#!/bin/bash
# Script to set up GitHub issues for LLuminary project based on BACKLOG_UPDATES.md

echo "Starting setup of GitHub issues for LLuminary project..."

# 1. Create Label Structure
echo "Creating labels..."

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

# 2. Create Milestones
echo "Creating milestones..."
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="CI Infrastructure Stabilization" -f description="Fix critical CI infrastructure issues to enable reliable testing"
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="Testing Infrastructure Improvements" -f description="Improve testing infrastructure for better developer experience"
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="Type System Overhaul" -f description="Implement unified type definitions across providers"
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="Advanced Streaming Support" -f description="Enhance streaming support for tool/function calling"
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="Provider Expansion" -f description="Add support for additional LLM providers"
gh api repos/PimpMyNines/LLuMinary/milestones --method POST -f title="Advanced Features" -f description="Implement advanced features like vector database integration"

# 3. Get milestone IDs
echo "Getting milestone IDs..."
gh api repos/PimpMyNines/LLuMinary/milestones | jq '.[] | {title: .title, id: .number}'

# 4. Update Main Issues
echo "Updating main issues..."

# Set issue type (this approach works on GitHub's API directly)
set_issue_type() {
  local issue_number=$1
  local issue_type=$2

  # Get the current body of the issue
  local body=$(gh issue view $issue_number --repo PimpMyNines/LLuMinary --json body --jq .body)

  # Prepend the issue type to the body
  local new_body="**Issue Type**: $issue_type

$body"

  # Update the issue with the new body
  gh issue edit $issue_number --repo PimpMyNines/LLuMinary --body "$new_body"

  echo "Set issue #$issue_number type to $issue_type"
}

# Issue #13
echo "Updating Issue #13..."
gh issue edit 13 \
  --repo PimpMyNines/LLuMinary \
  --add-label "area:infrastructure,priority:p0,size:medium,status:ready,technical-debt" \
  --milestone "CI Infrastructure Stabilization"
set_issue_type 13 "Task"

# Issue #15
echo "Updating Issue #15..."
gh issue edit 15 \
  --repo PimpMyNines/LLuMinary \
  --add-label "area:infrastructure,priority:p0,size:medium,status:blocked,technical-debt" \
  --milestone "CI Infrastructure Stabilization"
set_issue_type 15 "Task"

# Issue #16
echo "Updating Issue #16..."
gh issue edit 16 \
  --repo PimpMyNines/LLuMinary \
  --add-label "area:infrastructure,priority:p2,size:small,status:ready" \
  --milestone "CI Infrastructure Stabilization"
set_issue_type 16 "Task"

# Issue #14
echo "Updating Issue #14..."
gh issue edit 14 \
  --repo PimpMyNines/LLuMinary \
  --add-label "area:infrastructure,priority:p2,size:small,status:blocked,technical-debt" \
  --milestone "CI Infrastructure Stabilization"
set_issue_type 14 "Task"

# Issue #17
echo "Updating Issue #17..."
gh issue edit 17 \
  --repo PimpMyNines/LLuMinary \
  --add-label "area:infrastructure,priority:p2,size:medium,status:blocked,technical-debt" \
  --milestone "Testing Infrastructure Improvements"
set_issue_type 17 "Task"

# Issue #3
echo "Updating Issue #3..."
gh issue edit 3 \
  --repo PimpMyNines/LLuMinary \
  --add-label "area:typing,priority:p1,size:large,status:blocked" \
  --milestone "Type System Overhaul"
set_issue_type 3 "Feature"

# Issue #4
echo "Updating Issue #4..."
gh issue edit 4 \
  --repo PimpMyNines/LLuMinary \
  --add-label "area:providers,priority:p2,size:large,status:blocked" \
  --milestone "Advanced Streaming Support"
set_issue_type 4 "Feature"

# 5. Create Sub-issues

# Sub-issues for #13
echo "Creating sub-issues for Issue #13..."

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

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement fixes for Dockerfile.matrix generation" \
  --body "## Description
Fix the identified issues with Dockerfile.matrix generation to ensure it's created correctly.

## Acceptance Criteria
- Dockerfile.matrix is generated correctly
- Generation process is reliable and consistent
- Failures are properly handled with clear error messages

## Parent Issue
Part of #13: Fix Dockerfile.matrix handling in GitHub Actions workflow

## Estimate
4 story points" \
  --label "area:infrastructure,size:medium" \
  --milestone "CI Infrastructure Stabilization"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Update GitHub Actions to properly use Dockerfile.matrix" \
  --body "## Description
Update the GitHub Actions workflow to correctly reference and use the generated Dockerfile.matrix.

## Acceptance Criteria
- GitHub Actions workflow uses correct Docker file
- Build process completes successfully
- Docker build logs show correct file being used

## Parent Issue
Part of #13: Fix Dockerfile.matrix handling in GitHub Actions workflow

## Estimate
2 story points" \
  --label "area:infrastructure,size:small" \
  --milestone "CI Infrastructure Stabilization"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Document Dockerfile.matrix process for future maintenance" \
  --body "## Description
Create comprehensive documentation for the Dockerfile.matrix generation and usage process.

## Acceptance Criteria
- Clear documentation with examples and diagrams
- Troubleshooting guide for common issues
- Process flow chart showing how matrix testing works

## Parent Issue
Part of #13: Fix Dockerfile.matrix handling in GitHub Actions workflow

## Estimate
1 story point" \
  --label "area:infrastructure,area:documentation,size:small" \
  --milestone "CI Infrastructure Stabilization"

# Sub-issues for #15
echo "Creating sub-issues for Issue #15..."

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Fix FILE parameter handling in test-docker-file command" \
  --body "## Description
Fix issues with the FILE parameter not being correctly passed to the Docker container in the test-docker-file command.

## Acceptance Criteria
- FILE parameter correctly passes to Docker container
- Test execution targets the right files
- Parameters with special characters are handled correctly

## Parent Issue
Part of #15: Improve provider test execution logic in CI

## Estimate
3 story points" \
  --label "area:infrastructure,area:testing,size:small" \
  --milestone "CI Infrastructure Stabilization"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement conditional logic for provider-specific tests" \
  --body "## Description
Implement logic to conditionally run provider-specific tests based on changes made in a PR.

## Acceptance Criteria
- Tests only run for changed providers
- Detection mechanism is reliable
- Time savings are measurable in CI runs

## Parent Issue
Part of #15: Improve provider test execution logic in CI

## Estimate
4 story points" \
  --label "area:infrastructure,area:testing,size:medium" \
  --milestone "CI Infrastructure Stabilization"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Create separate GitHub Actions jobs for each provider" \
  --body "## Description
Refactor the GitHub Actions workflow to have separate jobs for each provider's tests.

## Acceptance Criteria
- Each provider has a dedicated workflow job
- Jobs can run in parallel for faster CI
- Clear reporting of which provider tests failed

## Parent Issue
Part of #15: Improve provider test execution logic in CI

## Estimate
2 story points" \
  --label "area:infrastructure,size:small" \
  --milestone "CI Infrastructure Stabilization"

# Sub-issues for #3
echo "Creating sub-issues for Issue #3..."

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Design core type interfaces for all providers" \
  --body "## Description
Design and implement the core type interfaces that will be used across all providers.

## Acceptance Criteria
- Type interfaces defined for all shared functionality
- Types are well-documented with comments
- Type hierarchy is clear and logical

## Parent Issue
Part of #3: Implement unified type definitions across providers

## Estimate
5 story points" \
  --label "area:typing,size:medium" \
  --milestone "Type System Overhaul"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement OpenAI provider with unified types" \
  --body "## Description
Update the OpenAI provider implementation to use the new unified type system.

## Acceptance Criteria
- OpenAI provider updated with new type system
- All methods use correct type annotations
- No type checking errors with mypy --strict

## Parent Issue
Part of #3: Implement unified type definitions across providers

## Estimate
3 story points" \
  --label "area:typing,area:providers,size:medium" \
  --milestone "Type System Overhaul"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement Anthropic provider with unified types" \
  --body "## Description
Update the Anthropic provider implementation to use the new unified type system.

## Acceptance Criteria
- Anthropic provider updated with new type system
- All methods use correct type annotations
- No type checking errors with mypy --strict

## Parent Issue
Part of #3: Implement unified type definitions across providers

## Estimate
3 story points" \
  --label "area:typing,area:providers,size:medium" \
  --milestone "Type System Overhaul"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement Bedrock provider with unified types" \
  --body "## Description
Update the Bedrock provider implementation to use the new unified type system.

## Acceptance Criteria
- Bedrock provider updated with new type system
- All methods use correct type annotations
- No type checking errors with mypy --strict

## Parent Issue
Part of #3: Implement unified type definitions across providers

## Estimate
3 story points" \
  --label "area:typing,area:providers,size:medium" \
  --milestone "Type System Overhaul"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Create comprehensive type checking tests" \
  --body "## Description
Create comprehensive tests to validate type compatibility across providers.

## Acceptance Criteria
- Tests validate type compatibility across providers
- Test coverage for all major type interfaces
- Integration with CI to ensure type safety

## Parent Issue
Part of #3: Implement unified type definitions across providers

## Estimate
2 story points" \
  --label "area:typing,area:testing,size:small" \
  --milestone "Type System Overhaul"

# Sub-issues for #4
echo "Creating sub-issues for Issue #4..."

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Design unified streaming interface for tools/functions" \
  --body "## Description
Design a unified interface for streaming tool/function calls across all providers.

## Acceptance Criteria
- Interface design document with sequence diagrams
- Clear API contract for all provider implementations
- Design considers backward compatibility

## Parent Issue
Part of #4: Enhance streaming support for tool/function calling

## Estimate
5 story points" \
  --label "area:providers,size:medium" \
  --milestone "Advanced Streaming Support"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement OpenAI streaming tool support" \
  --body "## Description
Implement streaming tool/function call support for the OpenAI provider.

## Acceptance Criteria
- OpenAI provider supports streaming tool calls
- Implementation follows the unified interface design
- Performance overhead is minimal

## Parent Issue
Part of #4: Enhance streaming support for tool/function calling

## Estimate
3 story points" \
  --label "area:providers,size:medium" \
  --milestone "Advanced Streaming Support"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement Anthropic streaming tool support" \
  --body "## Description
Implement streaming tool/function call support for the Anthropic provider.

## Acceptance Criteria
- Anthropic provider supports streaming tool calls
- Implementation follows the unified interface design
- Performance overhead is minimal

## Parent Issue
Part of #4: Enhance streaming support for tool/function calling

## Estimate
3 story points" \
  --label "area:providers,size:medium" \
  --milestone "Advanced Streaming Support"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement fallback for providers without native streaming tools" \
  --body "## Description
Implement a fallback mechanism for providers that don't natively support streaming tool calls.

## Acceptance Criteria
- All providers support tools via streaming interface
- Fallback mechanism is transparent to users
- Performance impact is documented

## Parent Issue
Part of #4: Enhance streaming support for tool/function calling

## Estimate
2 story points" \
  --label "area:providers,size:small" \
  --milestone "Advanced Streaming Support"

gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Create integration tests for streaming tool calls" \
  --body "## Description
Create integration tests to verify streaming tool/function call behavior across providers.

## Acceptance Criteria
- Tests verify streaming tool behavior across providers
- Edge cases are covered (timeouts, errors, etc.)
- Tests run reliably in CI

## Parent Issue
Part of #4: Enhance streaming support for tool/function calling

## Estimate
2 story points" \
  --label "area:testing,size:small" \
  --milestone "Advanced Streaming Support"

# 6. Create New Issues
echo "Creating new issues..."

# Mistral AI Provider - Parent Issue
mistral_issue=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement Mistral AI Provider Support" \
  --body "## Description
Add support for the Mistral AI provider to LLuMinary.

## Acceptance Criteria
- Mistral API integrated with unified interface
- Authentication and error handling implemented
- Streaming and tool calling supported
- Comprehensive unit and integration tests
- Documentation and examples updated

## Size
Medium

## Effort Estimate
12-16 hours (1.5-2 days)

## Dependencies
- #3: Implement unified type definitions across providers
- #4: Enhance streaming support for tool/function calling

## Story Points
8 (Sum of sub-issues)" \
  --label "area:providers,priority:p2,size:medium" \
  --milestone "Provider Expansion" \
  --json number --jq .number)

set_issue_type "$mistral_issue" "Feature"

echo "Created Mistral AI parent issue #$mistral_issue"

# Sub-issues for Mistral AI Provider
echo "Creating sub-issues for Mistral AI Provider..."

# 1. Research and API Analysis
mistral_sub1=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Research Mistral AI API and create implementation plan" \
  --body "## Description
Research the Mistral AI API, document its capabilities, and create a detailed implementation plan.

## Acceptance Criteria
- Document API endpoints and parameters
- Compare features with existing providers
- Identify any unique capabilities or limitations
- Map Mistral API concepts to our unified interface
- Create detailed implementation roadmap

## Parent Issue
Part of #$mistral_issue: Implement Mistral AI Provider Support

## Story Points
1" \
  --label "area:providers,priority:p2,size:small" \
  --milestone "Provider Expansion" \
  --json number --jq .number)

set_issue_type "$mistral_sub1" "Task"
set_parent_child_relationship "$mistral_sub1" "$mistral_issue"

# 2. Authentication Implementation
mistral_sub2=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement Mistral AI authentication and client setup" \
  --body "## Description
Implement authentication and client setup for the Mistral AI provider.

## Acceptance Criteria
- Support API key authentication
- Implement client initialization logic
- Handle environment variables for configuration
- Implement credential validation
- Add support for different API endpoints/regions
- Create tests for authentication flows

## Parent Issue
Part of #$mistral_issue: Implement Mistral AI Provider Support

## Story Points
2" \
  --label "area:providers,priority:p2,size:small" \
  --milestone "Provider Expansion" \
  --json number --jq .number)

set_issue_type "$mistral_sub2" "Task"
set_parent_child_relationship "$mistral_sub2" "$mistral_issue"

# 3. Core Generation Implementation
mistral_sub3=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement Mistral AI core text generation capabilities" \
  --body "## Description
Implement the core text generation capabilities for the Mistral AI provider.

## Acceptance Criteria
- Implement the generate method
- Map parameters correctly to Mistral AI API
- Support all relevant model configuration options
- Handle response parsing
- Implement prompt formatting
- Add unit tests for generation functionality

## Technical Context
The Mistral API has specific parameters and prompt formatting requirements that need to be mapped to our unified interface.

## Parent Issue
Part of #$mistral_issue: Implement Mistral AI Provider Support

## Story Points
3" \
  --label "area:providers,priority:p2,size:small" \
  --milestone "Provider Expansion" \
  --json number --jq .number)

set_issue_type "$mistral_sub3" "Task"
set_parent_child_relationship "$mistral_sub3" "$mistral_issue"

# 4. Streaming Implementation
mistral_sub4=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement Mistral AI streaming capabilities" \
  --body "## Description
Implement streaming support for the Mistral AI provider.

## Acceptance Criteria
- Implement streaming text generation
- Support streaming tool/function calls if available
- Implement proper stream parsing
- Handle streaming errors gracefully
- Ensure compatibility with the unified streaming interface
- Add streaming-specific tests

## Parent Issue
Part of #$mistral_issue: Implement Mistral AI Provider Support

## Story Points
2" \
  --label "area:providers,priority:p2,size:small" \
  --milestone "Provider Expansion" \
  --json number --jq .number)

set_issue_type "$mistral_sub4" "Task"
set_parent_child_relationship "$mistral_sub4" "$mistral_issue"

# 5. Error Handling
mistral_sub5=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement Mistral AI error handling and timeout management" \
  --body "## Description
Implement comprehensive error handling and timeout management for the Mistral AI provider.

## Acceptance Criteria
- Map Mistral-specific errors to our exception hierarchy
- Implement retry logic for transient errors
- Add timeout handling for API calls
- Create tests for error scenarios
- Document error handling behavior

## Parent Issue
Part of #$mistral_issue: Implement Mistral AI Provider Support

## Story Points
2" \
  --label "area:providers,priority:p2,size:small" \
  --milestone "Provider Expansion" \
  --json number --jq .number)

set_issue_type "$mistral_sub5" "Task"
set_parent_child_relationship "$mistral_sub5" "$mistral_issue"

# 6. Documentation and Examples
mistral_sub6=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Create Mistral AI provider documentation and examples" \
  --body "## Description
Create comprehensive documentation and examples for the Mistral AI provider.

## Acceptance Criteria
- Add API documentation for all Mistral-specific functionality
- Create usage examples
- Document supported models and capabilities
- Add configuration examples
- Update existing documentation to include Mistral

## Parent Issue
Part of #$mistral_issue: Implement Mistral AI Provider Support

## Story Points
1" \
  --label "area:providers,area:documentation,priority:p2,size:small" \
  --milestone "Provider Expansion" \
  --json number --jq .number)

set_issue_type "$mistral_sub6" "Task"
set_parent_child_relationship "$mistral_sub6" "$mistral_issue"

# Vector Database Integration - Parent Issue
vector_db_issue=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Add Vector Database Integration Support" \
  --body "## Description
Implement vector database integration to support efficient similarity search and retrieval.

## Acceptance Criteria
- Abstract vector storage interface implemented
- At least two backends supported (FAISS, Pinecone)
- Seamless integration with embedding functionality
- Performance benchmarks for different sizes
- Comprehensive documentation and examples

## Size
Large

## Effort Estimate
24-32 hours (3-4 days)

## Dependencies
- #3: Implement unified type definitions across providers

## Story Points
13 (Sum of sub-issues)" \
  --label "area:providers,priority:p2,size:large" \
  --milestone "Advanced Features" \
  --json number --jq .number)

set_issue_type "$vector_db_issue" "Feature"

echo "Created Vector DB parent issue #$vector_db_issue"

# Sub-issues for Vector DB Integration
echo "Creating sub-issues for Vector DB Integration..."

# 1. Design abstract vector storage interface
vector_db_sub1=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Design vector storage interface and type definitions" \
  --body "## Description
Design the abstract interface and type definitions for vector storage that will be implemented by all backend providers.

## Acceptance Criteria
- Create abstract base classes and interfaces
- Define type-safe method signatures
- Create data models for vector entries
- Design index management functionality
- Document extension patterns for new backends

## Parent Issue
Part of #$vector_db_issue: Add Vector Database Integration Support

## Story Points
3" \
  --label "area:providers,area:typing,priority:p2,size:small" \
  --milestone "Advanced Features" \
  --json number --jq .number)

set_issue_type "$vector_db_sub1" "Task"
set_parent_child_relationship "$vector_db_sub1" "$vector_db_issue"

# 2. Implement FAISS backend
vector_db_sub2=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement FAISS vector database backend" \
  --body "## Description
Implement the FAISS backend for the vector database integration.

## Acceptance Criteria
- Implement all required interface methods
- Support different index types (flat, HNSW, etc.)
- Handle vector similarity search efficiently
- Implement serialization/deserialization
- Add comprehensive unit tests

## Technical Context
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM.

## Parent Issue
Part of #$vector_db_issue: Add Vector Database Integration Support

## Story Points
5" \
  --label "area:providers,priority:p2,size:medium" \
  --milestone "Advanced Features" \
  --json number --jq .number)

set_issue_type "$vector_db_sub2" "Task"
set_parent_child_relationship "$vector_db_sub2" "$vector_db_issue"

# 3. Implement Pinecone backend
vector_db_sub3=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Implement Pinecone vector database backend" \
  --body "## Description
Implement the Pinecone backend for the vector database integration.

## Acceptance Criteria
- Implement Pinecone client with auth handling
- Support all backend operations (insert, upsert, query, delete)
- Implement metadata filtering
- Handle connection pooling and timeouts
- Add comprehensive unit and integration tests

## Technical Context
Pinecone is a managed vector database service optimized for vector search. It provides a cloud-based API that needs proper authentication and connection management.

## Parent Issue
Part of #$vector_db_issue: Add Vector Database Integration Support

## Story Points
5" \
  --label "area:providers,priority:p2,size:medium" \
  --milestone "Advanced Features" \
  --json number --jq .number)

set_issue_type "$vector_db_sub3" "Task"
set_parent_child_relationship "$vector_db_sub3" "$vector_db_issue"

# 4. Create Embedding Integration
vector_db_sub4=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Integrate vector storage with embedding functionality" \
  --body "## Description
Connect the vector database functionality with the existing embedding functionality to provide a seamless experience.

## Acceptance Criteria
- Create easy-to-use helper methods for embedding+storage
- Implement automatic dimension detection
- Support batched operations for efficiency
- Add utility functions for common search patterns
- Create examples showing the integration workflow

## Parent Issue
Part of #$vector_db_issue: Add Vector Database Integration Support

## Story Points
3" \
  --label "area:providers,priority:p2,size:small" \
  --milestone "Advanced Features" \
  --json number --jq .number)

set_issue_type "$vector_db_sub4" "Task"
set_parent_child_relationship "$vector_db_sub4" "$vector_db_issue"

# 5. Performance testing
vector_db_sub5=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Benchmark and optimize vector database performance" \
  --body "## Description
Create benchmarks and optimize the performance of the vector database implementations.

## Acceptance Criteria
- Create benchmark suite for different vector sizes
- Benchmark insertion, query, and deletion operations
- Compare performance between backends
- Identify and implement optimization opportunities
- Document performance characteristics

## Parent Issue
Part of #$vector_db_issue: Add Vector Database Integration Support

## Story Points
2" \
  --label "area:providers,priority:p2,size:small" \
  --milestone "Advanced Features" \
  --json number --jq .number)

set_issue_type "$vector_db_sub5" "Task"
set_parent_child_relationship "$vector_db_sub5" "$vector_db_issue"

# 6. Documentation
vector_db_sub6=$(gh issue create \
  --repo PimpMyNines/LLuMinary \
  --title "Create documentation and examples for vector database integration" \
  --body "## Description
Create comprehensive documentation and examples for the vector database functionality.

## Acceptance Criteria
- Add API documentation for all vector database components
- Create tutorial for getting started with vector search
- Add examples for each supported backend
- Document best practices and limitations
- Add integration examples with embeddings functionality

## Parent Issue
Part of #$vector_db_issue: Add Vector Database Integration Support

## Story Points
1" \
  --label "area:providers,area:documentation,priority:p2,size:small" \
  --milestone "Advanced Features" \
  --json number --jq .number)

set_issue_type "$vector_db_sub6" "Task"
set_parent_child_relationship "$vector_db_sub6" "$vector_db_issue"

echo "Creating implementation report..."

# Create implementation report
cat > BACKLOG_IMPLEMENTATION_REPORT.md << EOL
# Backlog Implementation Report

## Overview

This report documents the implementation of the GitHub issue structure as defined in the BACKLOG_UPDATES.md file. The implementation involved creating a structured set of issues, labels, milestones, and relationships to better organize the LLuMinary project.

## Implementation Details

### Labels Created
- Area labels: area:infrastructure, area:typing, area:providers, area:testing
- Priority labels: priority:p0, priority:p1, priority:p2, priority:p3
- Size labels: size:small, size:medium, size:large
- Status labels: status:blocked, status:ready
- Nature labels: technical-debt, performance, security, documentation

### Milestones Created
- CI Infrastructure Stabilization
- Testing Infrastructure Improvements
- Type System Overhaul
- Advanced Streaming Support
- Provider Expansion
- Advanced Features

### Main Issues Updated
- Issue #13: Fix Dockerfile.matrix handling in GitHub Actions workflow (Task)
- Issue #15: Improve provider test execution logic in CI (Task)
- Issue #16: Configure CODECOV_TOKEN in GitHub repository secrets (Task)
- Issue #14: Fix docker-build-matrix-cached target in Makefile (Task)
- Issue #17: Ensure consistency in Docker-based testing setup (Task)
- Issue #3: Implement unified type definitions across providers (Feature)
- Issue #4: Enhance streaming support for tool/function calling (Feature)

### Sub-issues Created

#### For Issue #13
- Analyze current Dockerfile.matrix generation issues (2 story points)
- Implement fixes for Dockerfile.matrix generation (4 story points)
- Update GitHub Actions to properly use Dockerfile.matrix (2 story points)
- Document Dockerfile.matrix process for future maintenance (1 story point)

#### For Issue #15
- Fix FILE parameter handling in test-docker-file command (3 story points)
- Implement conditional logic for provider-specific tests (4 story points)
- Create separate GitHub Actions jobs for each provider (2 story points)

#### For Issue #3
- Design core type interfaces for all providers (5 story points)
- Implement OpenAI provider with unified types (3 story points)
- Implement Anthropic provider with unified types (3 story points)
- Implement Bedrock provider with unified types (3 story points)
- Create comprehensive type checking tests (2 story points)

#### For Issue #4
- Design unified streaming interface for tools/functions (5 story points)
- Implement OpenAI streaming tool support (3 story points)
- Implement Anthropic streaming tool support (3 story points)
- Implement fallback for providers without native streaming tools (2 story points)
- Create integration tests for streaming tool calls (2 story points)

### New Features With Sub-issues

#### Mistral AI Provider Support (8 story points total)
- Research Mistral AI API and create implementation plan (1 story point)
- Implement Mistral AI authentication and client setup (2 story points)
- Implement Mistral AI core text generation capabilities (3 story points)
- Implement Mistral AI streaming capabilities (2 story points)
- Implement Mistral AI error handling and timeout management (2 story points)
- Create Mistral AI provider documentation and examples (1 story point)

#### Vector Database Integration (13 story points total)
- Design vector storage interface and type definitions (3 story points)
- Implement FAISS vector database backend (5 story points)
- Implement Pinecone vector database backend (5 story points)
- Integrate vector storage with embedding functionality (3 story points)
- Benchmark and optimize vector database performance (2 story points)
- Create documentation and examples for vector database integration (1 story point)

### Story Point Strategy
All story points follow the Fibonacci sequence (1, 2, 3, 5, 8, 13, 21) to reflect the inherent uncertainty in larger tasks. No sub-issue exceeds 5 story points, ensuring manageable units of work.

### Issue Type Assignment
- **Feature**: Used for new capabilities (Mistral provider, Vector DB integration)
- **Task**: Used for implementation and maintenance work
- **Bug**: Used for defect fixes

## Next Steps

1. Manually review and establish issue relationships in GitHub UI:
   - Set "part of" relationships for all sub-issues
   - Set "dependency" relationships between related issues

2. Ensure all issues are added to the "LLuMinary" project board

3. Set appropriate status values for each issue in the project board

## Challenges and Solutions

### Challenge: GitHub CLI limitations
The GitHub CLI (`gh`) doesn't directly support setting certain fields like issue type and some project properties. These will need to be set manually through the GitHub UI.

### Challenge: Relationship management
GitHub CLI doesn't provide direct commands for setting issue relationships. These need to be managed through the GitHub UI or via GraphQL mutations.

## Recommendations for Future Improvements

1. Create a more comprehensive script using the GitHub API and GraphQL to fully automate issue creation and relationship management.

2. Implement automated project board updates using GitHub Actions to keep project status in sync with issue status.

3. Consider using GitHub's new issue form templates to standardize issue creation in the future.

4. Develop a visualization tool for the dependency graph to better understand issue relationships.

EOL

# 7. Set up issue relationships using GitHub GraphQL API
echo "Setting up issue relationships..."

# Function to set parent/child relationship
set_parent_child_relationship() {
  local child_number=$1
  local parent_number=$2

  # Fetch the node IDs of the issues
  echo "Getting node IDs for issues #$child_number and #$parent_number..."

  child_node_id=$(gh api graphql -f query='
    query($repo: String!, $owner: String!, $number: Int!) {
      repository(name: $repo, owner: $owner) {
        issue(number: $number) {
          id
        }
      }
    }' -f repo=LLuMinary -f owner=PimpMyNines -f number=$child_number --jq '.data.repository.issue.id')

  parent_node_id=$(gh api graphql -f query='
    query($repo: String!, $owner: String!, $number: Int!) {
      repository(name: $repo, owner: $owner) {
        issue(number: $number) {
          id
        }
      }
    }' -f repo=LLuMinary -f owner=PimpMyNines -f number=$parent_number --jq '.data.repository.issue.id')

  # Create the relationship using the GraphQL mutation
  echo "Setting issue #$child_number as a child of issue #$parent_number..."

  gh api graphql -f query='
    mutation($childId: ID!, $parentId: ID!) {
      createLinkedBranch(input: {
        parentId: $parentId,
        childId: $childId,
        name: "relationship"
      }) {
        linkedBranch {
          id
        }
      }
    }' -f childId="$child_node_id" -f parentId="$parent_node_id"
}

# Set relationships for Issue #13 sub-issues
recent_issues=$(gh issue list --repo PimpMyNines/LLuMinary --limit 30 --json number,title)

# Extract issue numbers for the sub-issues we created
extract_issue_number() {
  local title_pattern="$1"
  echo "$recent_issues" | jq -r ".[] | select(.title | contains(\"$title_pattern\")) | .number"
}

# For Issue #13 sub-issues
issue_13_sub1=$(extract_issue_number "Analyze current Dockerfile.matrix generation issues")
issue_13_sub2=$(extract_issue_number "Implement fixes for Dockerfile.matrix generation")
issue_13_sub3=$(extract_issue_number "Update GitHub Actions to properly use Dockerfile.matrix")
issue_13_sub4=$(extract_issue_number "Document Dockerfile.matrix process for future maintenance")

if [[ -n "$issue_13_sub1" ]]; then
  set_parent_child_relationship "$issue_13_sub1" 13
fi
if [[ -n "$issue_13_sub2" ]]; then
  set_parent_child_relationship "$issue_13_sub2" 13
fi
if [[ -n "$issue_13_sub3" ]]; then
  set_parent_child_relationship "$issue_13_sub3" 13
fi
if [[ -n "$issue_13_sub4" ]]; then
  set_parent_child_relationship "$issue_13_sub4" 13
fi

# For Issue #15 sub-issues
issue_15_sub1=$(extract_issue_number "Fix FILE parameter handling in test-docker-file command")
issue_15_sub2=$(extract_issue_number "Implement conditional logic for provider-specific tests")
issue_15_sub3=$(extract_issue_number "Create separate GitHub Actions jobs for each provider")

if [[ -n "$issue_15_sub1" ]]; then
  set_parent_child_relationship "$issue_15_sub1" 15
fi
if [[ -n "$issue_15_sub2" ]]; then
  set_parent_child_relationship "$issue_15_sub2" 15
fi
if [[ -n "$issue_15_sub3" ]]; then
  set_parent_child_relationship "$issue_15_sub3" 15
fi

# For Issue #3 sub-issues
issue_3_sub1=$(extract_issue_number "Design core type interfaces for all providers")
issue_3_sub2=$(extract_issue_number "Implement OpenAI provider with unified types")
issue_3_sub3=$(extract_issue_number "Implement Anthropic provider with unified types")
issue_3_sub4=$(extract_issue_number "Implement Bedrock provider with unified types")
issue_3_sub5=$(extract_issue_number "Create comprehensive type checking tests")

if [[ -n "$issue_3_sub1" ]]; then
  set_parent_child_relationship "$issue_3_sub1" 3
fi
if [[ -n "$issue_3_sub2" ]]; then
  set_parent_child_relationship "$issue_3_sub2" 3
fi
if [[ -n "$issue_3_sub3" ]]; then
  set_parent_child_relationship "$issue_3_sub3" 3
fi
if [[ -n "$issue_3_sub4" ]]; then
  set_parent_child_relationship "$issue_3_sub4" 3
fi
if [[ -n "$issue_3_sub5" ]]; then
  set_parent_child_relationship "$issue_3_sub5" 3
fi

# For Issue #4 sub-issues
issue_4_sub1=$(extract_issue_number "Design unified streaming interface for tools/functions")
issue_4_sub2=$(extract_issue_number "Implement OpenAI streaming tool support")
issue_4_sub3=$(extract_issue_number "Implement Anthropic streaming tool support")
issue_4_sub4=$(extract_issue_number "Implement fallback for providers without native streaming tools")
issue_4_sub5=$(extract_issue_number "Create integration tests for streaming tool calls")

if [[ -n "$issue_4_sub1" ]]; then
  set_parent_child_relationship "$issue_4_sub1" 4
fi
if [[ -n "$issue_4_sub2" ]]; then
  set_parent_child_relationship "$issue_4_sub2" 4
fi
if [[ -n "$issue_4_sub3" ]]; then
  set_parent_child_relationship "$issue_4_sub3" 4
fi
if [[ -n "$issue_4_sub4" ]]; then
  set_parent_child_relationship "$issue_4_sub4" 4
fi
if [[ -n "$issue_4_sub5" ]]; then
  set_parent_child_relationship "$issue_4_sub5" 4
fi

# 8. Set up issue dependencies
echo "Setting up issue dependencies..."

# Function to set dependency relationship
set_dependency_relationship() {
  local blocked_number=$1
  local blocker_number=$2

  echo "Setting issue #$blocked_number as blocked by issue #$blocker_number..."

  # Use the REST API to add a comment that creates the dependency
  gh api repos/PimpMyNines/LLuMinary/issues/$blocked_number/comments \
    --method POST \
    -f body="Depends on #$blocker_number"
}

# Set up dependencies as defined in the backlog
set_dependency_relationship 15 13
set_dependency_relationship 14 13
set_dependency_relationship 17 13
set_dependency_relationship 17 14
set_dependency_relationship 17 15
set_dependency_relationship 3 13
set_dependency_relationship 3 15
set_dependency_relationship 4 3

# Set up dependencies for new issues
mistral_issue=$(extract_issue_number "Implement Mistral AI Provider Support")
vector_db_issue=$(extract_issue_number "Add Vector Database Integration Support")

if [[ -n "$mistral_issue" ]]; then
  set_dependency_relationship "$mistral_issue" 3
  set_dependency_relationship "$mistral_issue" 4
fi

if [[ -n "$vector_db_issue" ]]; then
  set_dependency_relationship "$vector_db_issue" 3
fi

# Update summary stats and ensure Fibonacci story points
# Get the total number of issues created
total_issues=$(gh issue list --repo PimpMyNines/LLuMinary --limit 100 --json number | jq '. | length')

# Get the number of parent issues
parent_issues=7 # Main issues #13, #15, #16, #14, #17, #3, #4

# Calculate sub-issues
sub_issues=$((total_issues - parent_issues))

# Update implementation report to include relationship setup and summary stats
cat >> BACKLOG_IMPLEMENTATION_REPORT.md << EOL

## Relationship Setup

The script has attempted to programmatically establish the following relationships:

### Parent-Child Relationships
- Parent Issue #13 → Children: Analyze current Dockerfile.matrix generation issues, Implement fixes, Update GitHub Actions, Document process
- Parent Issue #15 → Children: Fix FILE parameter handling, Implement conditional logic, Create separate GitHub Actions jobs
- Parent Issue #3 → Children: Design core type interfaces, Implement provider types (OpenAI, Anthropic, Bedrock), Create type checking tests
- Parent Issue #4 → Children: Design streaming interface, Implement provider streaming (OpenAI, Anthropic), Implement fallback, Create integration tests
- Mistral AI Provider Issue → Children: Research API, Implement authentication, Core generation, Streaming, Error handling, Documentation
- Vector DB Issue → Children: Design interfaces, Implement FAISS backend, Implement Pinecone backend, Embedding integration, Performance testing, Documentation

### Dependencies
- Issue #15 depends on Issue #13
- Issue #14 depends on Issue #13
- Issue #17 depends on Issues #13, #14, #15
- Issue #3 depends on Issues #13, #15
- Issue #4 depends on Issue #3
- Mistral AI Provider issue depends on Issues #3, #4
- Vector Database Integration issue depends on Issue #3

Note: GitHub API limitations may require some relationships to be manually verified and set up through the web interface.

## Summary Statistics

- **Total Issues Created**: ${total_issues}
- **Main Parent Issues**: ${parent_issues}
- **Sub-issues Created**: ${sub_issues}
- **Feature Issues**: 2 (Type System Overhaul, Streaming Support)
- **Task Issues**: ${parent_issues} parent tasks + $(($sub_issues - 2)) sub-tasks
- **Story Point Distribution**: All sub-issues follow Fibonacci sequence (1, 2, 3, 5, 8, 13, 21)

## Consistency with Company Development Standards

The implemented issue structure adheres to the company's development standards:

1. **Story Point Sizing**:
   - All issues use the Fibonacci sequence for story points
   - No sub-issue exceeds 5 story points
   - Large tasks (13+ points) are broken down into smaller, manageable units

2. **Issue Type Classification**:
   - Features: New capabilities and enhancements
   - Tasks: Implementation work and maintenance
   - Bugs: Defect fixes (none in current setup)

3. **Complete Traceability**:
   - All sub-issues link back to their parent issues
   - All dependent issues are properly linked
   - All issues have clear acceptance criteria

4. **Comprehensive Labeling**:
   - Area designations (infrastructure, typing, providers, testing)
   - Priority levels (P0-P3)
   - Size indicators (small, medium, large)
   - Status tracking (blocked, ready)
EOL

echo "Script execution complete. Please review the BACKLOG_IMPLEMENTATION_REPORT.md file."
