# CI/CD Issue Fix Plan

This document provides a prioritized plan for addressing the CI/CD issues in the LLuMinary project.

## Issue Prioritization

### High Priority (Fix First)

1. **[Issue #13: Fix Dockerfile.matrix handling in GitHub Actions workflow](https://github.com/PimpMyNines/LLuMinary/issues/13)**
   - This is blocking CI execution and needs immediate attention
   - Fix how the workflow uses the dynamically created Dockerfile.matrix

2. **[Issue #15: Improve provider test execution logic in CI](https://github.com/PimpMyNines/LLuMinary/issues/15)**
   - Critical for ensuring proper test coverage
   - Fix conditional execution and FILE parameter passing

3. **[Issue #3: Implement unified type definitions across providers](https://github.com/PimpMyNines/LLuMinary/issues/3)**
   - Already in progress
   - Required for consistent code structure
   - Update all provider files to use standard types

### Medium Priority (Fix Second)

4. **[Issue #14: Fix docker-build-matrix-cached target in Makefile](https://github.com/PimpMyNines/LLuMinary/issues/14)**
   - Important for build performance but not blocking functionality
   - Ensure proper caching for faster builds

5. **[Issue #16: Configure CODECOV_TOKEN in GitHub repository secrets](https://github.com/PimpMyNines/LLuMinary/issues/16)**
   - Needed for visibility into test coverage
   - Not blocking core functionality

6. **[Issue #17: Ensure consistency in Docker-based testing setup](https://github.com/PimpMyNines/LLuMinary/issues/17)**
   - Important for developer experience
   - Standardize testing approaches for reliability

### Lower Priority (Fix Later)

7. **[Issue #4: Enhance streaming support for tool/function calling](https://github.com/PimpMyNines/LLuMinary/issues/4)**
   - Feature enhancement rather than bug fix
   - Dependent on type system implementation

## Implementation Plan

### Phase 1: Fix Critical CI Blockers (Issues #13, #15)

1. Review matrix-docker-tests.yml workflow in detail
2. Reproduce issues locally to understand failure modes
3. Fix Dockerfile.matrix handling
4. Fix provider test execution logic
5. Test fixes to ensure they resolve the issues
6. Document changes made for future reference

### Phase 2: Complete Type System Implementation (Issue #3)

1. Finish implementing unified type definitions in types.py
2. Update each provider file one by one to use standard types
3. Add comprehensive type checking tests
4. Ensure mypy type checking works with --strict flag
5. Run full test suite to verify changes

### Phase 3: Improve Build Process (Issues #14, #16, #17)

1. Fix docker-build-matrix-cached target in Makefile
2. Configure CODECOV_TOKEN and verify coverage reporting
3. Standardize Docker-based testing setup
4. Create developer documentation for local testing
5. Verify all CI/CD flows work end to end

### Phase 4: Enhance Features (Issue #4)

1. Design unified streaming tool calls interface
2. Update types.py to include streaming tool call types
3. Implement streaming tool calls for each provider
4. Create tests for streaming tool calls

## Success Criteria

- All GitHub Actions workflows pass successfully
- Local and CI testing environments provide consistent results
- All tests pass with good coverage metrics
- Type system provides consistent interfaces across providers
- Developer documentation is clear and up-to-date

## Time Estimates

- Phase 1: 1-2 days
- Phase 2: 2-3 days
- Phase 3: 1-2 days
- Phase 4: 2-3 days

Total estimated time: 6-10 days of focused work
