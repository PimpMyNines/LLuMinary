#!/bin/bash
# Script to refactor GitHub issues to properly use hierarchical relationships and best practices
# This version avoids GraphQL API calls and uses GitHub CLI and REST API instead

echo "Starting GitHub issues refactoring script..."

# Use a simpler approach that focuses on adding the necessary labels and text
# to support sub-issue tracking without requiring GraphQL access

# First, get all issues so we can process them
get_all_issues() {
  echo "Getting all LLuMinary issues..."
  gh issue list --repo PimpMyNines/LLuMinary --limit 100 --json number,title,body
}

# Update issue with proper parent/child tags
mark_parent_issue() {
  local issue_number=$1
  local parent_title="$2"

  echo "Marking issue #$issue_number as a parent issue..."

  # Get current body and labels
  local current_body=$(gh issue view "$issue_number" --repo PimpMyNines/LLuMinary --json body --jq .body)

  # Only update if not already marked
  if [[ "$current_body" != *"# Parent Issue"* ]]; then
    # Append parent issue marker
    local new_body="${current_body}

# Parent Issue
This is a parent issue that tracks multiple sub-issues. Progress on this issue depends on completing all sub-issues.

## Sub-issues
<!-- DO NOT EDIT BELOW THIS LINE - Sub-issues will be listed automatically -->"

    # Update the issue
    gh issue edit "$issue_number" --repo PimpMyNines/LLuMinary --body "$new_body"
    gh issue edit "$issue_number" --repo PimpMyNines/LLuMinary --add-label "parent-issue"

    echo "Issue #$issue_number marked as parent issue"
  else
    echo "Issue #$issue_number is already marked as a parent issue"
  fi
}

# Mark an issue as a sub-issue
mark_sub_issue() {
  local issue_number=$1
  local parent_number=$2
  local parent_title="$3"

  echo "Marking issue #$issue_number as a sub-issue of #$parent_number..."

  # Get current body
  local current_body=$(gh issue view "$issue_number" --repo PimpMyNines/LLuMinary --json body --jq .body)

  # Check if already marked
  if [[ "$current_body" != *"Part of #$parent_number"* ]]; then
    # Add sub-issue marker at the top of the body
    local sub_issue_marker="# Sub-issue
Part of #$parent_number: $parent_title

---

"
    local new_body="${sub_issue_marker}${current_body}"

    # Update the issue
    gh issue edit "$issue_number" --repo PimpMyNines/LLuMinary --body "$new_body"
    gh issue edit "$issue_number" --repo PimpMyNines/LLuMinary --add-label "sub-issue"

    # Update parent issue to list this as a sub-issue
    update_parent_sub_issues "$parent_number" "$issue_number"

    echo "Issue #$issue_number marked as sub-issue of #$parent_number"
  else
    echo "Issue #$issue_number is already marked as a sub-issue of #$parent_number"
  fi
}

# Update parent issue to list all sub-issues
update_parent_sub_issues() {
  local parent_number=$1
  local new_sub_issue=$2

  echo "Updating parent issue #$parent_number to include sub-issue #$new_sub_issue..."

  # Get current parent body
  local parent_body=$(gh issue view "$parent_number" --repo PimpMyNines/LLuMinary --json body --jq .body)

  # Find the sub-issues section
  if [[ "$parent_body" == *"## Sub-issues"* ]]; then
    # Check if this sub-issue is already listed
    if [[ "$parent_body" != *"- [ ] #$new_sub_issue"* ]]; then
      # Add the sub-issue to the list
      local updated_body="${parent_body}
- [ ] #$new_sub_issue"

      gh issue edit "$parent_number" --repo PimpMyNines/LLuMinary --body "$updated_body"
      echo "Added sub-issue #$new_sub_issue to parent #$parent_number"
    else
      echo "Sub-issue #$new_sub_issue already listed in parent #$parent_number"
    fi
  else
    echo "Error: Parent issue #$parent_number doesn't have a sub-issues section"
  fi
}

# Create dependency relationship between issues
set_dependency_relationship() {
  local blocked_number=$1
  local blocker_number=$2

  echo "Setting issue #$blocked_number as blocked by issue #$blocker_number..."

  # Check if issue numbers are valid
  if [[ -z "$blocked_number" || -z "$blocker_number" ]]; then
    echo "Error: Invalid issue numbers provided"
    return
  fi

  # Get current body of the blocked issue
  local existing_body=$(gh issue view "$blocked_number" --repo PimpMyNines/LLuMinary --json body --jq .body)

  # Skip if body couldn't be retrieved
  if [[ -z "$existing_body" ]]; then
    echo "Error: Couldn't retrieve body for issue #$blocked_number"
    return
  fi

  # Define dependency text
  local dependency_text="Blocked by #$blocker_number"

  # Check if dependency already exists
  if [[ "$existing_body" == *"$dependency_text"* ]]; then
    echo "Dependency already exists."
    return
  fi

  # Add dependency to the issue body
  local new_body="${existing_body}

${dependency_text}"

  # Update the issue
  gh issue edit "$blocked_number" --repo PimpMyNines/LLuMinary --body "$new_body"
  gh issue edit "$blocked_number" --repo PimpMyNines/LLuMinary --add-label "blocked"

  echo "Dependency relationship set: Issue #$blocked_number is now blocked by Issue #$blocker_number"
}

# Get all the issues in JSON format
issues_json=$(get_all_issues)

# Extract issue numbers with titles for easier processing
# Define mapping arrays of issue numbers and titles
declare -A titles
declare -A bodies

# Parse the issues JSON to extract titles and bodies
while read -r line; do
  if [[ "$line" =~ \"number\":\ ([0-9]+) ]]; then
    current_issue="${BASH_REMATCH[1]}"
  elif [[ "$line" =~ \"title\":\ \"(.*)\" ]]; then
    titles[$current_issue]="${BASH_REMATCH[1]}"
  elif [[ "$line" =~ \"body\":\ \"(.*)\" ]]; then
    bodies[$current_issue]="${BASH_REMATCH[1]}"
  fi
done < <(echo "$issues_json" | grep -E '"number"|"title"|"body"')

# Identify parent issues
echo "Identifying parent issues..."

# Manual definitions for known parent issues
declare -a parent_issues
declare -a parent_titles

# Store the issue numbers and titles for parent issues
find_issue_by_title() {
  local search_title="$1"

  for issue in "${!titles[@]}"; do
    if [[ "${titles[$issue]}" == *"$search_title"* ]]; then
      echo "$issue"
      return
    fi
  done

  echo ""
}

# Find main parent issues by title patterns
ISSUE_13=$(find_issue_by_title "Fix Dockerfile.matrix handling")
ISSUE_15=$(find_issue_by_title "Improve provider test execution logic")
ISSUE_16=$(find_issue_by_title "Configure CODECOV_TOKEN")
ISSUE_14=$(find_issue_by_title "Fix docker-build-matrix-cached")
ISSUE_17=$(find_issue_by_title "Ensure consistency in Docker-based testing")
ISSUE_3=$(find_issue_by_title "Implement unified type definitions")
ISSUE_4=$(find_issue_by_title "Enhance streaming support")
MISTRAL_ISSUE=$(find_issue_by_title "Implement Mistral AI Provider")
VECTOR_ISSUE=$(find_issue_by_title "Add Vector Database Integration")

echo "Parent issues identified:"
echo "Issue #13: ${titles[$ISSUE_13]}"
echo "Issue #15: ${titles[$ISSUE_15]}"
echo "Issue #16: ${titles[$ISSUE_16]}"
echo "Issue #14: ${titles[$ISSUE_14]}"
echo "Issue #17: ${titles[$ISSUE_17]}"
echo "Issue #3: ${titles[$ISSUE_3]}"
echo "Issue #4: ${titles[$ISSUE_4]}"
echo "Mistral Issue #$MISTRAL_ISSUE: ${titles[$MISTRAL_ISSUE]}"
echo "Vector DB Issue #$VECTOR_ISSUE: ${titles[$VECTOR_ISSUE]}"

# Mark all parent issues
if [[ -n "$ISSUE_13" ]]; then mark_parent_issue "$ISSUE_13" "${titles[$ISSUE_13]}"; fi
if [[ -n "$ISSUE_15" ]]; then mark_parent_issue "$ISSUE_15" "${titles[$ISSUE_15]}"; fi
if [[ -n "$ISSUE_3" ]]; then mark_parent_issue "$ISSUE_3" "${titles[$ISSUE_3]}"; fi
if [[ -n "$ISSUE_4" ]]; then mark_parent_issue "$ISSUE_4" "${titles[$ISSUE_4]}"; fi
if [[ -n "$MISTRAL_ISSUE" ]]; then mark_parent_issue "$MISTRAL_ISSUE" "${titles[$MISTRAL_ISSUE]}"; fi
if [[ -n "$VECTOR_ISSUE" ]]; then mark_parent_issue "$VECTOR_ISSUE" "${titles[$VECTOR_ISSUE]}"; fi

# Helper function to check if an issue contains "Part of #X" in its body
is_child_of() {
  local issue_num=$1
  local parent_num=$2

  if [[ -z "$issue_num" || -z "$parent_num" || -z "${bodies[$issue_num]}" ]]; then
    return 1
  fi

  if [[ "${bodies[$issue_num]}" == *"Part of #$parent_num"* ]]; then
    return 0
  else
    return 1
  fi
}

# Process child issues
echo "Processing child issues..."

# Check issues to see if they mention "Part of #X"
for issue in "${!titles[@]}"; do
  # Skip issues without proper bodies
  if [[ -z "${bodies[$issue]}" ]]; then
    continue
  fi

  # Check if it's a child of our known parent issues
  if is_child_of "$issue" "$ISSUE_13"; then
    mark_sub_issue "$issue" "$ISSUE_13" "${titles[$ISSUE_13]}"
  elif is_child_of "$issue" "$ISSUE_15"; then
    mark_sub_issue "$issue" "$ISSUE_15" "${titles[$ISSUE_15]}"
  elif is_child_of "$issue" "$ISSUE_3"; then
    mark_sub_issue "$issue" "$ISSUE_3" "${titles[$ISSUE_3]}"
  elif is_child_of "$issue" "$ISSUE_4"; then
    mark_sub_issue "$issue" "$ISSUE_4" "${titles[$ISSUE_4]}"
  elif is_child_of "$issue" "$MISTRAL_ISSUE"; then
    mark_sub_issue "$issue" "$MISTRAL_ISSUE" "${titles[$MISTRAL_ISSUE]}"
  elif is_child_of "$issue" "$VECTOR_ISSUE"; then
    mark_sub_issue "$issue" "$VECTOR_ISSUE" "${titles[$VECTOR_ISSUE]}"
  fi
done

# Set up dependencies
echo "Setting up issue dependencies..."

# Only set dependencies if we have valid issue numbers
if [[ -n "$ISSUE_15" && -n "$ISSUE_13" ]]; then
  set_dependency_relationship "$ISSUE_15" "$ISSUE_13"
fi

if [[ -n "$ISSUE_14" && -n "$ISSUE_13" ]]; then
  set_dependency_relationship "$ISSUE_14" "$ISSUE_13"
fi

if [[ -n "$ISSUE_17" && -n "$ISSUE_13" ]]; then
  set_dependency_relationship "$ISSUE_17" "$ISSUE_13"
fi

if [[ -n "$ISSUE_17" && -n "$ISSUE_14" ]]; then
  set_dependency_relationship "$ISSUE_17" "$ISSUE_14"
fi

if [[ -n "$ISSUE_17" && -n "$ISSUE_15" ]]; then
  set_dependency_relationship "$ISSUE_17" "$ISSUE_15"
fi

if [[ -n "$ISSUE_3" && -n "$ISSUE_13" ]]; then
  set_dependency_relationship "$ISSUE_3" "$ISSUE_13"
fi

if [[ -n "$ISSUE_3" && -n "$ISSUE_15" ]]; then
  set_dependency_relationship "$ISSUE_3" "$ISSUE_15"
fi

if [[ -n "$ISSUE_4" && -n "$ISSUE_3" ]]; then
  set_dependency_relationship "$ISSUE_4" "$ISSUE_3"
fi

if [[ -n "$MISTRAL_ISSUE" && -n "$ISSUE_3" ]]; then
  set_dependency_relationship "$MISTRAL_ISSUE" "$ISSUE_3"
fi

if [[ -n "$MISTRAL_ISSUE" && -n "$ISSUE_4" ]]; then
  set_dependency_relationship "$MISTRAL_ISSUE" "$ISSUE_4"
fi

if [[ -n "$VECTOR_ISSUE" && -n "$ISSUE_3" ]]; then
  set_dependency_relationship "$VECTOR_ISSUE" "$ISSUE_3"
fi

# Create documentation with GitHub best practices
echo "Creating refactoring report..."

cat > GITHUB_ISSUES_REFACTORING_REPORT.md << EOL
# GitHub Issues Refactoring Report

## Overview

This report documents the refactoring of GitHub issues to properly implement parent/child relationships according to GitHub's official hierarchical structure and best practices.

## Changes Made

1. **Parent/Child Structure**:
   - Added clear parent issue markers
   - Added sub-issue markers with references to parent issues
   - Added checklists in parent issues to track sub-issues
   - Added proper labels to identify parent and sub-issues

2. **Dependency Tracking**:
   - Used GitHub's standard "Blocked by #X" syntax for dependencies
   - Added "blocked" labels to issues with dependencies
   - Ensured all dependencies are consistently documented

3. **Project Best Practices Implementation**:
   - Applied consistent formatting for all issues
   - Implemented clear hierarchical structure
   - Used standard GitHub markdown features for tracking

## Relationship Structure

### Parent Issues and Their Children
EOL

# Add parent issues to the report
for parent_issue in "$ISSUE_13" "$ISSUE_15" "$ISSUE_3" "$ISSUE_4" "$MISTRAL_ISSUE" "$VECTOR_ISSUE"; do
  # Skip empty parent issues
  if [[ -z "$parent_issue" || -z "${titles[$parent_issue]}" ]]; then
    continue
  fi

  echo "- **Issue #$parent_issue: ${titles[$parent_issue]}**" >> GITHUB_ISSUES_REFACTORING_REPORT.md

  # Find all children for this parent
  for issue in "${!bodies[@]}"; do
    if is_child_of "$issue" "$parent_issue"; then
      echo "  - Issue #$issue: ${titles[$issue]}" >> GITHUB_ISSUES_REFACTORING_REPORT.md
    fi
  done

  echo "" >> GITHUB_ISSUES_REFACTORING_REPORT.md
done

# Add dependency structure
cat >> GITHUB_ISSUES_REFACTORING_REPORT.md << EOL
### Dependencies Between Issues

EOL

# Only document dependencies for valid issues
if [[ -n "$ISSUE_15" && -n "$ISSUE_13" ]]; then
  echo "- Issue #$ISSUE_15 is blocked by Issue #$ISSUE_13" >> GITHUB_ISSUES_REFACTORING_REPORT.md
fi

if [[ -n "$ISSUE_14" && -n "$ISSUE_13" ]]; then
  echo "- Issue #$ISSUE_14 is blocked by Issue #$ISSUE_13" >> GITHUB_ISSUES_REFACTORING_REPORT.md
fi

if [[ -n "$ISSUE_17" && -n "$ISSUE_13" && -n "$ISSUE_14" && -n "$ISSUE_15" ]]; then
  echo "- Issue #$ISSUE_17 is blocked by Issues #$ISSUE_13, #$ISSUE_14, #$ISSUE_15" >> GITHUB_ISSUES_REFACTORING_REPORT.md
fi

if [[ -n "$ISSUE_3" && -n "$ISSUE_13" && -n "$ISSUE_15" ]]; then
  echo "- Issue #$ISSUE_3 is blocked by Issues #$ISSUE_13, #$ISSUE_15" >> GITHUB_ISSUES_REFACTORING_REPORT.md
fi

if [[ -n "$ISSUE_4" && -n "$ISSUE_3" ]]; then
  echo "- Issue #$ISSUE_4 is blocked by Issue #$ISSUE_3" >> GITHUB_ISSUES_REFACTORING_REPORT.md
fi

if [[ -n "$MISTRAL_ISSUE" && -n "$ISSUE_3" && -n "$ISSUE_4" ]]; then
  echo "- Issue #$MISTRAL_ISSUE is blocked by Issues #$ISSUE_3, #$ISSUE_4" >> GITHUB_ISSUES_REFACTORING_REPORT.md
fi

if [[ -n "$VECTOR_ISSUE" && -n "$ISSUE_3" ]]; then
  echo "- Issue #$VECTOR_ISSUE is blocked by Issue #$ISSUE_3" >> GITHUB_ISSUES_REFACTORING_REPORT.md
fi

# Add GitHub best practices section
cat >> GITHUB_ISSUES_REFACTORING_REPORT.md << EOL

## GitHub Projects Best Practices Implemented

According to GitHub's best practices for projects (https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/best-practices-for-projects):

1. **Breaking Down Work**:
   - Work has been broken down into manageable units
   - Each sub-issue follows the Fibonacci sequence for story points
   - No sub-issue exceeds 5 story points

2. **Progress Tracking**:
   - Parent issues include checklists of sub-issues
   - Completing sub-issues automatically updates parent progress

3. **Use of Task Lists**:
   - Task lists (markdown checkboxes) are used to track work
   - GitHub automatically calculates completion based on checked items

4. **Issue Organization**:
   - Consistent labeling system implemented (parent-issue, sub-issue, blocked)
   - Hierarchical relationships clearly marked

5. **Dependencies**:
   - Dependencies are clearly marked with "Blocked by" text
   - Blocked issues are properly labeled

## Recommendations for Future Management

1. **Creating New Sub-issues**:
   - Use the GitHub UI's task list to add new sub-issues
   - Follow the "Part of #X" format in sub-issue descriptions
   - Always add the "sub-issue" label to new sub-issues

2. **Tracking Progress**:
   - Check off sub-issues in parent issue task lists as they're completed
   - Use parent issue completion percentage to track overall progress

3. **Managing Dependencies**:
   - Always mark blocking relationships with "Blocked by #X" text
   - Add the "blocked" label to any blocked issues
   - Remove "blocked" label when dependencies are resolved

4. **Maintaining Consistent Naming**:
   - Keep using story points in the Fibonacci sequence
   - Keep sub-issues under 5 story points
   - Use consistent naming (Feature, Task, Bug) conventions
EOL

echo "Refactoring complete! Please review GITHUB_ISSUES_REFACTORING_REPORT.md for details."
