#!/bin/bash
# Script to delete all existing issues and start fresh

# Check if GITHUB_TOKEN is set
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Error: GITHUB_TOKEN environment variable is not set."
  echo "Please set it with: export GITHUB_TOKEN='your_github_token'"
  exit 1
fi

# Get repository information from git config
REPO_OWNER=$(git config --get remote.origin.url | sed -n 's/.*github.com[:/]\([^/]*\)\/\([^.]*\).*/\1/p')
REPO_NAME=$(git config --get remote.origin.url | sed -n 's/.*github.com[:/]\([^/]*\)\/\([^.]*\).*/\2/p')

# If we couldn't get the repo info from git, use default
if [ -z "$REPO_OWNER" ] || [ -z "$REPO_NAME" ]; then
  echo "Warning: Could not determine repository from git config."
  echo "Using default repository: PimpMyNines/LLuMinary"
  REPO="PimpMyNines/LLuMinary"
else
  REPO="$REPO_OWNER/$REPO_NAME"
  echo "Using repository: $REPO"
fi

# Default values
DRY_RUN=false
BACKUP_FILE="issues_backup_$(date +%Y%m%d_%H%M%S).json"
ISSUE_STRUCTURE_FILE=""
SKIP_CONFIRMATION=false
LINK_ONLY=false
PARENT_ISSUE=""
LINK_ALL=false

# Check for organization name in environment variable
if [ -z "$ORG_NAME" ] && [ ! -z "$GITHUB_ORG_NAME" ]; then
  ORG_NAME="$GITHUB_ORG_NAME"
  echo "Using organization from GITHUB_ORG_NAME environment variable: $ORG_NAME"
fi

# Function to display help message
show_help() {
  echo "Usage: $0 [options]"
  echo "Delete all existing issues and start fresh with correctly created issues and subtasks."
  echo ""
  echo "Options:"
  echo "  -d, --dry-run              Show what would be done without making changes"
  echo "  -b, --backup <file>        Backup file to save existing issues (default: $BACKUP_FILE)"
  echo "  -s, --structure <file>     JSON file with new issue structure to create"
  echo "  -o, --org <name>           GitHub organization name (required for Sub-issues API)"
  echo "  -y, --yes                  Skip confirmation prompt"
  echo "  -l, --link-only <issue>    Only run the sub-issue linking step for the specified parent issue"
  echo "  -a, --link-all             Find all potential parent issues and link their sub-issues"
  echo "  -h, --help                 Display this help message"
  echo ""
  echo "Environment Variables:"
  echo "  GITHUB_TOKEN               Required: GitHub Personal Access Token for API access"
  echo "  GITHUB_ORG_NAME            Optional: GitHub organization name (alternative to -o/--org)"
  echo ""
  echo "Examples:"
  echo "  $0 -d                      Dry run to see what would be deleted"
  echo "  $0 -b my_backup.json       Backup issues to my_backup.json before deleting"
  echo "  $0 -s new_structure.json   Create new issues from structure in new_structure.json"
  echo "  $0 -o myorg                Specify GitHub organization 'myorg' for API access"
  echo "  $0 -l 123                  Only link sub-issues for parent issue #123"
  echo "  $0 -a                      Find all parent issues and link their sub-issues"
  echo "  $0 -y                      Skip confirmation prompt and proceed with deletion"
  exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -d|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -b|--backup)
      BACKUP_FILE="$2"
      shift
      shift
      ;;
    -s|--structure)
      ISSUE_STRUCTURE_FILE="$2"
      shift
      shift
      ;;
    -o|--org)
      ORG_NAME="$2"
      shift
      shift
      ;;
    -l|--link-only)
      LINK_ONLY=true
      PARENT_ISSUE="$2"
      shift
      shift
      ;;
    -a|--link-all)
      LINK_ALL=true
      shift
      ;;
    -y|--yes)
      SKIP_CONFIRMATION=true
      shift
      ;;
    -h|--help)
      show_help
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      ;;
  esac
done

# Function to backup all issues
backup_issues() {
  echo "Backing up all issues to $BACKUP_FILE..."

  # Get all open issues
  open_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues?state=open&per_page=100")

  # Get all closed issues
  closed_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues?state=closed&per_page=100")

  # Combine open and closed issues
  all_issues=$(echo '[]' | jq --argjson open "$open_issues" --argjson closed "$closed_issues" '. + $open + $closed')

  # Save to backup file
  echo "$all_issues" > "$BACKUP_FILE"

  # Count issues
  issue_count=$(echo "$all_issues" | jq 'length')
  echo "Backed up $issue_count issues to $BACKUP_FILE"
}

# Function to close all issues
close_all_issues() {
  echo "Closing all issues..."

  # Get all open issues
  open_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues?state=open&per_page=100")

  # Count issues
  issue_count=$(echo "$open_issues" | jq 'length')
  echo "Found $issue_count open issues"

  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would close $issue_count issues"
    return
  fi

  # Close each issue
  for i in $(seq 0 $(($issue_count - 1))); do
    issue_number=$(echo "$open_issues" | jq -r ".[$i].number")
    issue_title=$(echo "$open_issues" | jq -r ".[$i].title")

    echo "Closing issue #$issue_number: $issue_title"

    curl -s -X PATCH -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues/$issue_number" \
      -d '{"state":"closed", "state_reason":"not_planned"}'
  done

  echo "Closed $issue_count issues"
}

# Function to create new issues from structure file
create_new_issues() {
  if [ -z "$ISSUE_STRUCTURE_FILE" ]; then
    echo "No issue structure file provided. Skipping issue creation."
    return
  fi

  if [ ! -f "$ISSUE_STRUCTURE_FILE" ]; then
    echo "Error: Issue structure file '$ISSUE_STRUCTURE_FILE' not found."
    exit 1
  fi

  echo "Creating new issues from $ISSUE_STRUCTURE_FILE..."

  # Read issue structure file
  issue_structure=$(cat "$ISSUE_STRUCTURE_FILE")

  # Count parent issues
  parent_count=$(echo "$issue_structure" | jq '.parent_issues | length')
  echo "Found $parent_count parent issues to create"

  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would create $parent_count parent issues and their sub-issues"
    return
  fi

  # Create each parent issue and its sub-issues
  for i in $(seq 0 $(($parent_count - 1))); do
    parent=$(echo "$issue_structure" | jq -r ".parent_issues[$i]")
    parent_title=$(echo "$parent" | jq -r '.title')
    parent_body=$(echo "$parent" | jq -r '.body')
    parent_labels=$(echo "$parent" | jq -c '.labels')

    echo "Creating parent issue: $parent_title"

    # Create parent issue using jq to properly escape JSON values
    payload=$(jq -n \
      --arg title "$parent_title" \
      --arg body "$parent_body" \
      --argjson labels "$parent_labels" \
      '{title: $title, body: $body, labels: $labels}')
    
    parent_response=$(curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues" \
      -d "$payload")

    parent_number=$(echo "$parent_response" | jq -r '.number')

    # Check if parent issue was created successfully
    if [ "$parent_number" = "null" ]; then
      echo "Error creating parent issue: $(echo "$parent_response" | jq -r '.message')"
      # Add more detailed error information for debugging
      echo "Error details: $(echo "$parent_response" | jq '.')"
      echo "Attempted payload: $(echo "$payload" | jq '.')"
      continue
    fi

    echo "Created parent issue #$parent_number"

    # Create sub-issues
    sub_issues=$(echo "$parent" | jq -c '.sub_issues[]')
    sub_count=$(echo "$parent" | jq '.sub_issues | length')
    echo "Found $sub_count sub-issues to create for parent #$parent_number"

    # Create a temporary file to store sub-issue numbers
    sub_numbers_file=$(mktemp)

    # Process each sub-issue
    echo "$sub_issues" | while read -r sub_issue; do
      sub_title=$(echo "$sub_issue" | jq -r '.title')
      sub_body=$(echo "$sub_issue" | jq -r '.body')
      sub_labels=$(echo "$sub_issue" | jq -c '.labels')

      # Add parent reference to body
      sub_body="Part of #$parent_number: $parent_title\n\n$sub_body"

      echo "Creating sub-issue: $sub_title"

      # Create sub-issue using jq to properly escape JSON values
      sub_payload=$(jq -n \
        --arg title "$sub_title" \
        --arg body "$sub_body" \
        --argjson labels "$sub_labels" \
        '{title: $title, body: $body, labels: $labels}')
      
      sub_response=$(curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "https://api.github.com/repos/$REPO/issues" \
        -d "$sub_payload")

      sub_number=$(echo "$sub_response" | jq -r '.number')

      # Check if sub-issue was created successfully
      if [ "$sub_number" = "null" ]; then
        echo "Error creating sub-issue: $(echo "$sub_response" | jq -r '.message')"
        # Add more detailed error information for debugging
        echo "Error details: $(echo "$sub_response" | jq '.')"
        echo "Attempted payload: $(echo "$sub_payload" | jq '.')"
        continue
      fi

      echo "Created sub-issue #$sub_number"

      # Add to temporary file with parent number for reference
      echo "$sub_number:$parent_number" >> "$sub_numbers_file"
    done

    # Wait a moment to ensure all issues are created
    sleep 2

    # Now try to add all sub-issues using the sub-issues API
    echo "Attempting to link sub-issues to parent #$parent_number using the Sub-issues API..."

    # Set up headers including organization if specified
    API_HEADERS=("-H" "Authorization: token $GITHUB_TOKEN" \
                 "-H" "Accept: application/vnd.github+json" \
                 "-H" "X-GitHub-Api-Version: 2022-11-28")
    
    # Add organization header if specified
    if [ ! -z "$ORG_NAME" ]; then
      echo "Using organization: $ORG_NAME"
      API_HEADERS+=("-H" "X-GitHub-Organization: $ORG_NAME")
    fi

    # First, check if the Sub-issues API is available by trying to list existing sub-issues
    api_check=$(curl -s -o /dev/null -w "%{http_code}" "${API_HEADERS[@]}" \
      "https://api.github.com/repos/$REPO/issues/$parent_number/sub_issues")

    if [ "$api_check" = "404" ] || [ "$api_check" = "403" ]; then
      echo "Warning: Sub-issues API is not available (HTTP $api_check). This feature requires:"
      echo "  - GitHub organization account"
      echo "  - Sub-issues feature to be enabled for the organization"
      echo "  - Token with appropriate permissions"
      echo "Using body references instead. Sub-issues will appear as regular issues with 'Part of #' references."
    else
      echo "Sub-issues API is available. Linking sub-issues..."

      # Link each sub-issue to the parent
      linking_failed=false
      linking_error=""
      
      while IFS=: read -r sub_number parent_num; do
        echo "Linking sub-issue #$sub_number to parent #$parent_num"

        # Use jq to properly create JSON payload for the sub-issue linking
        sub_link_payload=$(jq -n --arg id "$sub_number" '{sub_issue_id: ($id|tonumber)}')
        
        sub_api_response=$(curl -s -X POST "${API_HEADERS[@]}" \
          "https://api.github.com/repos/$REPO/issues/$parent_num/sub_issues" \
          -d "$sub_link_payload")

        # Check for errors
        error_message=$(echo "$sub_api_response" | jq -r '.message // empty')
        if [ ! -z "$error_message" ]; then
          echo "Error linking sub-issue: $error_message"
          
          # Store the error for later handling
          linking_failed=true
          linking_error="$error_message"
          
          # If it's an access error, break out of the loop early
          if [[ "$error_message" == *"not accessible by"* || "$error_message" == *"permission"* ]]; then
            echo "Authorization error detected. Stopping sub-issue linking attempts."
            break
          fi
        else
          echo "Successfully linked sub-issue #$sub_number to parent #$parent_num"
        fi
      done < "$sub_numbers_file"
      
      # Provide guidance if linking failed due to permissions
      if [ "$linking_failed" = true ]; then
        echo ""
        echo "Note: Failed to link some or all sub-issues."
        
        if [[ "$linking_error" == *"not accessible by"* || "$linking_error" == *"permission"* ]]; then
          echo "The sub-issues API requires a Personal Access Token with organization permissions."
          echo "To fix this issue:"
          echo "1. Create a new token at https://github.com/settings/tokens with org:write scope"
          echo "2. For organization repositories, ensure you have granted the token access to your organization"
          echo "3. IMPORTANT: Use the -o/--org parameter to specify your organization name"
          echo "   Example: $0 -s $ISSUE_STRUCTURE_FILE -o YOUR_ORG_NAME"
          echo "4. Export the new token as GITHUB_TOKEN and run the script again"
          echo ""
          echo "Sub-issues will still appear as regular issues with 'Part of #' references in the description."
        fi
      fi
    fi

    # Clean up temporary file
    rm -f "$sub_numbers_file"
  done

  echo "Created $parent_count parent issues and their sub-issues"
}

# Function to create a relationship between issues using comments when the Sub-issues API fails
create_issue_relationship_by_comment() {
  parent_num="$1"
  sub_num="$2"
  parent_title="$3"
  sub_title="$4"
  
  echo "Creating issue relationship through comments (parent #$parent_num, sub-issue #$sub_num)..."
  
  # Get parent issue
  parent_info=$(curl -s \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues/$parent_num")
  
  # Get sub-issue
  sub_info=$(curl -s \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues/$sub_num")
  
  # First, check if parent issue exists
  parent_state=$(echo "$parent_info" | jq -r '.state // "unknown"')
  if [ "$parent_state" = "unknown" ] || [ "$parent_state" = "null" ]; then
    echo "Error: Parent issue #$parent_num does not exist or cannot be accessed."
    return 1
  fi
  
  # Then check if sub-issue exists
  sub_state=$(echo "$sub_info" | jq -r '.state // "unknown"')
  if [ "$sub_state" = "unknown" ] || [ "$sub_state" = "null" ]; then
    echo "Error: Sub-issue #$sub_num does not exist or cannot be accessed."
    return 1
  fi
  
  # If titles weren't provided, get them from the API response
  if [ -z "$parent_title" ]; then
    parent_title=$(echo "$parent_info" | jq -r '.title // "Unknown Issue"')
  fi
  
  if [ -z "$sub_title" ]; then
    sub_title=$(echo "$sub_info" | jq -r '.title // "Unknown Issue"')
  fi
  
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would add relationship comments to issues #$parent_num and #$sub_num"
    return 0
  fi
  
  # Create comment on parent issue
  parent_comment_payload=$(jq -n \
    --arg body "📋 **Related sub-issue:** #$sub_num ($sub_title)" \
    '{body: $body}')
  
  parent_comment_response=$(curl -s -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues/$parent_num/comments" \
    -d "$parent_comment_payload")
  
  # Create comment on sub-issue
  sub_comment_payload=$(jq -n \
    --arg body "🔍 **Related to parent issue:** #$parent_num ($parent_title)" \
    '{body: $body}')
  
  sub_comment_response=$(curl -s -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues/$sub_num/comments" \
    -d "$sub_comment_payload")
  
  # Check for errors in parent comment
  parent_error=$(echo "$parent_comment_response" | jq -r '.message // empty')
  if [ ! -z "$parent_error" ]; then
    echo "Warning: Failed to add comment to parent issue #$parent_num: $parent_error"
  else
    echo "Added relationship comment to parent issue #$parent_num"
  fi
  
  # Check for errors in sub-issue comment
  sub_error=$(echo "$sub_comment_response" | jq -r '.message // empty')
  if [ ! -z "$sub_error" ]; then
    echo "Warning: Failed to add comment to sub-issue #$sub_num: $sub_error"
  else
    echo "Added relationship comment to sub-issue #$sub_num"
  fi
  
  # Return success if at least one comment was created successfully
  if [ -z "$parent_error" ] || [ -z "$sub_error" ]; then
    return 0
  else
    return 1
  fi
}

# Function to link sub-issues to a parent issue
link_sub_issues() {
  parent_num="$1"
  echo "Attempting to link sub-issues to parent #$parent_num..."
  
  # Set up headers including organization if specified
  API_HEADERS=("-H" "Authorization: token $GITHUB_TOKEN" \
               "-H" "Accept: application/vnd.github+json" \
               "-H" "X-GitHub-Api-Version: 2022-11-28")
  
  # Add organization header if specified
  if [ ! -z "$ORG_NAME" ]; then
    echo "Using organization: $ORG_NAME"
    API_HEADERS+=("-H" "X-GitHub-Organization: $ORG_NAME")
  fi

  # First, check if the Sub-issues API is available by trying to list existing sub-issues
  api_check=$(curl -s -o /dev/null -w "%{http_code}" "${API_HEADERS[@]}" \
    "https://api.github.com/repos/$REPO/issues/$parent_num/sub_issues")

  # Get parent issue title for fallback mechanism
  parent_info=$(curl -s \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues/$parent_num")
  parent_title=$(echo "$parent_info" | jq -r '.title // "Unknown Issue"')

  if [ "$api_check" = "404" ] || [ "$api_check" = "403" ]; then
    echo "Warning: Sub-issues API is not available (HTTP $api_check). This feature requires:"
    echo "  - GitHub organization account"
    echo "  - Sub-issues feature to be enabled for the organization"
    echo "  - Token with appropriate permissions"
    echo "  - Organization specified with -o/--org or GITHUB_ORG_NAME environment variable"
    echo ""
    echo "Using fallback mechanism to create issue relationships with comments instead."
    
    # Get all issues that reference this parent issue
    all_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues?state=open&per_page=100")
    
    # Create a temporary file to store sub-issue numbers
    sub_numbers_file=$(mktemp)
    
    # Extract sub-issues that mention the parent issue
    echo "$all_issues" | jq -r ".[] | select(.body | contains(\"Part of #$parent_num\")) | .number" > "$sub_numbers_file"
    
    # Count sub-issues found
    sub_count=$(wc -l < "$sub_numbers_file")
    sub_count=$(echo $sub_count | tr -d ' ') # Remove any whitespace
    
    if [ "$sub_count" -eq 0 ]; then
      echo "No sub-issues found referencing parent #$parent_num."
      rm -f "$sub_numbers_file"
      return
    fi
    
    echo "Found $sub_count potential sub-issues to link to parent #$parent_num"
    
    # Link each sub-issue to the parent using comments
    successful_links=0
    
    while read -r sub_number; do
      # Get sub-issue title
      sub_info=$(curl -s \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "https://api.github.com/repos/$REPO/issues/$sub_number")
      
      sub_title=$(echo "$sub_info" | jq -r '.title // "Unknown Issue"')
      
      if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would create relationship between parent #$parent_num and sub-issue #$sub_number using comments"
        continue
      fi
      
      # Create relationship using comments
      if create_issue_relationship_by_comment "$parent_num" "$sub_number" "$parent_title" "$sub_title"; then
        successful_links=$((successful_links + 1))
      fi
    done < "$sub_numbers_file"
    
    # Clean up temporary file
    rm -f "$sub_numbers_file"
    
    echo "Created $successful_links issue relationships using comments."
  else
    echo "Sub-issues API is available. Retrieving potential sub-issues..."

    # Get all issues that reference this parent issue
    all_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues?state=open&per_page=100")
    
    # Create a temporary file to store sub-issue numbers
    sub_numbers_file=$(mktemp)
    
    # Extract sub-issues that mention the parent issue
    echo "$all_issues" | jq -r ".[] | select(.body | contains(\"Part of #$parent_num\")) | .number" > "$sub_numbers_file"
    
    # Count sub-issues found
    sub_count=$(wc -l < "$sub_numbers_file")
    sub_count=$(echo $sub_count | tr -d ' ') # Remove any whitespace
    
    if [ "$sub_count" -eq 0 ]; then
      echo "No sub-issues found referencing parent #$parent_num."
      rm -f "$sub_numbers_file"
      return
    fi
    
    echo "Found $sub_count potential sub-issues to link to parent #$parent_num"

    # Link each sub-issue to the parent
    linking_failed=false
    linking_error=""
    
    while read -r sub_number; do
      # Verify that the sub-issue exists
      echo "Verifying that sub-issue #$sub_number exists..."
      issue_check=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "https://api.github.com/repos/$REPO/issues/$sub_number")
      
      if [ "$issue_check" != "200" ]; then
        echo "Warning: Sub-issue #$sub_number cannot be accessed (HTTP $issue_check). Skipping."
        continue
      fi
      
      echo "Linking sub-issue #$sub_number to parent #$parent_num"
      
      if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would link sub-issue #$sub_number to parent #$parent_num"
        continue
      fi
      
      # Use jq to properly create JSON payload for the sub-issue linking
      sub_link_payload=$(jq -n --arg id "$sub_number" '{sub_issue_id: ($id|tonumber)}')
      
      sub_api_response=$(curl -s -X POST "${API_HEADERS[@]}" \
        "https://api.github.com/repos/$REPO/issues/$parent_num/sub_issues" \
        -d "$sub_link_payload")
      
      # Check for errors
      error_message=$(echo "$sub_api_response" | jq -r '.message // empty')
      if [ ! -z "$error_message" ]; then
        echo "Error linking sub-issue: $error_message"
        
        # Store the error for later handling
        linking_failed=true
        linking_error="$error_message"
        
        # Handle specific error types
        if [[ "$error_message" == *"not accessible by"* || "$error_message" == *"permission"* ]]; then
          echo "Authorization error detected. Stopping sub-issue linking attempts."
          break
        elif [[ "$error_message" == *"sub-issue does not exist"* ]]; then
          echo "The GitHub API cannot find sub-issue #$sub_number. This may be due to caching or recent creation."
          echo "Trying fallback method to get more information..."
          
          # Get more information about the issue
          issue_info=$(curl -s \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            "https://api.github.com/repos/$REPO/issues/$sub_number")
          
          issue_state=$(echo "$issue_info" | jq -r '.state // "unknown"')
          issue_title=$(echo "$issue_info" | jq -r '.title // "unknown"')
          
          echo "Issue #$sub_number info - Title: $issue_title, State: $issue_state"
          echo "Waiting 5 seconds before retrying..."
          sleep 5
          
          # Retry the linking operation once
          echo "Retrying link operation for sub-issue #$sub_number..."
          retry_response=$(curl -s -X POST "${API_HEADERS[@]}" \
            "https://api.github.com/repos/$REPO/issues/$parent_num/sub_issues" \
            -d "$sub_link_payload")
            
          retry_error=$(echo "$retry_response" | jq -r '.message // empty')
          if [ -z "$retry_error" ]; then
            echo "Successfully linked sub-issue #$sub_number to parent #$parent_num on retry"
          else
            echo "Retry failed: $retry_error"
          fi
        fi
      else
        echo "Successfully linked sub-issue #$sub_number to parent #$parent_num"
      fi
    done < "$sub_numbers_file"
    
    # Provide guidance if linking failed due to permissions
    if [ "$linking_failed" = true ]; then
      echo ""
      echo "Note: Failed to link some or all sub-issues."
      
      if [[ "$linking_error" == *"not accessible by"* || "$linking_error" == *"permission"* ]]; then
        echo "The sub-issues API requires a Personal Access Token with organization permissions."
        echo "To fix this issue:"
        echo "1. Create a new token at https://github.com/settings/tokens with org:write scope"
        echo "2. For organization repositories, ensure you have granted the token access to your organization"
        echo "3. IMPORTANT: Use the -o/--org parameter or set GITHUB_ORG_NAME environment variable"
        echo "   Example: $0 -l $parent_num -o YOUR_ORG_NAME"
        echo "   or: export GITHUB_ORG_NAME=YOUR_ORG_NAME && $0 -l $parent_num"
      fi
    else
      echo "Successfully completed linking sub-issues to parent #$parent_num."
    fi

    # Clean up temporary file
    rm -f "$sub_numbers_file"
  fi
}

# Function to find and link all parent issues and their sub-issues
find_and_link_all_sub_issues() {
  echo "Finding all potential parent issues and their sub-issues..."
  
  # Set up headers including organization if specified
  API_HEADERS=("-H" "Authorization: token $GITHUB_TOKEN" \
              "-H" "Accept: application/vnd.github+json" \
              "-H" "X-GitHub-Api-Version: 2022-11-28")
  
  # Add organization header if specified
  if [ ! -z "$ORG_NAME" ]; then
    echo "Using organization: $ORG_NAME"
    API_HEADERS+=("-H" "X-GitHub-Organization: $ORG_NAME")
  fi

  # First, check if the Sub-issues API is available by trying to list all issues
  api_check=$(curl -s -o /dev/null -w "%{http_code}" "${API_HEADERS[@]}" \
    "https://api.github.com/repos/$REPO/issues")

  if [ "$api_check" = "404" ] || [ "$api_check" = "403" ]; then
    echo "Error: GitHub API is not accessible (HTTP $api_check). Please check your token permissions."
    exit 1
  fi

  # Get all open issues
  echo "Retrieving all open issues..."
  all_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/$REPO/issues?state=open&per_page=100")
  
  # Count issues
  issue_count=$(echo "$all_issues" | jq 'length')
  echo "Found $issue_count open issues"
  
  if [ "$issue_count" -eq 0 ]; then
    echo "No open issues found. Nothing to do."
    return
  fi
  
  # Create temporary files
  potential_parents_file=$(mktemp)
  sub_issues_file=$(mktemp)
  
  # Identify potential parent issues (those with checklist items)
  echo "$all_issues" | jq -r '.[] | select(.body | contains("- [ ]") or contains("* [ ]")) | .number' > "$potential_parents_file"
  
  # Count potential parent issues
  parent_count=$(wc -l < "$potential_parents_file")
  parent_count=$(echo $parent_count | tr -d ' ') # Remove any whitespace
  
  if [ "$parent_count" -eq 0 ]; then
    echo "No potential parent issues found (issues with checklists/tasks)."
    rm -f "$potential_parents_file" "$sub_issues_file"
    return
  fi
  
  echo "Found $parent_count potential parent issues with checklists"
  
  # Process each potential parent issue
  cat "$potential_parents_file" | while read -r parent_num; do
    parent_title=$(echo "$all_issues" | jq -r ".[] | select(.number == $parent_num) | .title")
    echo ""
    echo "Processing potential parent issue #$parent_num: $parent_title"
    
    # Extract all sub-issues that reference this parent issue
    echo "$all_issues" | jq -r ".[] | select(.body | contains(\"Part of #$parent_num\")) | .number" > "$sub_issues_file"
    
    # Count sub-issues found
    sub_count=$(wc -l < "$sub_issues_file")
    sub_count=$(echo $sub_count | tr -d ' ') # Remove any whitespace
    
    if [ "$sub_count" -eq 0 ]; then
      echo "No sub-issues found referencing parent #$parent_num."
      continue
    fi
    
    echo "Found $sub_count potential sub-issues referencing parent #$parent_num"
    
    # Check if the Sub-issues API is available for this parent
    parent_api_check=$(curl -s -o /dev/null -w "%{http_code}" "${API_HEADERS[@]}" \
      "https://api.github.com/repos/$REPO/issues/$parent_num/sub_issues")
    
    if [ "$parent_api_check" = "404" ] || [ "$parent_api_check" = "403" ]; then
      echo "Warning: Sub-issues API is not available for this parent issue (HTTP $parent_api_check)."
      echo "Falling back to creating issue relationships with comments instead."
      
      # Link each sub-issue to the parent using comments
      successful_links=0
      
      cat "$sub_issues_file" | while read -r sub_number; do
        # Get sub-issue title
        sub_info=$(curl -s \
          -H "Authorization: token $GITHUB_TOKEN" \
          -H "Accept: application/vnd.github+json" \
          -H "X-GitHub-Api-Version: 2022-11-28" \
          "https://api.github.com/repos/$REPO/issues/$sub_number")
        
        sub_title=$(echo "$sub_info" | jq -r '.title // "Unknown Issue"')
        
        if [ "$DRY_RUN" = true ]; then
          echo "[DRY RUN] Would create relationship between parent #$parent_num and sub-issue #$sub_number using comments"
          continue
        fi
        
        # Create relationship using comments
        if create_issue_relationship_by_comment "$parent_num" "$sub_number" "$parent_title" "$sub_title"; then
          successful_links=$((successful_links + 1))
        fi
      done
      
      echo "Created $successful_links issue relationships using comments."
      continue
    fi
    
    echo "Sub-issues API is available for parent #$parent_num. Proceeding with linking..."
    
    # Get existing sub-issues to avoid duplicates
    existing_sub_issues=$(curl -s "${API_HEADERS[@]}" \
      "https://api.github.com/repos/$REPO/issues/$parent_num/sub_issues")
    
    # Link each sub-issue to the parent if not already linked
    linking_failed=false
    linking_error=""
    successful_links=0
    
    cat "$sub_issues_file" | while read -r sub_number; do
      # Check if already linked
      is_linked=$(echo "$existing_sub_issues" | jq -r "any(.sub_issue.number == $sub_number)")
      
      if [ "$is_linked" = "true" ]; then
        echo "Sub-issue #$sub_number is already linked to parent #$parent_num. Skipping."
        continue
      fi
      
      # Verify that the sub-issue exists
      echo "Verifying that sub-issue #$sub_number exists..."
      issue_check=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "https://api.github.com/repos/$REPO/issues/$sub_number")
      
      if [ "$issue_check" != "200" ]; then
        echo "Warning: Sub-issue #$sub_number cannot be accessed (HTTP $issue_check). Skipping."
        continue
      fi
      
      echo "Linking sub-issue #$sub_number to parent #$parent_num"
      
      if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would link sub-issue #$sub_number to parent #$parent_num"
        continue
      fi
      
      # Use jq to properly create JSON payload for the sub-issue linking
      sub_link_payload=$(jq -n --arg id "$sub_number" '{sub_issue_id: ($id|tonumber)}')
      
      sub_api_response=$(curl -s -X POST "${API_HEADERS[@]}" \
        "https://api.github.com/repos/$REPO/issues/$parent_num/sub_issues" \
        -d "$sub_link_payload")
      
      # Check for errors
      error_message=$(echo "$sub_api_response" | jq -r '.message // empty')
      if [ ! -z "$error_message" ]; then
        echo "Error linking sub-issue: $error_message"
        
        # Store the error for later handling
        linking_failed=true
        linking_error="$error_message"
        
        # Handle specific error types
        if [[ "$error_message" == *"not accessible by"* || "$error_message" == *"permission"* ]]; then
          echo "Authorization error detected. Stopping sub-issue linking attempts for this parent."
          break
        elif [[ "$error_message" == *"sub-issue does not exist"* ]]; then
          echo "The GitHub API cannot find sub-issue #$sub_number. This may be due to caching or recent creation."
          echo "Trying fallback method to get more information..."
          
          # Get more information about the issue
          issue_info=$(curl -s \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            "https://api.github.com/repos/$REPO/issues/$sub_number")
          
          issue_state=$(echo "$issue_info" | jq -r '.state // "unknown"')
          issue_title=$(echo "$issue_info" | jq -r '.title // "unknown"')
          
          echo "Issue #$sub_number info - Title: $issue_title, State: $issue_state"
          echo "Waiting 5 seconds before retrying..."
          sleep 5
          
          # Retry the linking operation once
          echo "Retrying link operation for sub-issue #$sub_number..."
          retry_response=$(curl -s -X POST "${API_HEADERS[@]}" \
            "https://api.github.com/repos/$REPO/issues/$parent_num/sub_issues" \
            -d "$sub_link_payload")
            
          retry_error=$(echo "$retry_response" | jq -r '.message // empty')
          if [ -z "$retry_error" ]; then
            echo "Successfully linked sub-issue #$sub_number to parent #$parent_num on retry"
            successful_links=$((successful_links + 1))
          else
            echo "Retry failed: $retry_error"
          fi
        fi
      else
        echo "Successfully linked sub-issue #$sub_number to parent #$parent_num"
        successful_links=$((successful_links + 1))
      fi
    done
    
    # Report on linking results for this parent
    if [ "$successful_links" -gt 0 ]; then
      echo "Successfully linked $successful_links sub-issues to parent #$parent_num"
    elif [ "$linking_failed" = true ]; then
      echo "Failed to link any sub-issues to parent #$parent_num"
      
      if [[ "$linking_error" == *"not accessible by"* || "$linking_error" == *"permission"* ]]; then
        echo "The sub-issues API requires a Personal Access Token with organization permissions."
        echo "To fix this issue:"
        echo "1. Create a new token at https://github.com/settings/tokens with org:write scope"
        echo "2. For organization repositories, ensure you have granted the token access to your organization"
        echo "3. IMPORTANT: Use the -o/--org parameter or set GITHUB_ORG_NAME environment variable"
        echo "   Example: $0 -a -o YOUR_ORG_NAME"
        echo "   or: export GITHUB_ORG_NAME=YOUR_ORG_NAME && $0 -a"
      fi
    fi
  done
  
  # Clean up temporary files
  rm -f "$potential_parents_file" "$sub_issues_file"
  
  echo ""
  echo "Completed processing all potential parent issues."
}

# Main execution
echo "=== LLuMinary GitHub Issue Reset Tool ==="

if [ "$LINK_ALL" = true ]; then
  echo "Running in find-and-link-all mode to discover and link all sub-issues"
  echo "Repository: $REPO"
  
  # Confirm before proceeding
  if [ "$SKIP_CONFIRMATION" = false ] && [ "$DRY_RUN" = false ]; then
    echo ""
    echo "This will find all potential parent issues and link their sub-issues."
    echo ""
    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Operation cancelled."
      exit 1
    fi
  fi
  
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would find and link all sub-issues"
  fi
  
  find_and_link_all_sub_issues
  
  echo ""
  echo "Find-and-link-all operation completed."
elif [ "$LINK_ONLY" = true ]; then
  echo "Running in link-only mode for parent issue #$PARENT_ISSUE"
  echo "Repository: $REPO"
  
  # Confirm before proceeding
  if [ "$SKIP_CONFIRMATION" = false ] && [ "$DRY_RUN" = false ]; then
    echo ""
    echo "This will attempt to link sub-issues to parent issue #$PARENT_ISSUE."
    echo ""
    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Operation cancelled."
      exit 1
    fi
  fi
  
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would link sub-issues to parent issue #$PARENT_ISSUE"
  else
    link_sub_issues "$PARENT_ISSUE"
    echo ""
    echo "Sub-issue linking completed."
  fi
else
  echo "This script will delete all existing issues and start fresh."
  echo "Repository: $REPO"

  # Backup issues
  backup_issues

  # Confirm before proceeding
  if [ "$SKIP_CONFIRMATION" = false ] && [ "$DRY_RUN" = false ]; then
    echo ""
    echo "WARNING: This will close ALL issues in the repository."
    echo "A backup has been created at: $BACKUP_FILE"
    echo ""
    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      echo "Operation cancelled."
      exit 1
    fi
  fi

  # Close all issues
  close_all_issues

  # Check if organization parameter is needed but not provided
  if [ -z "$ORG_NAME" ] && [[ "$REPO" == *"/"* ]]; then
    repo_owner=$(echo "$REPO" | cut -d '/' -f1)
    
    # Check if the repository owner might be an organization
    # This is a heuristic - it's not foolproof, but helps alert users
    org_check=$(curl -s -o /dev/null -w "%{http_code}" \
      -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/orgs/$repo_owner")
    
    if [ "$org_check" = "200" ]; then
      echo ""
      echo "⚠️ WARNING: You appear to be working with an organization repository ($repo_owner),"
      echo "   but you haven't specified the organization name with -o/--org."
      echo "   This might cause issues with the Sub-issues API."
      echo ""
      echo "   If you encounter 'Resource not accessible by personal access token' errors,"
      echo "   try running the script again with: -o $repo_owner"
      echo "   or set the GITHUB_ORG_NAME environment variable: export GITHUB_ORG_NAME=$repo_owner"
      echo ""
      
      # Ask if user wants to proceed anyway
      if [ "$SKIP_CONFIRMATION" = false ] && [ "$DRY_RUN" = false ]; then
        read -p "Do you want to proceed anyway? (y/N) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
          echo "Operation cancelled. Please run again with -o $repo_owner"
          exit 1
        fi
      fi
    fi
  fi

  # Create new issues if structure file provided
  create_new_issues

  echo ""
  if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed. No changes were made."
  else
    echo "All issues have been closed."
    if [ ! -z "$ISSUE_STRUCTURE_FILE" ]; then
      echo "New issues have been created from $ISSUE_STRUCTURE_FILE."
      echo ""
      echo "Note: If sub-issues couldn't be linked using the GitHub Sub-issues API,"
      echo "they have been created as regular issues with 'Part of #' references in the description."
      echo "This still allows for tracking, but without the hierarchical UI features."
      echo ""
      echo "If you want to use the official GitHub Sub-issues feature, make sure:"
      echo "1. You're working with an organization repository (not a personal one)"
      echo "2. The Sub-issues feature is enabled for your organization"
      echo "3. Your token has organization-level permissions (org:write scope)"
      echo "4. Use -o/--org or set GITHUB_ORG_NAME environment variable with your organization name"
    else
      echo "To create new issues, run this script with the -s option and a JSON structure file."
    fi
  fi
fi

echo ""
echo "Done!"
