#!/bin/bash
# Script to find potential duplicate issues in the repository

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

# Display help if no arguments provided
if [ $# -eq 0 ]; then
    echo "LLuMinary GitHub Issue Duplicate Finder"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --issue <number>    Check for duplicates of a specific issue"
    echo "  -t, --title <title>     Check for duplicates with a similar title"
    echo "  -a, --all               Check all issues for potential duplicates"
    echo "  -s, --sub-issues        Check only recently created sub-issues for duplicates"
    echo "  -d, --days <number>     Only check issues created in the last N days (default: 7)"
    echo "  -p, --threshold <number> Similarity threshold percentage (default: 70)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i 123               Check for duplicates of issue #123"
    echo "  $0 -t \"Fix Docker\"      Check for issues with similar titles"
    echo "  $0 -a                   Check all issues for potential duplicates"
    echo "  $0 -s                   Check only recently created sub-issues"
    echo ""
    exit 0
fi

# Default values
ISSUE_NUMBER=""
TITLE=""
CHECK_ALL=false
CHECK_SUB_ISSUES=false
DAYS=7
THRESHOLD=70

# Function to display help message
show_help() {
    echo "Usage: $0 [options]"
    echo "Find potential duplicate issues in a GitHub repository."
    echo ""
    echo "Options:"
    echo "  -i, --issue <number>      Check for duplicates of a specific issue"
    echo "  -t, --title <title>       Check for duplicates with a similar title"
    echo "  -a, --all                 Check all issues for potential duplicates"
    echo "  -s, --sub-issues          Check only recently created sub-issues for duplicates"
    echo "  -d, --days <number>       Only check issues created in the last N days (default: 7)"
    echo "  -p, --threshold <number>  Similarity threshold percentage (default: 70)"
    echo "  -h, --help                Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -i 123                 Check for duplicates of issue #123"
    echo "  $0 -t \"Fix Docker\"        Check for issues with a similar title"
    echo "  $0 -a                     Check all issues for potential duplicates"
    echo "  $0 -s                     Check recently created sub-issues for duplicates"
    echo "  $0 -s -d 14               Check sub-issues created in the last 14 days"
    echo "  $0 -s -p 80               Check sub-issues with a higher similarity threshold (80%)"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--issue)
            ISSUE_NUMBER="$2"
            shift
            shift
            ;;
        -t|--title)
            TITLE="$2"
            shift
            shift
            ;;
        -a|--all)
            CHECK_ALL=true
            shift
            ;;
        -s|--sub-issues)
            CHECK_SUB_ISSUES=true
            shift
            ;;
        -d|--days)
            DAYS="$2"
            shift
            shift
            ;;
        -p|--threshold)
            THRESHOLD="$2"
            shift
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

# Function to normalize text for comparison
normalize_text() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9 ]//g' | tr -s ' '
}

# Function to calculate similarity between two strings
calculate_similarity() {
    local str1="$(normalize_text "$1")"
    local str2="$(normalize_text "$2")"

    # If either string is empty, return 0
    if [ -z "$str1" ] || [ -z "$str2" ]; then
        echo "0.00"
        return
    fi

    # If strings are identical after normalization, return 100
    if [ "$str1" = "$str2" ]; then
        echo "100.00"
        return
    fi

    # Create arrays of words
    local words1=($str1)
    local words2=($str2)

    # Count unique words in each string
    local unique1=()
    local unique2=()
    local common=()

    # Find common and unique words
    for word in "${words1[@]}"; do
        if [[ " ${words2[*]} " == *" $word "* ]]; then
            if [[ ! " ${common[*]} " == *" $word "* ]]; then
                common+=("$word")
            fi
        else
            if [[ ! " ${unique1[*]} " == *" $word "* ]]; then
                unique1+=("$word")
            fi
        fi
    done

    for word in "${words2[@]}"; do
        if [[ ! " ${words1[*]} " == *" $word "* ]] && [[ ! " ${unique2[*]} " == *" $word "* ]]; then
            unique2+=("$word")
        fi
    done

    # Calculate Jaccard similarity
    local common_count=${#common[@]}
    local unique1_count=${#unique1[@]}
    local unique2_count=${#unique2[@]}
    local total_unique=$((common_count + unique1_count + unique2_count))

    if [ $total_unique -eq 0 ]; then
        echo "0.00"
    else
        local similarity=$(echo "scale=2; ($common_count * 100) / $total_unique" | bc -l)
        printf "%.2f" $similarity
    fi
}

# Function to check for duplicates of a specific issue
check_duplicate() {
    local issue_number="$1"
    echo "Checking for duplicates of issue #$issue_number..."

    # Get the issue details
    issue_details=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues/$issue_number")

    # Check if issue exists
    if [ "$(echo "$issue_details" | jq -r '.message // empty')" = "Not Found" ]; then
        echo "Error: Issue #$issue_number not found"
        exit 1
    fi

    issue_title=$(echo "$issue_details" | jq -r '.title')
    issue_body=$(echo "$issue_details" | jq -r '.body // ""')

    echo "Issue #$issue_number: $issue_title"

    # Get all issues
    all_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues?state=all&per_page=100")

    # Get the number of issues
    issue_count=$(echo "$all_issues" | jq 'length')

    # Check each issue for similarity
    for i in $(seq 0 $(($issue_count - 1))); do
        other_number=$(echo "$all_issues" | jq -r ".[$i].number")

        # Skip the same issue
        if [ "$other_number" -eq "$issue_number" ]; then
            continue
        fi

        other_title=$(echo "$all_issues" | jq -r ".[$i].title")
        other_body=$(echo "$all_issues" | jq -r ".[$i].body // \"\"")

        # Calculate title similarity
        title_similarity=$(calculate_similarity "$issue_title" "$other_title")

        # If title similarity is high, check body similarity
        if (( $(echo "$title_similarity >= $THRESHOLD" | bc -l) )); then
            body_similarity=$(calculate_similarity "$issue_body" "$other_body")

            echo "Potential duplicate found:"
            echo "  Issue #$other_number: $other_title"
            echo "  Title similarity: $title_similarity%"
            echo "  Body similarity: $body_similarity%"
            echo ""
        fi
    done
}

# Function to find all potential duplicates
find_all_duplicates() {
    echo "Checking all issues for potential duplicates..."

    # Get all issues
    all_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues?state=all&per_page=100")

    # Get the number of issues
    issue_count=$(echo "$all_issues" | jq 'length')

    # Compare each issue with others
    for i in $(seq 0 $(($issue_count - 1))); do
        issue_number=$(echo "$all_issues" | jq -r ".[$i].number")
        issue_title=$(echo "$all_issues" | jq -r ".[$i].title")
        issue_body=$(echo "$all_issues" | jq -r ".[$i].body // \"\"")

        echo "Checking issue #$issue_number: $issue_title"

        # Only compare with issues that have higher numbers to avoid duplicate comparisons
        for j in $(seq $(($i + 1)) $(($issue_count - 1))); do
            other_number=$(echo "$all_issues" | jq -r ".[$j].number")
            other_title=$(echo "$all_issues" | jq -r ".[$j].title")
            other_body=$(echo "$all_issues" | jq -r ".[$j].body // \"\"")

            # Calculate title similarity
            title_similarity=$(calculate_similarity "$issue_title" "$other_title")

            # If title similarity is high, check body similarity
            if (( $(echo "$title_similarity >= $THRESHOLD" | bc -l) )); then
                body_similarity=$(calculate_similarity "$issue_body" "$other_body")

                echo "Potential duplicate found:"
                echo "  Issue #$issue_number: $issue_title"
                echo "  Issue #$other_number: $other_title"
                echo "  Title similarity: $title_similarity%"
                echo "  Body similarity: $body_similarity%"
                echo ""
            fi
        done

        echo ""
    done
}

# Function to check recently created sub-issues
check_sub_issues() {
    echo "Checking recently created sub-issues for duplicates..."

    # Get recent issues (created in the last N days)
    since_date=$(date -v-${DAYS}d +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date --date="$DAYS days ago" -u +"%Y-%m-%dT%H:%M:%SZ")

    recent_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues?state=all&per_page=100&since=$since_date")

    # Filter for issues that are likely sub-issues (contain "Part of #" in the body)
    sub_issues_json=$(echo "$recent_issues" | jq -c '[.[] | select(.body | contains("Part of #"))]')

    # Get the number of sub-issues
    sub_issue_count=$(echo "$sub_issues_json" | jq 'length')

    # Check each sub-issue for duplicates
    for i in $(seq 0 $(($sub_issue_count - 1))); do
        sub_number=$(echo "$sub_issues_json" | jq -r ".[$i].number")
        sub_title=$(echo "$sub_issues_json" | jq -r ".[$i].title")
        sub_body=$(echo "$sub_issues_json" | jq -r ".[$i].body // \"\"")

        # Extract parent issue number
        parent_number=$(echo "$sub_body" | grep -o "Part of #[0-9]*" | head -1 | grep -o "[0-9]*")

        echo "Checking sub-issue #$sub_number (parent #$parent_number): $sub_title"

        # Get the number of all issues
        all_issue_count=$(echo "$recent_issues" | jq 'length')

        # Check against all other issues
        for j in $(seq 0 $(($all_issue_count - 1))); do
            other_number=$(echo "$recent_issues" | jq -r ".[$j].number")

            # Skip the same issue
            if [ "$other_number" -eq "$sub_number" ]; then
                continue
            fi

            other_title=$(echo "$recent_issues" | jq -r ".[$j].title")
            other_body=$(echo "$recent_issues" | jq -r ".[$j].body // \"\"")

            # Calculate title similarity
            title_similarity=$(calculate_similarity "$sub_title" "$other_title")

            # If title similarity is high, check body similarity
            if (( $(echo "$title_similarity >= $THRESHOLD" | bc -l) )); then
                body_similarity=$(calculate_similarity "$sub_body" "$other_body")

                echo "Potential duplicate found:"
                echo "  Issue #$other_number: $other_title"
                echo "  Title similarity: $title_similarity%"
                echo "  Body similarity: $body_similarity%"
                echo ""
            fi
        done

        echo ""
    done
}

# Function to check for issues with similar titles
check_title_similarity() {
    local search_title="$1"
    echo "Checking for issues with titles similar to: \"$search_title\""

    # Get all issues
    all_issues=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "https://api.github.com/repos/$REPO/issues?state=all&per_page=100")

    # Get the number of issues
    issue_count=$(echo "$all_issues" | jq 'length')

    # Check each issue for title similarity
    for i in $(seq 0 $(($issue_count - 1))); do
        issue_number=$(echo "$all_issues" | jq -r ".[$i].number")
        issue_title=$(echo "$all_issues" | jq -r ".[$i].title")
        issue_body=$(echo "$all_issues" | jq -r ".[$i].body // \"\"")

        # Calculate title similarity
        title_similarity=$(calculate_similarity "$search_title" "$issue_title")

        # If title similarity is high, report it
        if (( $(echo "$title_similarity >= $THRESHOLD" | bc -l) )); then
            echo "Potential match found:"
            echo "  Issue #$issue_number: $issue_title"
            echo "  Title similarity: $title_similarity%"
            echo ""
        fi
    done
}

# Main execution
if [ ! -z "$ISSUE_NUMBER" ]; then
    check_duplicate "$ISSUE_NUMBER"
elif [ ! -z "$TITLE" ]; then
    check_title_similarity "$TITLE"
elif [ "$CHECK_ALL" = true ]; then
    find_all_duplicates
elif [ "$CHECK_SUB_ISSUES" = true ]; then
    check_sub_issues
else
    echo "Error: No action specified."
    echo "Run '$0 --help' for usage information."
    exit 1
fi

echo "Done!"
