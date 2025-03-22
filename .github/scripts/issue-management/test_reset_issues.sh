#!/bin/bash
# Test script to simulate the reset_issues.sh script without making actual API calls

# Default values
DRY_RUN=false

# Parse command line arguments
while getopts "d" opt; do
  case $opt in
    d)
      DRY_RUN=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

echo "=== LLuMinary GitHub Issue Reset Tool (TEST MODE) ==="
echo "This script simulates the reset_issues.sh script without making actual API calls."
if [ "$DRY_RUN" = true ]; then
  echo "DRY RUN MODE: No changes will be made"
fi
echo ""

# Create a test issue structure file
cat > test_structure.json << EOF
{
  "parent_issues": [
    {
      "title": "Parent Issue 1",
      "body": "This is a parent issue with sub-issues",
      "labels": ["enhancement", "documentation"],
      "sub_issues": [
        {
          "title": "Sub-Issue 1.1",
          "body": "This is a sub-issue of Parent Issue 1",
          "labels": ["bug", "good first issue"]
        },
        {
          "title": "Sub-Issue 1.2",
          "body": "This is another sub-issue of Parent Issue 1",
          "labels": ["enhancement"]
        }
      ]
    },
    {
      "title": "Parent Issue 2",
      "body": "This is another parent issue with sub-issues",
      "labels": ["bug"],
      "sub_issues": [
        {
          "title": "Sub-Issue 2.1",
          "body": "This is a sub-issue of Parent Issue 2",
          "labels": ["enhancement"]
        }
      ]
    }
  ]
}
EOF

echo "Created test issue structure file: test_structure.json"
echo ""

# Simulate backing up issues
echo "Simulating backup of existing issues..."
echo "Backed up 127 issues to issues_backup_test.json"
echo ""

# Simulate closing issues
echo "Simulating closing all issues..."
echo "Found 72 open issues"
if [ "$DRY_RUN" = true ]; then
  echo "[DRY RUN] Would close 72 issues"
else
  echo "[TEST MODE] Would close 72 issues"
fi
echo ""

# Simulate creating new issues
echo "Simulating creating new issues from test_structure.json..."
echo "Found 2 parent issues to create"
echo ""

# Only proceed with creating issues if not in dry run mode
if [ "$DRY_RUN" = true ]; then
  echo "[DRY RUN] Would create 2 parent issues and 3 sub-issues"
  echo "[DRY RUN] Would link 3 sub-issues to their parent issues"
  echo ""
  echo "Dry run completed successfully!"
else
  # Simulate creating parent issues and sub-issues
  parent_numbers=(101 102)
  sub_numbers=(201 202 203)

  for i in {0..1}; do
    parent_number=${parent_numbers[$i]}
    echo "Creating parent issue #$parent_number: Parent Issue $((i+1))"

    if [ $i -eq 0 ]; then
      sub_count=2
      sub_start=0
    else
      sub_count=1
      sub_start=2
    fi

    echo "Found $sub_count sub-issues to create for parent #$parent_number"

    # Create a temporary file to store sub-issue numbers
    sub_numbers_file=$(mktemp)

    # Process each sub-issue
    for j in $(seq 0 $(($sub_count-1))); do
      sub_index=$(($sub_start + $j))
      sub_number=${sub_numbers[$sub_index]}
      echo "Creating sub-issue: Sub-Issue $((i+1)).$((j+1))"
      echo "Created sub-issue #$sub_number"

      # Add to temporary file with parent number for reference
      echo "$sub_number:$parent_number" >> "$sub_numbers_file"
    done

    # Simulate linking sub-issues
    echo "Attempting to link sub-issues to parent #$parent_number using the Sub-issues API..."
    echo "Sub-issues API is available. Linking sub-issues..."

    # Link each sub-issue to the parent
    while IFS=: read -r sub_number parent_num; do
      echo "Linking sub-issue #$sub_number to parent #$parent_num"
      echo "Successfully linked sub-issue #$sub_number to parent #$parent_num"
    done < "$sub_numbers_file"

    # Clean up temporary file
    rm -f "$sub_numbers_file"

    echo ""
  done

  echo "Created 2 parent issues and their sub-issues"
  echo ""
  echo "Test completed successfully!"
fi

echo ""
echo "Cleaning up test files..."
rm -f test_structure.json

echo "Done!"
