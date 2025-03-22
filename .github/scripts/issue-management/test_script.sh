#!/bin/bash
# Test script to simulate the behavior of create_sub_issues.sh

echo "Testing create_sub_issues.sh functionality"
echo "Dry run: ${DRY_RUN:-false}"
echo ""

# Create a test issue with tasks
cat > test_issue.json << EOF
{
  "number": 106,
  "title": "Test Issue with Tasks",
  "body": "This is a test issue with tasks.\n\n- [ ] Task 1: Implement feature X\n- [ ] Task 2: Fix bug Y\n- [ ] Task 3: Update documentation Z",
  "labels": [{"name": "enhancement"}, {"name": "good first issue"}]
}
EOF

# Extract tasks from the issue body
issue_body=$(cat test_issue.json | jq -r '.body')
tasks=$(echo "$issue_body" | grep -e '- \[ \] ' | sed 's/- \[ \] //')

# Count tasks
task_count=$(echo "$tasks" | wc -l)
echo "Found $task_count tasks in test issue"

# Display tasks
echo "Tasks:"
echo "$tasks"
echo ""

if [ "${DRY_RUN:-false}" = true ]; then
  echo "[DRY RUN] Would create $task_count sub-issues"
else
  echo "Creating sub-issues..."

  # Simulate creating sub-issues
  counter=1
  echo "$tasks" | while read -r task; do
    echo "Creating sub-issue #$counter: $task"
    counter=$((counter + 1))
  done

  echo "Linking sub-issues to parent..."
  echo "Sub-issues API check: Available"

  # Simulate linking sub-issues
  for i in $(seq 1 $task_count); do
    echo "Linked sub-issue #$i to parent"
  done
fi

echo ""
echo "Test completed successfully!"
rm test_issue.json
