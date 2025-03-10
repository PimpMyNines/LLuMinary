#!/bin/bash
# Script to run tests in Docker for consistent environment

# Build the test container
echo "Building test container..."
docker build -t lluminary-test -f Dockerfile.test .

# Check if specific tests were requested
if [ $# -eq 0 ]; then
  # No arguments, run all tests
  echo "Running all tests..."
  docker run --rm lluminary-test
else
  # Run specific tests
  echo "Running specified tests: $@"
  docker run --rm lluminary-test "$@"
fi

# Get the exit code
EXIT_CODE=$?

# Print results summary
if [ $EXIT_CODE -eq 0 ]; then
  echo -e "\n\033[0;32mAll tests passed successfully!\033[0m"
else
  echo -e "\n\033[0;31mTests failed with exit code: $EXIT_CODE\033[0m"
fi

exit $EXIT_CODE
