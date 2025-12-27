#!/usr/bin/env bash
set -e  # stop on first failure

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
TEST_ROOT="$PROJECT_ROOT/tests"

echo "Project root: $PROJECT_ROOT"
echo "Test root: $TEST_ROOT"

# Find all test_*.py files
mapfile -t TEST_FILES < <(find "$TEST_ROOT" -type f -name "test_*.py" | sort)

if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "No test files found."
    exit 1
fi

FAILED=0

for test_file in "${TEST_FILES[@]}"; do
    rel_path="${test_file#$PROJECT_ROOT/}"
    echo
    echo "Running $rel_path"
    echo "----------------------------------------"

    (
        cd "$PROJECT_ROOT"
        python -m unittest "$rel_path"
    ) || FAILED=1
done

echo
if [ $FAILED -eq 0 ]; then
    echo "All tests passed ✔"
else
    echo "Some tests failed ✘"
fi

exit $FAILED