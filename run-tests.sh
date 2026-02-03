#!/bin/bash
# Run wrapper tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Wrapper Test Suite ===${NC}"

# Create/activate venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
pip install -q -r tests/requirements-test.txt

# Parse arguments
RUN_INTEGRATION=false
VERBOSE=""
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --integration|-i)
            RUN_INTEGRATION=true
            shift
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --unit|-u)
            RUN_INTEGRATION=false
            shift
            ;;
        *)
            SPECIFIC_TEST="$1"
            shift
            ;;
    esac
done

# Run tests
echo ""
if [ -n "$SPECIFIC_TEST" ]; then
    echo -e "${GREEN}Running specific test: $SPECIFIC_TEST${NC}"
    python -m pytest "$SPECIFIC_TEST" $VERBOSE
elif [ "$RUN_INTEGRATION" = true ]; then
    echo -e "${GREEN}Running ALL tests (including integration)${NC}"
    echo -e "${YELLOW}Note: Integration tests require Claude SDK connection${NC}"
    python -m pytest $VERBOSE
else
    echo -e "${GREEN}Running unit tests only (excluding integration)${NC}"
    echo -e "${YELLOW}Use --integration or -i to include integration tests${NC}"
    python -m pytest -m "not integration" $VERBOSE
fi

echo ""
echo -e "${GREEN}=== Tests Complete ===${NC}"
