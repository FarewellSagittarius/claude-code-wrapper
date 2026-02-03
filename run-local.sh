#!/bin/bash
# Run wrapper locally for development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default values
export PORT=${PORT:-8790}
export HOST=${HOST:-0.0.0.0}
export DEBUG_MODE=${DEBUG_MODE:-true}
export CLAUDE_CWD=${CLAUDE_CWD:-$HOME}
export API_KEY_LIGHT=${API_KEY_LIGHT:-sk-light-dev}
export API_KEY_BASIC=${API_KEY_BASIC:-sk-basic-dev}
export API_KEY_HEAVY=${API_KEY_HEAVY:-sk-heavy-dev}
export API_KEY_CUSTOM=${API_KEY_CUSTOM:-sk-custom-dev}

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
pip install -q -r requirements.txt

# Run server with auto-reload
echo "Starting wrapper on http://${HOST}:${PORT}"
echo "Press Ctrl+C to stop"
python -m uvicorn src.main:app --host $HOST --port $PORT --reload
