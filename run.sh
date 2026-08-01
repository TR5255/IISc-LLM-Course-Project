#!/usr/bin/env bash
set -e

echo "=== Launching Smart AI Router Application ==="

# Check environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run backend in background
./run_backend.sh &
BACKEND_PID=$!

echo "Backend PID: $BACKEND_PID"
sleep 2

# Open frontend UI
./run_frontend.sh

echo "Press Ctrl+C to stop backend..."
wait $BACKEND_PID
