#!/usr/bin/env bash
set -e

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Clear port 8000 if already in use to prevent [Errno 98] Address already in use
PID_8000=$(lsof -ti:8000 2>/dev/null || true)
if [ -n "$PID_8000" ]; then
    echo "Clearing process $PID_8000 on port 8000..."
    kill -9 $PID_8000 2>/dev/null || true
    sleep 1
fi

echo "=== Starting Smart AI Router FastAPI Backend ==="
PYTHONPATH=. python -m uvicorn ui.backend.app:app --host 0.0.0.0 --port 8000 --reload
