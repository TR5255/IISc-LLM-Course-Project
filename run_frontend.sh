#!/usr/bin/env bash
set -e

echo "=== Opening Smart AI Router Web Dashboard ==="
UI_PATH="$(pwd)/ui/frontend/index.html"

if command -v xdg-open > /dev/null; then
    xdg-open "$UI_PATH"
elif command -v open > /dev/null; then
    open "$UI_PATH"
else
    echo "Please open in your browser: file://$UI_PATH"
fi
