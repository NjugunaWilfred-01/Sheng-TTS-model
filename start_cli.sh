#!/usr/bin/env bash
# One-click launcher for Terminal Voice REPL
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "🎤 Launching Terminal Sheng S2S Interactive Voice Chat..."
./venv/bin/python cli.py
