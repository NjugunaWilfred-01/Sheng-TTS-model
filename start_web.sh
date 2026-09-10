#!/usr/bin/env bash
# One-click launcher for Swahili & Sheng S2S Web UI
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "🚀 Starting Swahili & Sheng S2S Gradio Server..."
echo "🌐 Opening at http://0.0.0.0:7860"
./venv/bin/python app.py
