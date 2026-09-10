#!/usr/bin/env bash
# Runs automated end-to-end verification across 6 demo scenarios
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "🧪 Running Automated 6-Track Saturday Demo Verification..."
./venv/bin/python cli.py --demo
