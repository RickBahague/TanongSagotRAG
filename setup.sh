#!/usr/bin/env bash
# One-time setup: creates a virtual env confined to this repo (.venv/)
# and installs dependencies into it. ALL python execution in this repo goes
# through this venv — never the system Python, and never another project's
# venv (AskJ's stays its own). VENV_DIR overrides the location if you need
# the env outside the repo (e.g. network mounts).
set -e
cd "$(dirname "$0")"
VENV="${VENV_DIR:-.venv}"

python3 -m venv "$VENV"
"$VENV"/bin/pip install --upgrade pip -q
"$VENV"/bin/pip install -r requirements.txt -q

echo "Setup complete. Run everything through the venv:"
echo "  .venv/bin/python ingest.py -c <collection>"
echo "  .venv/bin/python ask.py -c <collection> \"your question\""
echo "  .venv/bin/python serve.py"
echo "Or activate it first:  source .venv/bin/activate"
