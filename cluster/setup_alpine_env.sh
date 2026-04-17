#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${1:-$PWD}"
VENV_DIR="${VENV_DIR:-/scratch/alpine/$USER/receipt-parsing/.venv}"

cd "$PROJECT_ROOT"

python3 --version
mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "Environment ready at: $VENV_DIR"
