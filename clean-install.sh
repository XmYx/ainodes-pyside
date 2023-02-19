#!/bin/bash

# Check for Python in the system path
if command -v python3 &>/dev/null; then
    PY_EXECUTABLE=python3
elif command -v python &>/dev/null; then
    PY_EXECUTABLE=python
else
    echo "Error: Python not found in system path"
    exit 1
fi

# Check for venv module in the Python installation
if $PY_EXECUTABLE -c "import venv" &>/dev/null; then
    VENV_CMD="$PY_EXECUTABLE -m venv"
else
    echo "Error: venv module not found in Python installation"
    exit 1
fi

# Create a new virtual environment
$VENV_CMD ai-nodes
source ./ai-nodes/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run the setup script
pip install termcolor
$PY_EXECUTABLE install.py