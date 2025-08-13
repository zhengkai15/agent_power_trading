#!/bin/bash

# This script runs the RL agent training pipeline.

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add the project root to PYTHONPATH so that Python can find the 'src' module
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Define the path to the RL agent training script
RL_AGENT_SCRIPT="$PROJECT_ROOT/src/training/train_strategy.py"

# --- Run RL Agent Training ---
echo "Starting RL agent training..."
if [ ! -f "$RL_AGENT_SCRIPT" ]; then
    echo "Error: RL agent training script not found at $RL_AGENT_SCRIPT"
    exit 1
fi
python3 "$RL_AGENT_SCRIPT"

if [ $? -eq 0 ]; then
    echo "RL agent training completed successfully."
else
    echo "Error: RL agent training failed."
    exit 1
fi

echo "\nAll RL training processes completed."
