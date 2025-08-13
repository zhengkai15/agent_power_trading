#!/bin/bash

# This script runs the full inference and trading simulation pipeline
# using models from specified experiment directories.

# --- Configuration ---
# !!! IMPORTANT !!!
# Change these variables to the paths of your experiment directories.

# Path to the experiment directory for the RL Trading Agent
AGENT_EXP_PATH="aexp/rl_exp_20250813_172925"

# Path to the experiment directory for the Price Predictor model
PRICE_EXP_PATH="aexp/exp_20250813_164108"
# --- End of Configuration ---


# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add the project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Define paths
MAIN_SCRIPT="$PROJECT_ROOT/src/inference/trader.py"
FULL_AGENT_EXP_PATH="$PROJECT_ROOT/$AGENT_EXP_PATH"
FULL_PRICE_EXP_PATH="$PROJECT_ROOT/$PRICE_EXP_PATH"

# --- Validations ---
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: Main inference script not found at $MAIN_SCRIPT"
    exit 1
fi

if [ ! -d "$FULL_AGENT_EXP_PATH" ]; then
    echo "Error: Agent experiment directory not found at $FULL_AGENT_EXP_PATH"
    echo "Please check the AGENT_EXP_PATH variable in this script."
    exit 1
fi

if [ ! -d "$FULL_PRICE_EXP_PATH" ]; then
    echo "Error: Price predictor experiment directory not found at $FULL_PRICE_EXP_PATH"
    echo "Please check the PRICE_EXP_PATH variable in this script."
    exit 1
fi
# --- End of Validations ---

echo "Starting full trading simulation..."
echo "Using Agent from: $AGENT_EXP_PATH"
echo "Using Price Predictor from: $PRICE_EXP_PATH"

# Execute the Python script with experiment path arguments
python3 "$MAIN_SCRIPT" \
    --agent_exp_path "$FULL_AGENT_EXP_PATH" \
    --price_exp_path "$FULL_PRICE_EXP_PATH"

if [ $? -eq 0 ]; then
    echo "Full trading simulation inference completed successfully."
else
    echo "Error: Full trading simulation inference failed."
    exit 1
fi