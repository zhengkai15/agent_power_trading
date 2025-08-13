#!/bin/bash

# This script runs the price prediction inference pipeline.
# It uses a pre-trained model from a specified experiment directory.

# --- Configuration ---
# !!! IMPORTANT !!!
# Change this variable to the path of the experiment directory containing your trained model.
# Example: EXP_PATH="aexp/exp_20250813_164108"
EXP_PATH="aexp/exp_20250813_164108"
# --- End of Configuration ---


# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add the project root to PYTHONPATH so that Python can find the 'src' module
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Define the path to the main inference script
MAIN_SCRIPT="$PROJECT_ROOT/src/inference/predictor.py"
FULL_EXP_PATH="$PROJECT_ROOT/$EXP_PATH"

# Check if the main script exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: Main inference script not found at $MAIN_SCRIPT"
    exit 1
fi

# Check if the experiment directory exists
if [ ! -d "$FULL_EXP_PATH" ]; then
    echo "Error: Experiment directory not found at $FULL_EXP_PATH"
    echo "Please check the EXP_PATH variable in this script."
    exit 1
fi

echo "Starting price prediction inference using model from: $EXP_PATH"
# Execute the Python script with the experiment path argument
python3 "$MAIN_SCRIPT" --exp_path "$FULL_EXP_PATH"

if [ $? -eq 0 ]; then
    echo "Price prediction inference completed successfully."
else
    echo "Error: Price prediction inference failed."
    exit 1
fi