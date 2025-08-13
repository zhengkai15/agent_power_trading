#!/bin/bash

# This script runs the full inference and trading simulation pipeline.

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add the project root to PYTHONPATH so that Python can find the 'src' module
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Define the path to the main inference script
MAIN_SCRIPT="$PROJECT_ROOT/src/inference/trader.py"

# Check if the main script exists
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Error: Main inference script not found at $MAIN_SCRIPT"
    exit 1
fi

# Check if processed data exists (optional, but good for robustness)
PROCESSED_DATA_PATH="$PROJECT_ROOT/data/aexp/final_dataset.csv"
if [ ! -f "$PROCESSED_DATA_PATH" ]; then
    echo "Error: Processed data not found at $PROCESSED_DATA_PATH. Please run data processing first."
    exit 1
fi

# Note: Agent and Price Predictor models are checked within the Python script
# as they might not exist if training was skipped.

echo "Starting full trading simulation inference..."
# Execute the Python script directly
python3 "$MAIN_SCRIPT"

if [ $? -eq 0 ]; then
    echo "Full trading simulation inference completed successfully."
else
    echo "Error: Full trading simulation inference failed."
    exit 1
fi
