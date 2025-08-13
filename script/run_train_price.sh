#!/bin/bash

# This script runs the price predictor and RL agent training pipelines.

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add the project root to PYTHONPATH so that Python can find the 'src' module
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Define the path to the price predictor training script
PRICE_PREDICTOR_SCRIPT="$PROJECT_ROOT/src/training/train_price.py"


# --- Run Price Predictor Training ---
echo "Starting price predictor training..."
if [ ! -f "$PRICE_PREDICTOR_SCRIPT" ]; then
    echo "Error: Price predictor training script not found at $PRICE_PREDICTOR_SCRIPT"
    exit 1
fi
python3 "$PRICE_PREDICTOR_SCRIPT"

if [ $? -eq 0 ]; then
    echo "Price predictor training completed successfully."
else
    echo "Error: Price predictor training failed."
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "RL agent training completed successfully."
else
    echo "Error: RL agent training failed."
    exit 1
fi

echo "\nAll training processes completed."
