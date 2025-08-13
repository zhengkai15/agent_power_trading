#!/bin/bash

# Define the project root directory
PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")

# Set Python path to include the project root
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Run the main data processing script
echo "Starting data processing..."
python3 "$PROJECT_ROOT/src/data_processing/main.py"

if [ $? -eq 0 ]; then
    echo "Data processing completed successfully."
else
    echo "Data processing failed."
    exit 1
fi
