#!/bin/bash

# Path to the CSV move list
CSV_FILE="home/shared/earthlab/macrosystems/cross-sensor-cal/sorted_files/envi_file_move_list.csv"

# Set a safe number of parallel jobs
NUM_JOBS=5  # Adjust based on system capacity

# Run gocmd cp in parallel for each row
tail -n +2 "$CSV_FILE" | parallel -j "$NUM_JOBS" --load 80% --colsep ',' '
    echo "Copying: {1} -> {2}"
    ./gocmd cp --retry 3 -d "{1}" "{2}"
'

echo "✅ All iRODS files copied successfully."

