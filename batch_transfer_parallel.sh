#!/bin/bash

# Path to the CSV move list
CSV_FILE="home/shared/earthlab/macrosystems/cross-sensor-cal/sorted_files/envi_file_move_list.csv"

# Set a safe number of parallel jobs
NUM_JOBS=5  # Start with 5, adjust if necessary

# Run in parallel
tail -n +2 "$CSV_FILE" | parallel -j "$NUM_JOBS" --load 80% --colsep ',' '
    irods_destination="/iplant/{2}"
    echo "Transferring: {1} -> $irods_destination"
    ./gocmd put --diff --icat --retry 3 -d -k "{1}" "$irods_destination"
'

echo "✅ All files transferred successfully."
