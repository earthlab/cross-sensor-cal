#!/bin/bash

# Path to the CSV move list
CSV_FILE="cross-sensor-cal/sorted_files/file_move_list.csv"

# Number of parallel jobs
NUM_JOBS=10

# Run in parallel
tail -n +2 "$CSV_FILE" | parallel -j "$NUM_JOBS" --colsep ',' '
    irods_destination="/iplant/home/shared/earthlab/macrosystems/{2}"
    echo "Transferring: {1} -> $irods_destination"
    ./gocmd put --diff --icat --retry 3 -d -k "{1}" "$irods_destination"
'

echo "✅ All files transferred successfully."
