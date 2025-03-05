#!/bin/bash

# Path to the CSV move list
CSV_FILE="cross-sensor-cal/sorted_files/file_move_list.csv"

# Read the CSV file and execute gocmd for each line
tail -n +2 "$CSV_FILE" | while IFS=, read -r source_path destination_path; do
    echo "Transferring: $source_path -> $destination_path"
    ./gocmd put --diff --icat --retry 3 -d -k "$source_path" "$destination_path"
done

echo "✅ All files transferred successfully."
