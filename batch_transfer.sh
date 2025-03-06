#!/bin/bash

# Path to the CSV move list
CSV_FILE="cross-sensor-cal/sorted_files/file_move_list.csv"

# Read the CSV file and execute gocmd for each line
tail -n +2 "$CSV_FILE" | while IFS=, read -r source_path destination_path; do
    # Construct the absolute iRODS destination path
    irods_destination="/iplant/home/shared/earthlab/macrosystems/$destination_path"
    echo "Transferring: $source_path -> $irods_destination"
    ./gocmd put --diff --icat --retry 3 -d -k "$source_path" "$irods_destination"
done

echo "✅ All files transferred successfully."
