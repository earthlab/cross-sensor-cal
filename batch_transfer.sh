#!/bin/bash

# CSV move list
CSV_FILE="home/shared/earthlab/macrosystems/cross-sensor-cal/sorted_files/envi_file_move_list.csv"

# Loop through CSV and run gocmd cp on each row
tail -n +2 "$CSV_FILE" | while IFS=, read -r source_path destination_path; do
    echo "Copying: $source_path -> $destination_path"
    ./gocmd cp --retry 3 -d "$source_path" "$destination_path"
done

echo "✅ All iRODS files copied successfully."

