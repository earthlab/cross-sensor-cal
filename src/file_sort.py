import os
import re
from pathlib import Path
from typing import List, Optional, Type

import pandas as pd

from src.file_types import (
    DataFile,
    NEONReflectanceFile,
    NEONReflectanceENVIFile,
    NEONReflectanceENVIHDRFile,
    NEONReflectanceAncillaryENVIFile,
    NEONReflectanceAncillaryENVIHDRFile,
    NEONReflectanceConfigFile,
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceBRDFCorrectedENVIHDRFile,
    NEONReflectanceBRDFMaskENVIFile,
    NEONReflectanceBRDFMaskENVIHDRFile,
    NEONReflectanceCoefficientsFile,
    NEONReflectanceResampledENVIFile,
    NEONReflectanceResampledHDRFile,
    NEONReflectanceResampledMaskENVIFile,
    NEONReflectanceResampledMaskHDRFile,
    SpectralDataParquetFile,
    MaskedSpectralCSVFile,
    EndmembersCSVFile,
    UnmixingModelBestTIF,
    UnmixingModelFractionsTIF,
    UnmixingModelRMSETIF,
    SensorType,
)


RESAMPLED_RE = re.compile(
    r"_resampled(?:_mask)?_(?P<sensor>[A-Za-z0-9_+\-]+)_envi\.(?:img|hdr)$",
    re.IGNORECASE,
)


def _extract_sensor_from_name(path_str: str) -> Optional[str]:
    match = RESAMPLED_RE.search(path_str)
    return match.group("sensor") if match else None


def _normalize_sensor_label(sensor: str) -> str:
    return sensor.replace("_", " ")


def _is_masked(file_obj: DataFile, name_lower: str) -> bool:
    if hasattr(file_obj, "is_masked") and getattr(file_obj, "is_masked") is True:
        return True

    mask_classes = (
        NEONReflectanceBRDFMaskENVIFile,
        NEONReflectanceBRDFMaskENVIHDRFile,
        NEONReflectanceResampledMaskENVIFile,
        NEONReflectanceResampledMaskHDRFile,
    )
    try:
        if isinstance(file_obj, mask_classes):
            return True
    except Exception:
        pass

    return (
        "_resampled_mask_" in name_lower
        or name_lower.endswith("_mask_envi.img")
        or name_lower.endswith("_mask_envi.hdr")
    )


def _category_to_folder(category: str) -> str:
    if category in {"Reflectance", "Reflectance_Masked"}:
        return category
    return category.replace(" ", "_")


def categorize_file(file_obj: DataFile) -> str:
    """Return the destination category name for ``file_obj``."""

    path = getattr(file_obj, "path", None)
    path_str = str(path) if path else ""
    lower_name = path_str.lower()

    sensor_token = _extract_sensor_from_name(path_str)
    if sensor_token:
        label = _normalize_sensor_label(sensor_token)
        return f"{label}_Masked" if _is_masked(file_obj, lower_name) else label

    if isinstance(
        file_obj,
        (NEONReflectanceBRDFCorrectedENVIFile, NEONReflectanceBRDFCorrectedENVIHDRFile),
    ):
        return "Corrected"

    if isinstance(
        file_obj,
        (
            NEONReflectanceENVIFile,
            NEONReflectanceENVIHDRFile,
            NEONReflectanceBRDFMaskENVIFile,
            NEONReflectanceBRDFMaskENVIHDRFile,
        ),
    ):
        return "Reflectance_Masked" if _is_masked(file_obj, lower_name) else "Reflectance"

    if isinstance(
        file_obj,
        (
            NEONReflectanceFile,
            NEONReflectanceAncillaryENVIFile,
            NEONReflectanceAncillaryENVIHDRFile,
        ),
    ):
        return "Generic"

    if lower_name.endswith("_reflectance.h5") or "_ancillary_envi" in lower_name:
        return "Generic"

    return "Generic"


def generate_file_move_list(base_folder: str, destination_folder: str, remote_path_prefix: str = "") -> pd.DataFrame:
    """
    Scans base_folder recursively to find all files matching the defined file types
    and generates a move list with local and iRODS paths.
    
    Parameters:
    - base_folder: The path to the base directory to scan
    - destination_folder: The path to the destination folder for sorted files
    - remote_path_prefix: Optional custom path to add after i:/iplant/ for remote paths
    
    Returns:
    - DataFrame with columns ['Source Path', 'Destination Path'] where Source Path is local
    and Destination Path is the iRODS remote path
    """
    
    # Define all file type classes to check
    file_type_classes: List[Type[DataFile]] = [
        NEONReflectanceFile,
        NEONReflectanceENVIFile,
        NEONReflectanceENVIHDRFile,
        NEONReflectanceAncillaryENVIFile,
        NEONReflectanceAncillaryENVIHDRFile,
        NEONReflectanceConfigFile,
        NEONReflectanceBRDFCorrectedENVIFile,
        NEONReflectanceBRDFCorrectedENVIHDRFile,
        NEONReflectanceBRDFMaskENVIFile,
        NEONReflectanceBRDFMaskENVIHDRFile,
        NEONReflectanceCoefficientsFile,
        NEONReflectanceResampledENVIFile,
        NEONReflectanceResampledHDRFile,
        NEONReflectanceResampledMaskENVIFile,
        NEONReflectanceResampledMaskHDRFile,
        SpectralDataParquetFile,
        MaskedSpectralCSVFile,
        EndmembersCSVFile,
        UnmixingModelBestTIF,
        UnmixingModelFractionsTIF,
        UnmixingModelRMSETIF
    ]
    
    move_list = []
    base_path = Path(base_folder)
    
    # Find all files for each file type class
    for file_class in file_type_classes:
        try:
            # Classes that require suffix parameter
            suffix_classes = [
                NEONReflectanceConfigFile,
                NEONReflectanceBRDFCorrectedENVIFile,
                NEONReflectanceBRDFCorrectedENVIHDRFile,
                NEONReflectanceBRDFMaskENVIFile,
                NEONReflectanceBRDFMaskENVIHDRFile,
                NEONReflectanceCoefficientsFile
            ]
            
            # Resampled classes that have sensor information and need special handling
            resampled_classes = [
                NEONReflectanceResampledENVIFile,
                NEONReflectanceResampledHDRFile,
                NEONReflectanceResampledMaskENVIFile,
                NEONReflectanceResampledMaskHDRFile
            ]
            
            if file_class in suffix_classes:
                # These classes need suffix='envi' parameter
                found_files = file_class.find_in_directory(base_path, suffix='envi')
            elif file_class in resampled_classes:
                # Use find_all_sensors_in_directory for resampled files
                if file_class == NEONReflectanceResampledENVIFile:
                    found_files = file_class.find_all_sensors_in_directory(base_path, suffix='envi')
                else:
                    # For other resampled classes, we need to iterate through all sensors
                    found_files = []
                    for sensor in SensorType:
                        sensor_name = sensor.value.replace(' ', '_')
                        try:
                            sensor_files = file_class.find_in_directory(base_path, sensor=sensor_name, suffix='envi')
                            found_files.extend(sensor_files)
                        except Exception:
                            # Skip if no files found for this sensor
                            continue
            else:
                # All other classes use the default find_in_directory without parameters
                found_files = file_class.find_in_directory(base_path)
            
            for file_obj in found_files:
                category = categorize_file(file_obj)
                category_dir = _category_to_folder(category)

                source_path = file_obj.path

                dest_dir = Path(destination_folder) / "sorted_files" / "envi" / category_dir

                # Create destination path maintaining filename
                dest_path = dest_dir / source_path.name
                
                # Source path is local (no iRODS prefix)
                local_source = str(source_path)
                
                # Destination path is iRODS with optional custom prefix
                if remote_path_prefix:
                    # Ensure no double slashes
                    remote_prefix = remote_path_prefix.strip('/')
                    irods_dest = f"i:/iplant/{remote_prefix}/{dest_path}"
                else:
                    irods_dest = f"i:/iplant/{dest_path}"
                
                move_list.append((local_source, irods_dest))
                
        except Exception as e:
            print(f"Error processing {file_class.__name__}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(move_list, columns=["Source Path", "Destination Path"])
    
    # Save the move list
    os.makedirs(os.path.join(destination_folder, "sorted_files"), exist_ok=True)
    csv_path = os.path.join(destination_folder, "sorted_files", "envi_file_move_list.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Move list saved to {csv_path}")
    print(f"📊 Total files to move: {len(df)}")
    
    # Print category summary
    if len(df) > 0:
        print("\n📁 Files by category:")
        category_counts = df['Destination Path'].apply(
            lambda x: x.split('/envi/')[1].split('/')[0]
        ).value_counts()
        for category, count in category_counts.items():
            print(f"  - {category}: {count} files")
    
    return df


# Example usage
if __name__ == "__main__":
    # Adjust these paths to match your actual directory structure
    base_folder = "niwo_neon"
    destination_folder = "./"
    
    # Optional: Add a custom remote path prefix
    # remote_prefix = "projects/neon/data"
    # df_move_list = generate_file_move_list(base_folder, destination_folder, remote_prefix)
    
    # Generate the file move list (without custom prefix)
    df_move_list = generate_file_move_list(base_folder, destination_folder)