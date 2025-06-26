import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Type
from src.file_types import (
    DataFile,
    NEONReflectanceFile,
    NEONReflectanceENVIFile,
    NEONReflectanceENVHDRFile,
    NEONReflectanceAncillaryENVIFile,
    NEONReflectanceAncillaryENVIFileHeader,
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
    SpectralDataCSVFile,
    MaskedFileMixin,
    SensorType
)


def categorize_file(file_obj: DataFile) -> str:
    """
    Determine the category for a given file based on its type and attributes.
    
    Categories:
    - For sensor files: sensor name (e.g., "Landsat_5_TM")
    - For masked sensor files: sensor name + "_Masked" (e.g., "Landsat_5_TM_Masked")
    - For non-sensor reflectance files: "Reflectance" or "Reflectance_Masked"
    - For ancillary and non-img/hdr files: "Generic"
    """
    
    # Check if it's a resampled file with sensor information
    if isinstance(file_obj, (NEONReflectanceResampledENVIFile, 
                            NEONReflectanceResampledHDRFile,
                            NEONReflectanceResampledMaskENVIFile,
                            NEONReflectanceResampledMaskHDRFile)):
        # Get the sensor name and normalize it
        sensor = file_obj.sensor.replace("_", " ")
        
        # Check if it's a mask file or if the file has MaskedFileMixin and is masked
        is_masked = (isinstance(file_obj, (NEONReflectanceResampledMaskENVIFile, 
                                          NEONReflectanceResampledMaskHDRFile)) or
                    (isinstance(file_obj, MaskedFileMixin) and file_obj.is_masked))
        
        if is_masked:
            return f"{sensor}_Masked"
        else:
            return sensor
    
    # Check if it's a reflectance file (ENVI or BRDF corrected)
    elif isinstance(file_obj, (NEONReflectanceENVIFile,
                              NEONReflectanceENVHDRFile,
                              NEONReflectanceBRDFCorrectedENVIFile,
                              NEONReflectanceBRDFCorrectedENVIHDRFile)):
        # Check if it's masked
        if isinstance(file_obj, MaskedFileMixin) and file_obj.is_masked:
            return "Reflectance_Masked"
        else:
            return "Reflectance"
    
    # BRDF mask files are inherently masks
    elif isinstance(file_obj, (NEONReflectanceBRDFMaskENVIFile, NEONReflectanceBRDFMaskENVIHDRFile)):
        return "Reflectance_Masked"
    
    # All other file types go to Generic
    # This includes: ancillary files, config files, coefficient files, 
    # spectral data CSV files, and base reflectance H5 files
    else:
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
        NEONReflectanceENVHDRFile,
        NEONReflectanceAncillaryENVIFile,
        NEONReflectanceAncillaryENVIFileHeader,
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
        SpectralDataCSVFile
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
                # Get the category for this file
                category = categorize_file(file_obj)
                
                # Normalize category for directory name (replace spaces with underscores)
                category_dir = category.replace(" ", "_")
                
                # Construct source and destination paths
                source_path = file_obj.path
                relative_path = source_path.relative_to(base_path)
                
                # Create destination path maintaining filename
                dest_path = Path(destination_folder) / "sorted_files" / "envi" / category_dir / source_path.name
                
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