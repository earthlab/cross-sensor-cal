"""
Get files from remote matching certain keywords and put them in the instance in a folder of your choice.
Download flightline files from CyVerse/iRODS and organize by location (gordon/goldhill).

This script:
1. Lists all files directly in AGU_prep_NIWO
2. Downloads only files with both "georef" and "brightened" in their names
3. Organizes them into separate folders based on "gordon" or "goldhill" in filename
4. Can be run in a notebook cell
"""

from pathlib import Path
import subprocess

# ============================================================================
# CONFIGURATION
# ============================================================================

# Remote base directory on CyVerse/iRODS (files are directly in this directory)
REMOTE_BASE = "/iplant/home/shared/earthlab/macrosystems/cross-sensor-cal/AGU_prep_NIWO"

# Local base directory (where to create folders)
LOCAL_BASE = Path("/home/jovyan/data-store/cross-sensor-cal")

# Folder names for organizing files
# UPDATE THESE with your desired folder names
GORDON_FOLDER_NAME = "NIWO_gordon_1"  # UPDATE THIS - folder name for gordon files
GOLDHILL_FOLDER_NAME = "NIWO_goldhill_1"  # UPDATE THIS - folder name for goldhill files

# gocmd executable
GOCMD = "./gocmd"

# File filter: only download files containing BOTH of these keywords
REQUIRED_KEYWORDS = ["georef", "brightened"]

# Location keywords (files containing these will be organized accordingly)
GORDON_KEYWORD = "gordon"
GOLDHILL_KEYWORD = "goldhill"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def list_remote_files(remote_dir: str):
    """
    List all files in a remote directory.
    
    Parameters
    ----------
    remote_dir : str
        Remote directory path
        
    Returns
    -------
    list
        List of filenames
    """
    try:
        list_cmd = [GOCMD, "ls", f"i:{remote_dir}"]
        list_result = subprocess.run(list_cmd, capture_output=True, text=True, check=False)
        
        if list_result.returncode != 0:
            print(f"   ❌ Could not list remote directory: {list_result.stderr}")
            return []
        
        # Parse file list
        lines = [line.strip() for line in list_result.stdout.split('\n') if line.strip()]
        
        # Extract filenames from paths
        remote_files = []
        for line in lines:
            # Skip directory listings or other non-file lines
            if '  ' in line and ('/' in line or '.' in line):
                parts = line.split()
                if parts:
                    filename = parts[-1]
                    if filename and '.' in filename:  # Has extension, likely a file
                        remote_files.append(filename)
            elif '.' in line and '/' not in line:
                remote_files.append(line)
            elif '/' in line:
                filename = Path(line).name
                if filename:
                    remote_files.append(filename)
        
        return remote_files
            
    except FileNotFoundError as e:
        print(f"   ❌ ERROR: gocmd not found: {e}")
        return []
    except Exception as e:
        print(f"   ❌ ERROR listing files: {e}")
        import traceback
        traceback.print_exc()
        return []

def download_file(remote_file_path: str, local_dest: Path, filename: str):
    """
    Download a single file from remote to local destination.
    
    Parameters
    ----------
    remote_file_path : str
        Full remote file path
    local_dest : Path
        Local destination directory
    filename : str
        Filename for reporting
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    local_file = local_dest / Path(remote_file_path).name
    
    # Skip if already exists
    if local_file.exists():
        return "skipped"
    
    try:
        download_cmd = [
            GOCMD,
            "get",
            "--progress",
            "-K",
            "--icat",
            remote_file_path,
            str(local_dest)
        ]
        
        result = subprocess.run(download_cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            return "success"
        else:
            print(f"      ❌ Failed: {filename} - {result.stderr[:100]}")
            return "failed"
            
    except Exception as e:
        print(f"      ❌ Error downloading {filename}: {e}")
        return "failed"

def determine_location(filename: str):
    """
    Determine if a file belongs to gordon or goldhill based on filename.
    
    Parameters
    ----------
    filename : str
        Filename to check
        
    Returns
    -------
    str or None
        'gordon', 'goldhill', or None if cannot determine
    """
    filename_lower = filename.lower()
    
    if GORDON_KEYWORD.lower() in filename_lower:
        return 'gordon'
    elif GOLDHILL_KEYWORD.lower() in filename_lower:
        return 'goldhill'
    else:
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DOWNLOAD AND ORGANIZE FILES BY LOCATION (GORDON/GOLDHILL)")
    print("=" * 80)
    
    print(f"\n📂 Configuration:")
    print(f"   Remote directory: {REMOTE_BASE}")
    print(f"   Local base: {LOCAL_BASE}")
    print(f"   Gordon folder: {GORDON_FOLDER_NAME}")
    print(f"   Goldhill folder: {GOLDHILL_FOLDER_NAME}")
    print(f"   File filter: Only files containing BOTH '{' AND '.join(REQUIRED_KEYWORDS)}'")
    print(f"   Location keywords: '{GORDON_KEYWORD}' → gordon, '{GOLDHILL_KEYWORD}' → goldhill")
    
    # Create location folders
    gordon_folder = LOCAL_BASE / GORDON_FOLDER_NAME
    goldhill_folder = LOCAL_BASE / GOLDHILL_FOLDER_NAME
    
    gordon_folder.mkdir(parents=True, exist_ok=True)
    goldhill_folder.mkdir(parents=True, exist_ok=True)
    
    # List all files in remote directory
    print(f"\n{'='*80}")
    print("LISTING FILES IN REMOTE DIRECTORY")
    print(f"{'='*80}")
    print(f"\n   📥 Listing files in: {REMOTE_BASE}")
    
    all_files = list_remote_files(REMOTE_BASE)
    
    if not all_files:
        print(f"   ❌ No files found in remote directory")
        exit(1)
    
    print(f"   ✅ Found {len(all_files)} total files")
    
    # Filter files: must contain BOTH "georef" AND "brightened"
    print(f"\n   🔍 Filtering files (must contain both '{' AND '.join(REQUIRED_KEYWORDS)}')...")
    
    filtered_files = []
    for filename in all_files:
        filename_lower = filename.lower()
        if all(keyword.lower() in filename_lower for keyword in REQUIRED_KEYWORDS):
            filtered_files.append(filename)
    
    print(f"   ✅ {len(filtered_files)} files match filter criteria")
    print(f"   📋 Filtered out: {len(all_files) - len(filtered_files)} files")
    
    if not filtered_files:
        print(f"\n   ❌ No files match the filter criteria!")
        exit(1)
    
    # Organize files by location (gordon/goldhill)
    print(f"\n   🗂️  Organizing files by location...")
    
    gordon_files = []
    goldhill_files = []
    unassigned_files = []
    
    for filename in filtered_files:
        location = determine_location(filename)
        if location == 'gordon':
            gordon_files.append(filename)
        elif location == 'goldhill':
            goldhill_files.append(filename)
        else:
            unassigned_files.append(filename)
    
    print(f"   📊 Gordon files: {len(gordon_files)}")
    print(f"   📊 Goldhill files: {len(goldhill_files)}")
    if unassigned_files:
        print(f"   ⚠️  Unassigned files (no '{GORDON_KEYWORD}' or '{GOLDHILL_KEYWORD}' in name): {len(unassigned_files)}")
        for filename in unassigned_files[:5]:
            print(f"      - {filename}")
        if len(unassigned_files) > 5:
            print(f"      ... and {len(unassigned_files) - 5} more")
    
    # Download gordon files
    print(f"\n{'='*80}")
    print(f"DOWNLOADING GORDON FILES ({len(gordon_files)} files)")
    print(f"{'='*80}")
    
    gordon_results = {"downloaded": 0, "skipped": 0, "failed": 0}
    
    for i, filename in enumerate(gordon_files, 1):
        remote_file_path = f"i:{REMOTE_BASE}/{filename}"
        result = download_file(remote_file_path, gordon_folder, filename)
        
        if result == "success":
            gordon_results["downloaded"] += 1
        elif result == "skipped":
            gordon_results["skipped"] += 1
        else:
            gordon_results["failed"] += 1
        
        if i % 10 == 0:
            print(f"      📥 Progress: {i}/{len(gordon_files)} files processed...")
    
    print(f"\n   📊 Gordon download summary:")
    print(f"      ✅ Downloaded: {gordon_results['downloaded']}")
    print(f"      ⏭️  Skipped (already exists): {gordon_results['skipped']}")
    print(f"      ❌ Failed: {gordon_results['failed']}")
    
    # Download goldhill files
    print(f"\n{'='*80}")
    print(f"DOWNLOADING GOLDHILL FILES ({len(goldhill_files)} files)")
    print(f"{'='*80}")
    
    goldhill_results = {"downloaded": 0, "skipped": 0, "failed": 0}
    
    for i, filename in enumerate(goldhill_files, 1):
        remote_file_path = f"i:{REMOTE_BASE}/{filename}"
        result = download_file(remote_file_path, goldhill_folder, filename)
        
        if result == "success":
            goldhill_results["downloaded"] += 1
        elif result == "skipped":
            goldhill_results["skipped"] += 1
        else:
            goldhill_results["failed"] += 1
        
        if i % 10 == 0:
            print(f"      📥 Progress: {i}/{len(goldhill_files)} files processed...")
    
    print(f"\n   📊 Goldhill download summary:")
    print(f"      ✅ Downloaded: {goldhill_results['downloaded']}")
    print(f"      ⏭️  Skipped (already exists): {goldhill_results['skipped']}")
    print(f"      ❌ Failed: {goldhill_results['failed']}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"📁 GORDON ({GORDON_FOLDER_NAME}):")
    print(f"   Total files: {len(gordon_files)}")
    print(f"   ✅ Downloaded: {gordon_results['downloaded']}")
    print(f"   ⏭️  Skipped: {gordon_results['skipped']}")
    print(f"   ❌ Failed: {gordon_results['failed']}")
    print(f"   📂 Location: {gordon_folder}")
    
    print(f"\n📁 GOLDHILL ({GOLDHILL_FOLDER_NAME}):")
    print(f"   Total files: {len(goldhill_files)}")
    print(f"   ✅ Downloaded: {goldhill_results['downloaded']}")
    print(f"   ⏭️  Skipped: {goldhill_results['skipped']}")
    print(f"   ❌ Failed: {goldhill_results['failed']}")
    print(f"   📂 Location: {goldhill_folder}")
    
    if unassigned_files:
        print(f"\n⚠️  UNASSIGNED FILES ({len(unassigned_files)}):")
        print(f"   Files that matched filter but couldn't be assigned to gordon/goldhill")
        print(f"   (missing '{GORDON_KEYWORD}' or '{GOLDHILL_KEYWORD}' in filename)")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE")
    print(f"{'='*80}")
