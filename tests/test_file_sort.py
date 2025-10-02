import os
import re
import shutil

# Add parent directory to path to import modules
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from cross_sensor_cal.file_sort import (  # noqa: E402
    categorize_file,
    generate_file_move_list,
)
from cross_sensor_cal.file_types import (  # noqa: E402
    NEONReflectanceAncillaryENVIFile,
    NEONReflectanceBRDFCorrectedENVIFile,
    NEONReflectanceBRDFCorrectedENVIHDRFile,
    NEONReflectanceBRDFMaskENVIFile,
    NEONReflectanceBRDFMaskENVIHDRFile,
    NEONReflectanceCoefficientsFile,
    NEONReflectanceConfigFile,
    NEONReflectanceENVHDRFile,
    NEONReflectanceENVIFile,
    NEONReflectanceFile,
    NEONReflectanceResampledENVIFile,
    NEONReflectanceResampledHDRFile,
    NEONReflectanceResampledMaskENVIFile,
    NEONReflectanceResampledMaskHDRFile,
    SensorType,
    SpectralDataCSVFile,
)


class TestCategorizeFile(unittest.TestCase):
    """Test the categorize_file function with different file types."""

    def test_resampled_sensor_files(self):
        """Test categorization of resampled sensor files."""
        # Test unmasked resampled ENVI file
        file_obj = NEONReflectanceResampledENVIFile(
            path=Path(
                "test/NEON_D13_NIWO_DP1_20200801_161441_resampled_Landsat_5_TM_envi.img"
            ),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
            sensor="Landsat_5_TM",
            suffix="envi",
        )
        self.assertEqual(categorize_file(file_obj), "Landsat 5 TM")

        # Test with different sensors - using actual enum values
        sensors_and_expected = [
            ("Landsat_7_ETM+", "Landsat 7 ETM+"),
            ("Landsat_8_OLI", "Landsat 8 OLI"),
            ("Landsat_9_OLI-2", "Landsat 9 OLI-2"),
            ("MicaSense", "MicaSense"),
            ("MicaSense-to-match_TM_and_ETM+", "MicaSense-to-match TM and ETM+"),
            ("MicaSense-to-match_OLI_and_OLI-2", "MicaSense-to-match OLI and OLI-2"),
        ]

        for sensor, expected in sensors_and_expected:
            file_obj = NEONReflectanceResampledENVIFile(
                path=Path(
                    f"test/NEON_D13_NIWO_DP1_20200801_161441_resampled_{sensor}_envi.img"
                ),
                domain="D13",
                site="NIWO",
                date="20200801",
                time="161441",
                sensor=sensor,
                suffix="envi",
            )
            self.assertEqual(categorize_file(file_obj), expected)

    def test_resampled_masked_sensor_files(self):
        """Test categorization of resampled masked sensor files."""
        # Test mask ENVI file
        file_obj = NEONReflectanceResampledMaskENVIFile(
            path=Path(
                "test/NEON_D13_NIWO_DP1_20200801_161441_resampled_mask_Landsat_5_TM_envi.img"
            ),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
            sensor="Landsat_5_TM",
            suffix="envi",
        )
        self.assertEqual(categorize_file(file_obj), "Landsat 5 TM_Masked")

        # Test mask HDR file
        file_obj = NEONReflectanceResampledMaskHDRFile(
            path=Path(
                "test/NEON_D13_NIWO_DP1_20200801_161441_resampled_mask_MicaSense_envi.hdr"
            ),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
            sensor="MicaSense",
            suffix="envi",
        )
        self.assertEqual(categorize_file(file_obj), "MicaSense_Masked")

    def test_reflectance_files(self):
        """Test categorization of reflectance files."""
        # Test unmasked reflectance ENVI file
        file_obj = NEONReflectanceENVIFile(
            path=Path("test/NEON_D13_NIWO_DP1_20200801_161441_reflectance_envi.img"),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
        )
        self.assertEqual(categorize_file(file_obj), "Reflectance")

        # Test reflectance HDR file
        file_obj = NEONReflectanceENVHDRFile(
            path=Path("test/NEON_D13_NIWO_DP1_20200801_161441_reflectance_envi.hdr"),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
        )
        self.assertEqual(categorize_file(file_obj), "Reflectance")

        # Test BRDF corrected file
        file_obj = NEONReflectanceBRDFCorrectedENVIFile(
            path=Path("test/NEON_D13_NIWO_DP1_20200801_161441_brdf_corrected_envi.img"),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
            suffix="envi",
        )
        self.assertEqual(categorize_file(file_obj), "Reflectance")

        # Test BRDF corrected HDR file
        file_obj = NEONReflectanceBRDFCorrectedENVIHDRFile(
            path=Path("test/NEON_D13_NIWO_DP1_20200801_161441_brdf_corrected_envi.hdr"),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
            suffix="envi",
        )
        self.assertEqual(categorize_file(file_obj), "Reflectance")

        # Test BRDF mask file (inherently masked)
        file_obj = NEONReflectanceBRDFMaskENVIFile(
            path=Path(
                "test/NEON_D13_NIWO_DP1_20200801_161441_brdf_corrected_mask_envi.img"
            ),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
            suffix="envi",
        )
        self.assertEqual(categorize_file(file_obj), "Reflectance_Masked")

        # Test BRDF mask HDR file (inherently masked)
        file_obj = NEONReflectanceBRDFMaskENVIHDRFile(
            path=Path(
                "test/NEON_D13_NIWO_DP1_20200801_161441_brdf_corrected_mask_envi.hdr"
            ),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
            suffix="envi",
        )
        self.assertEqual(categorize_file(file_obj), "Reflectance_Masked")

    def test_masked_reflectance_files(self):
        """Test categorization of masked reflectance files."""
        # Create a mock masked file
        mock_file = Mock(spec=NEONReflectanceENVIFile)
        mock_file.is_masked = True
        mock_file.__class__ = NEONReflectanceENVIFile

        # Mock the isinstance checks
        with patch("cross_sensor_cal.file_sort.isinstance") as mock_isinstance:

            def isinstance_side_effect(obj, class_or_tuple):
                if class_or_tuple == (
                    NEONReflectanceResampledENVIFile,
                    NEONReflectanceResampledHDRFile,
                    NEONReflectanceResampledMaskENVIFile,
                    NEONReflectanceResampledMaskHDRFile,
                ):
                    return False
                elif (
                    class_or_tuple
                    == (
                        NEONReflectanceENVIFile,
                        NEONReflectanceENVHDRFile,
                        NEONReflectanceBRDFCorrectedENVIFile,
                        NEONReflectanceBRDFCorrectedENVIHDRFile,
                    )
                    or hasattr(class_or_tuple, "__name__")
                    and class_or_tuple.__name__ == "MaskedFileMixin"
                ):
                    return True
                elif class_or_tuple == (
                    NEONReflectanceBRDFMaskENVIFile,
                    NEONReflectanceBRDFMaskENVIHDRFile,
                ):
                    return False
                return False

            mock_isinstance.side_effect = isinstance_side_effect
            result = categorize_file(mock_file)
            self.assertEqual(result, "Reflectance_Masked")

    def test_generic_files(self):
        """Test categorization of files that should go to Generic category."""
        # Test H5 file
        file_obj = NEONReflectanceFile(
            path=Path("test/NEON_D13_NIWO_DP1_20200801_161441_reflectance.h5"),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
        )
        self.assertEqual(categorize_file(file_obj), "Generic")

        # Test ancillary ENVI file
        file_obj = NEONReflectanceAncillaryENVIFile(
            path=Path(
                "test/NEON_D13_NIWO_DP1_20200801_161441_reflectance_ancillary_envi.img"
            ),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
        )
        self.assertEqual(categorize_file(file_obj), "Generic")

        # Test config file
        file_obj = NEONReflectanceConfigFile(
            path=Path("test/NEON_D13_NIWO_DP1_20200801_161441_config_envi.json"),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
            suffix="envi",
        )
        self.assertEqual(categorize_file(file_obj), "Generic")

        # Test coefficients file
        file_obj = NEONReflectanceCoefficientsFile(
            path=Path(
                "test/NEON_D13_NIWO_DP1_20200801_161441_reflectance_brdf_coeffs_envi.json"
            ),
            domain="D13",
            site="NIWO",
            date="20200801",
            time="161441",
            correction="brdf",
            suffix="envi",
        )
        self.assertEqual(categorize_file(file_obj), "Generic")

        # Test spectral data CSV
        file_obj = SpectralDataCSVFile(
            path=Path("test/NEON_D13_NIWO_DP1_20200801_161441_spectral_data.csv"),
            base="NEON_D13_NIWO_DP1_20200801_161441",
        )
        self.assertEqual(categorize_file(file_obj), "Generic")


class TestGenerateFileMoveList(unittest.TestCase):
    """Test the generate_file_move_list function."""

    def setUp(self):
        """Create a temporary directory structure for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.base_folder = os.path.join(self.test_dir, "processed_flight_lines")
        self.dest_folder = os.path.join(self.test_dir, "cross-sensor-cal")

        # Create test directory structure
        os.makedirs(self.base_folder)
        os.makedirs(self.dest_folder)

        # Create site directories
        self.site1_dir = os.path.join(
            self.base_folder, "NEON_D13_NIWO_DP1_20200801_161441_reflectance"
        )
        self.site2_dir = os.path.join(
            self.base_folder, "NEON_D10_RMNP_DP1_20200701_153414_reflectance"
        )
        os.makedirs(self.site1_dir)
        os.makedirs(self.site2_dir)

        # Create convolution and standard resample directories
        self.conv_tm_dir = os.path.join(
            self.base_folder, "Convolution_Reflectance_Resample_Landsat_5_TM"
        )
        self.std_oli_dir = os.path.join(
            self.base_folder, "Standard_Reflectance_Resample_Landsat_8_OLI"
        )
        os.makedirs(self.conv_tm_dir)
        os.makedirs(self.std_oli_dir)

    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.test_dir)

    def create_test_file(self, directory, filename):
        """Create an empty test file."""
        filepath = os.path.join(directory, filename)
        open(filepath, "a").close()
        return filepath

    def test_basic_file_sorting(self):
        """Test basic file sorting with various file types."""
        # Create test files in site1
        self.create_test_file(
            self.site1_dir, "NEON_D13_NIWO_DP1_20200801_161441_reflectance.h5"
        )
        self.create_test_file(
            self.site1_dir, "NEON_D13_NIWO_DP1_20200801_161441_reflectance_envi.img"
        )
        self.create_test_file(
            self.site1_dir, "NEON_D13_NIWO_DP1_20200801_161441_reflectance_envi.hdr"
        )
        self.create_test_file(
            self.site1_dir, "NEON_D13_NIWO_DP1_20200801_161441_brdf_corrected_envi.img"
        )
        self.create_test_file(
            self.site1_dir, "NEON_D13_NIWO_DP1_20200801_161441_brdf_corrected_envi.hdr"
        )
        self.create_test_file(
            self.site1_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_brdf_corrected_mask_envi.img",
        )
        self.create_test_file(
            self.site1_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_brdf_corrected_mask_envi.hdr",
        )
        self.create_test_file(
            self.site1_dir, "NEON_D13_NIWO_DP1_20200801_161441_config_envi.json"
        )
        self.create_test_file(
            self.site1_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_reflectance_ancillary_envi.img",
        )

        # Run the function
        df = generate_file_move_list(self.base_folder, self.dest_folder)

        # Verify results
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 2)
        self.assertIn("Source Path", df.columns)
        self.assertIn("Destination Path", df.columns)

        # Check that files are categorized correctly
        for _, row in df.iterrows():
            source = row["Source Path"]
            dest = row["Destination Path"]

            # Source paths should be local (no iRODS prefix)
            self.assertFalse(source.startswith("i:/iplant/"))
            # Destination paths should have iRODS prefix
            self.assertTrue(dest.startswith("i:/iplant/"))

            # Check categorization
            if (
                "_reflectance.h5" in source
                or "_config_" in source
                or "_ancillary_" in source
            ):
                self.assertIn("/Generic/", dest)
            elif "_brdf_corrected_mask_" in source:
                self.assertIn("/Reflectance_Masked/", dest)
            elif "_reflectance_envi" in source or "_brdf_corrected_envi" in source:
                self.assertIn("/Reflectance/", dest)

    def test_sensor_file_sorting(self):
        """Test sorting of sensor-specific resampled files."""
        # Create resampled files in convolution directory
        self.create_test_file(
            self.conv_tm_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_resampled_Landsat_5_TM_envi.img",
        )
        self.create_test_file(
            self.conv_tm_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_resampled_Landsat_5_TM_envi.hdr",
        )
        self.create_test_file(
            self.conv_tm_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_resampled_mask_Landsat_5_TM_envi.img",
        )
        self.create_test_file(
            self.conv_tm_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_resampled_mask_Landsat_5_TM_envi.hdr",
        )

        # Create resampled files in standard directory
        self.create_test_file(
            self.std_oli_dir,
            "NEON_D10_RMNP_DP1_20200701_153414_resampled_Landsat_8_OLI_envi.img",
        )
        self.create_test_file(
            self.std_oli_dir,
            "NEON_D10_RMNP_DP1_20200701_153414_resampled_Landsat_8_OLI_envi.hdr",
        )
        self.create_test_file(
            self.std_oli_dir,
            "NEON_D10_RMNP_DP1_20200701_153414_resampled_mask_Landsat_8_OLI_envi.img",
        )
        self.create_test_file(
            self.std_oli_dir,
            "NEON_D10_RMNP_DP1_20200701_153414_resampled_mask_Landsat_8_OLI_envi.hdr",
        )

        # Run the function
        df = generate_file_move_list(self.base_folder, self.dest_folder)

        # Check sensor categorization
        for _, row in df.iterrows():
            source = row["Source Path"]
            dest = row["Destination Path"]

            if "_resampled_Landsat_5_TM_" in source and "_mask_" not in source:
                self.assertIn("/Landsat_5_TM/", dest)
            elif "_resampled_mask_Landsat_5_TM_" in source:
                self.assertIn("/Landsat_5_TM_Masked/", dest)
            elif "_resampled_Landsat_8_OLI_" in source and "_mask_" not in source:
                self.assertIn("/Landsat_8_OLI/", dest)
            elif "_resampled_mask_Landsat_8_OLI_" in source:
                self.assertIn("/Landsat_8_OLI_Masked/", dest)

    def test_multiple_sites(self):
        """Test handling of multiple site directories."""
        # Create files in both sites
        self.create_test_file(
            self.site1_dir, "NEON_D13_NIWO_DP1_20200801_161441_reflectance_envi.img"
        )
        self.create_test_file(
            self.site2_dir, "NEON_D10_RMNP_DP1_20200701_153414_reflectance_envi.img"
        )

        # Run the function
        df = generate_file_move_list(self.base_folder, self.dest_folder)

        # Verify both sites are processed
        sources = df["Source Path"].tolist()
        self.assertTrue(any("D13_NIWO" in s for s in sources))
        self.assertTrue(any("D10_RMNP" in s for s in sources))

    def test_csv_output(self):
        """Test that CSV file is created correctly."""
        # Create a test file
        self.create_test_file(
            self.site1_dir, "NEON_D13_NIWO_DP1_20200801_161441_reflectance.h5"
        )

        # Run the function
        df = generate_file_move_list(self.base_folder, self.dest_folder)

        # Check CSV was created
        csv_path = os.path.join(
            self.dest_folder, "sorted_files", "envi_file_move_list.csv"
        )
        self.assertTrue(os.path.exists(csv_path))

        # Read CSV and verify contents
        df_csv = pd.read_csv(csv_path)
        self.assertTrue(df.equals(df_csv))

    def test_empty_directory(self):
        """Test handling of empty directories."""
        # Run with empty base folder
        df = generate_file_move_list(self.base_folder, self.dest_folder)

        # Should return empty DataFrame
        self.assertEqual(len(df), 0)

        # CSV should still be created
        csv_path = os.path.join(
            self.dest_folder, "sorted_files", "envi_file_move_list.csv"
        )
        self.assertTrue(os.path.exists(csv_path))

    def test_masked_file_variants(self):
        """Test different masked file naming variants."""
        # Create masked files with different naming patterns
        self.create_test_file(
            self.site1_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_reflectance_envi_masked.img",
        )
        self.create_test_file(
            self.site1_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_brdf_corrected_envi_masked.img",
        )

        # Create MicaSense resampled files
        self.create_test_file(
            self.site1_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_resampled_MicaSense_envi.img",
        )
        self.create_test_file(
            self.site1_dir,
            "NEON_D13_NIWO_DP1_20200801_161441_resampled_MicaSense_envi_masked.img",
        )

        # Run the function
        df = generate_file_move_list(self.base_folder, self.dest_folder)

        # Check masked file categorization
        for _, row in df.iterrows():
            source = row["Source Path"]
            dest = row["Destination Path"]

            if (
                "_reflectance_envi_masked" in source
                or "_brdf_corrected_envi_masked" in source
            ):
                self.assertIn("/Reflectance_Masked/", dest)
            elif "_resampled_MicaSense_envi_masked" in source:
                self.assertIn("/MicaSense_Masked/", dest)
            elif "_resampled_MicaSense_envi" in source and "_masked" not in source:
                self.assertIn("/MicaSense/", dest)

    def test_convolution_and_standard_resampled_files(self):
        """Test convolution and standard resampled files for all sensors."""
        # Test each sensor type from the enum
        sensor_mappings = [
            (SensorType.LANDSAT_5_TM, "Landsat_5_TM", "Landsat 5 TM"),
            (SensorType.LANDSAT_7_ETM_PLUS, "Landsat_7_ETM+", "Landsat 7 ETM+"),
            (SensorType.LANDSAT_8_OLI, "Landsat_8_OLI", "Landsat 8 OLI"),
            (SensorType.LANDSAT_9_OLI_2, "Landsat_9_OLI-2", "Landsat 9 OLI-2"),
            (SensorType.MICASENSE, "MicaSense", "MicaSense"),
            (
                SensorType.MICASENSE_MATCH_TM_ETM,
                "MicaSense-to-match_TM_and_ETM+",
                "MicaSense-to-match TM and ETM+",
            ),
            (
                SensorType.MICASENSE_MATCH_OLI,
                "MicaSense-to-match_OLI_and_OLI-2",
                "MicaSense-to-match OLI and OLI-2",
            ),
        ]

        for sensor_enum, sensor_file, expected_category in sensor_mappings:
            # Test unmasked resampled ENVI file
            file_obj = NEONReflectanceResampledENVIFile(
                path=Path(
                    f"test/NEON_D13_NIWO_DP1_20200801_161441_resampled_{sensor_file}_envi.img"
                ),
                domain="D13",
                site="NIWO",
                date="20200801",
                time="161441",
                sensor=sensor_file,
                suffix="envi",
            )
            self.assertEqual(categorize_file(file_obj), expected_category)

            # Test unmasked resampled HDR file
            file_obj = NEONReflectanceResampledHDRFile(
                path=Path(
                    f"test/NEON_D13_NIWO_DP1_20200801_161441_resampled_{sensor_file}_envi.hdr"
                ),
                domain="D13",
                site="NIWO",
                date="20200801",
                time="161441",
                sensor=sensor_file,
                suffix="envi",
            )
            self.assertEqual(categorize_file(file_obj), expected_category)

            # Test masked resampled ENVI file
            file_obj = NEONReflectanceResampledMaskENVIFile(
                path=Path(
                    f"test/NEON_D13_NIWO_DP1_20200801_161441_resampled_mask_{sensor_file}_envi.img"
                ),
                domain="D13",
                site="NIWO",
                date="20200801",
                time="161441",
                sensor=sensor_file,
                suffix="envi",
            )
            self.assertEqual(categorize_file(file_obj), f"{expected_category}_Masked")

            # Test masked resampled HDR file
            file_obj = NEONReflectanceResampledMaskHDRFile(
                path=Path(
                    f"test/NEON_D13_NIWO_DP1_20200801_161441_resampled_mask_{sensor_file}_envi.hdr"
                ),
                domain="D13",
                site="NIWO",
                date="20200801",
                time="161441",
                sensor=sensor_file,
                suffix="envi",
            )
            self.assertEqual(categorize_file(file_obj), f"{expected_category}_Masked")

    def test_all_sensor_types(self):
        """Test all sensor types mentioned in the requirements."""
        # Map sensor file names to their expected directory names
        sensor_files = [
            ("Landsat_5_TM", "Landsat_5_TM"),
            ("Landsat_7_ETM+", "Landsat_7_ETM+"),  # Special chars preserved
            ("Landsat_8_OLI", "Landsat_8_OLI"),
            ("Landsat_9_OLI-2", "Landsat_9_OLI-2"),  # Hyphens preserved
            ("MicaSense", "MicaSense"),
            (
                "MicaSense-to-match_TM_and_ETM+",
                "MicaSense-to-match_TM_and_ETM+",
            ),  # Mixed chars preserved
            (
                "MicaSense-to-match_OLI_and_OLI-2",
                "MicaSense-to-match_OLI_and_OLI-2",
            ),  # Mixed chars preserved
        ]

        # Create files for each sensor type
        for sensor_file, sensor_dir in sensor_files:
            self.create_test_file(
                self.site1_dir,
                f"NEON_D13_NIWO_DP1_20200801_161441_resampled_{sensor_file}_envi.img",
            )
            self.create_test_file(
                self.site1_dir,
                f"NEON_D13_NIWO_DP1_20200801_161441_resampled_mask_{sensor_file}_envi.img",
            )

        # Run the function
        df = generate_file_move_list(self.base_folder, self.dest_folder)

        # Verify all sensors are categorized correctly
        for sensor_file, expected_dir in sensor_files:
            # Escape special regex characters in sensor name
            sensor_file_escaped = re.escape(sensor_file)

            # Check unmasked sensor files
            unmasked_rows = df[
                df["Source Path"].str.contains(f"_resampled_{sensor_file_escaped}_")
                & ~df["Source Path"].str.contains("_mask_")
            ]
            self.assertTrue(
                len(unmasked_rows) > 0,
                f"No unmasked files found for sensor {sensor_file}",
            )
            for _, row in unmasked_rows.iterrows():
                self.assertIn(f"/{expected_dir}/", row["Destination Path"])

            # Check masked sensor files
            masked_rows = df[
                df["Source Path"].str.contains(
                    f"_resampled_mask_{sensor_file_escaped}_"
                )
            ]
            self.assertTrue(len(masked_rows) > 0)
            for _, row in masked_rows.iterrows():
                self.assertIn(f"/{expected_dir}_Masked/", row["Destination Path"])

    def test_custom_remote_path_prefix(self):
        """Test custom remote path prefix functionality."""
        # Create a test file
        self.create_test_file(
            self.site1_dir, "NEON_D13_NIWO_DP1_20200801_161441_reflectance.h5"
        )

        # Test with custom prefix
        custom_prefix = "projects/neon/data"
        df = generate_file_move_list(self.base_folder, self.dest_folder, custom_prefix)

        # Verify custom prefix is applied
        for _, row in df.iterrows():
            dest = row["Destination Path"]
            self.assertTrue(dest.startswith(f"i:/iplant/{custom_prefix}/"))
            # Verify the rest of the path structure is maintained
            self.assertIn("/sorted_files/envi/", dest)

        # Test without custom prefix (default behavior)
        df_default = generate_file_move_list(self.base_folder, self.dest_folder)
        for _, row in df_default.iterrows():
            dest = row["Destination Path"]
            # Should start with i:/iplant/ directly followed by the local path
            self.assertTrue(dest.startswith("i:/iplant/"))
            self.assertNotIn("/projects/neon/data/", dest)

    def test_comprehensive_sensor_sorting(self):
        """Test comprehensive sorting with all sensor types and file variants."""
        # Create a mix of convolution and standard directories
        conv_dir = os.path.join(self.base_folder, "Convolution_Reflectance_Resample")
        std_dir = os.path.join(self.base_folder, "Standard_Reflectance_Resample")
        os.makedirs(conv_dir)
        os.makedirs(std_dir)

        # Create files for each sensor type with all variants
        sensor_mappings = [
            ("Landsat_5_TM", "Landsat_5_TM"),
            ("Landsat_7_ETM+", "Landsat_7_ETM+"),
            ("Landsat_8_OLI", "Landsat_8_OLI"),
            ("Landsat_9_OLI-2", "Landsat_9_OLI-2"),
            ("MicaSense", "MicaSense"),
            ("MicaSense-to-match_TM_and_ETM+", "MicaSense-to-match_TM_and_ETM+"),
            ("MicaSense-to-match_OLI_and_OLI-2", "MicaSense-to-match_OLI_and_OLI-2"),
        ]

        for sensor_file, sensor_dir in sensor_mappings:
            # Create in convolution directory
            self.create_test_file(
                conv_dir,
                f"NEON_D13_NIWO_DP1_20200801_161441_resampled_{sensor_file}_envi.img",
            )
            self.create_test_file(
                conv_dir,
                f"NEON_D13_NIWO_DP1_20200801_161441_resampled_{sensor_file}_envi.hdr",
            )
            self.create_test_file(
                conv_dir,
                f"NEON_D13_NIWO_DP1_20200801_161441_resampled_mask_{sensor_file}_envi.img",
            )
            self.create_test_file(
                conv_dir,
                f"NEON_D13_NIWO_DP1_20200801_161441_resampled_mask_{sensor_file}_envi.hdr",
            )

            # Create in standard directory with different site
            self.create_test_file(
                std_dir,
                f"NEON_D10_RMNP_DP1_20200701_153414_resampled_{sensor_file}_envi.img",
            )
            self.create_test_file(
                std_dir,
                f"NEON_D10_RMNP_DP1_20200701_153414_resampled_{sensor_file}_envi.hdr",
            )
            self.create_test_file(
                std_dir,
                f"NEON_D10_RMNP_DP1_20200701_153414_resampled_mask_{sensor_file}_envi.img",
            )
            self.create_test_file(
                std_dir,
                f"NEON_D10_RMNP_DP1_20200701_153414_resampled_mask_{sensor_file}_envi.hdr",
            )

        # Run the function
        df = generate_file_move_list(self.base_folder, self.dest_folder)

        # Verify we have files for all sensors (4 files per sensor per directory * 2 directories * 7 sensors)
        self.assertEqual(len(df), 7 * 4 * 2)

        # Verify each sensor has correct categorization
        for sensor_file, expected_dir in sensor_mappings:
            # Escape special characters for regex
            expected_dir_escaped = re.escape(expected_dir)

            # Check unmasked files
            unmasked_count = len(
                df[
                    df["Destination Path"].str.contains(f"/{expected_dir_escaped}/")
                    & ~df["Destination Path"].str.contains("_Masked/")
                ]
            )
            self.assertEqual(unmasked_count, 4)  # 2 sites * 2 file types (.img, .hdr)

            # Check masked files
            masked_count = len(
                df[
                    df["Destination Path"].str.contains(
                        f"/{expected_dir_escaped}_Masked/"
                    )
                ]
            )
            self.assertEqual(masked_count, 4)  # 2 sites * 2 file types (.img, .hdr)


if __name__ == "__main__":
    unittest.main()
