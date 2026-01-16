"""Polygon extraction helpers for building spectral libraries.

This module provides utilities for extracting polygon-based subsets from the
per-pixel Parquet products that the Cross-Sensor Calibration pipeline already
produces.  The helpers are intentionally orthogonal to the default flightline
pipeline so that they can be orchestrated separately while the workflow is
stabilised.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Mapping

import duckdb
import numpy as np
import pandas as pd

from cross_sensor_cal.exports.schema_utils import ensure_coord_columns

from ._optional import require_geopandas, require_rasterio
from .paths import FlightlinePaths

LOGGER = logging.getLogger(__name__)


def create_dummy_polygon(
    flight_paths: FlightlinePaths,
    reference_product: str = "brdfandtopo_corrected_envi",
    pixel_size: tuple[int, int] | None = None,  # (width, height) in pixels, e.g., (4, 4) for 4x4
    buffer_pct: float = 0.1,  # Only used if pixel_size is None
    output_path: Path | str | None = None,
) -> Path:
    """Create a dummy polygon covering a portion of the flightline.
    
    Creates a simple rectangular polygon. If `pixel_size` is provided, creates
    a polygon covering exactly that many pixels (e.g., (4, 4) for a 4x4 pixel
    polygon). Otherwise, creates a polygon covering (100 - buffer_pct*2)% of
    the image, centered.
    
    Parameters
    ----------
    flight_paths : FlightlinePaths
        Flightline paths object
    reference_product : str, default "brdfandtopo_corrected_envi"
        Reference product to use for determining image bounds
    pixel_size : tuple[int, int] | None, optional
        Size of polygon in pixels (width, height). If provided, creates a
        polygon covering exactly these pixels from the center of the image.
        Example: (4, 4) for a 4x4 pixel polygon.
        If None, uses buffer_pct instead.
    buffer_pct : float, default 0.1
        Percentage to buffer from edges (0.1 = 10% buffer on each side).
        Only used if pixel_size is None.
    output_path : Path | str | None, optional
        Output path for the dummy polygon GeoJSON. If None, creates:
        <flight_id>_dummy_polygon.geojson
    
    Returns
    -------
    Path
        Path to the created dummy polygon GeoJSON file
    
    Notes
    -----
    To use a different polygon, provide a GeoJSON file path to `polygon_path`
    in `go_forth_and_multiply()` instead of using this function.
    """
    geopandas = require_geopandas()
    rasterio = require_rasterio()
    from shapely.geometry import box
    from rasterio.transform import xy
    
    polygons_path = Path(output_path) if output_path else (
        flight_paths.flight_dir / f"{flight_paths.flight_id}_dummy_polygon.geojson"
    )
    
    if polygons_path.exists():
        # Validate existing polygon before reusing
        try:
            geopandas = require_geopandas()
            existing_gdf = geopandas.read_file(polygons_path)
            if not existing_gdf.empty:
                bounds = existing_gdf.total_bounds
                # Check if bounds are valid (not inf or nan)
                if all(np.isfinite(bounds)):
                    LOGGER.info("[polygons-dummy] Reusing existing dummy polygon → %s", polygons_path)
                    return polygons_path
                else:
                    LOGGER.warning(
                        "[polygons-dummy] ⚠️  Existing polygon has invalid bounds (inf/nan). "
                        "Will recreate polygon."
                    )
            else:
                LOGGER.warning(
                    "[polygons-dummy] ⚠️  Existing polygon file is empty. Will recreate polygon."
                )
        except Exception as e:
            LOGGER.warning(
                "[polygons-dummy] ⚠️  Failed to validate existing polygon: %s. Will recreate polygon.",
                e
            )
        # If validation failed, delete the invalid polygon and recreate
        polygons_path.unlink(missing_ok=True)
        LOGGER.info("[polygons-dummy] Deleted invalid polygon, will create new one")
    
    img_path, hdr_path = _resolve_reference_raster(flight_paths, reference_product)
    if not img_path.exists():
        raise FileNotFoundError(
            f"Reference image for {reference_product!r} not found: {img_path}"
        )
    
    with rasterio.open(img_path) as src:
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
        width_pixels = src.width
        height_pixels = src.height
    
    # If transform is None, try to build it from ENVI header map info
    # Check if transform is valid (rasterio transforms are Affine objects or tuples)
    # Note: We check for Affine objects first, then fall back to other checks
    transform_valid = False
    if transform is not None:
        try:
            from affine import Affine as AffineClass
            if isinstance(transform, AffineClass):
                transform_valid = True
            elif isinstance(transform, tuple) and len(transform) == 6:
                transform_valid = True
            elif callable(transform):
                transform_valid = True
        except ImportError:
            # Fallback if affine package not available
            if callable(transform) or (isinstance(transform, tuple) and len(transform) == 6):
                transform_valid = True
    
    if not transform_valid:
        LOGGER.warning(
            "[polygons-dummy] Raster has no transform. Attempting to build from ENVI header..."
        )
        try:
            from cross_sensor_cal.envi import _parse_envi_header_tolerant
            from cross_sensor_cal.io.neon import _map_info_core
            
            LOGGER.info("[polygons-dummy] Reading ENVI header: %s", hdr_path)
            header = _parse_envi_header_tolerant(hdr_path)
            map_info = header.get("map info")
            
            LOGGER.info("[polygons-dummy] Map info from header: %s", map_info)
            
            if map_info:
                if isinstance(map_info, str):
                    # Parse map info string
                    map_info_str = map_info.strip()
                    if map_info_str.startswith("{"):
                        map_info_str = map_info_str[1:]
                    if map_info_str.endswith("}"):
                        map_info_str = map_info_str[:-1]
                    map_info_list = [item.strip() for item in map_info_str.split(",")]
                elif isinstance(map_info, list):
                    map_info_list = [str(item).strip() for item in map_info]
                else:
                    map_info_list = []
                
                LOGGER.info("[polygons-dummy] Parsed map info list: %s", map_info_list)
                
                if len(map_info_list) >= 7:
                    ref_x, ref_y, ref_easting, ref_northing, pixel_x, pixel_y = _map_info_core(
                        map_info_list
                    )
                    LOGGER.info(
                        "[polygons-dummy] Map info values: ref_x=%.2f, ref_y=%.2f, easting=%.2f, northing=%.2f, pixel_x=%.2f, pixel_y=%.2f",
                        ref_x, ref_y, ref_easting, ref_northing, pixel_x, pixel_y
                    )
                    # Build transform: (ulx, pixel_width, 0, uly, 0, pixel_height)
                    # pixel_y is typically negative for north-up images
                    ulx = ref_easting - pixel_x * (ref_x - 0.5)
                    uly = ref_northing + abs(pixel_y) * (ref_y - 0.5)
                    yres = -abs(pixel_y)  # Negative for north-up
                    transform = AffineClass(pixel_x, 0.0, ulx, 0.0, yres, uly)
                    LOGGER.info(
                        "[polygons-dummy] ✅ Built transform from map info: ulx=%.2f, uly=%.2f, xres=%.2f, yres=%.2f",
                        ulx, uly, pixel_x, yres
                    )
                    LOGGER.info("[polygons-dummy] Transform object: %s (type: %s)", transform, type(transform).__name__)
                    # Mark transform as valid since we just built it
                    transform_valid = True
                else:
                    raise ValueError(f"Map info has insufficient elements: {len(map_info_list)}, need at least 7")
            else:
                raise ValueError("No map info found in ENVI header")
        except Exception as e:
            import traceback
            LOGGER.error(
                "[polygons-dummy] ❌ Failed to build transform from ENVI header: %s", e
            )
            LOGGER.error("[polygons-dummy] Traceback: %s", traceback.format_exc())
            raise ValueError(
                f"Invalid transform from raster {img_path} and could not build from ENVI header. "
                f"Transform is None or not callable. Error: {e}"
            ) from e
    
    # Validate transform - if we built it, it should be valid
    # Affine objects are valid transforms, as are tuples of 6 elements
    if transform is None:
        raise ValueError(
            f"Invalid transform from raster {img_path}. "
            f"Transform is None after attempting to build from ENVI header."
        )
    
    # Log the transform for debugging
    try:
        from affine import Affine as AffineClass
        is_affine = isinstance(transform, AffineClass)
    except ImportError:
        is_affine = False
    
    LOGGER.info(
        "[polygons-dummy] ✅ Transform validated: %s (type: %s, is_affine: %s)",
        transform, type(transform).__name__, is_affine
    )
    
    # Validate bounds
    if not all(np.isfinite([bounds.left, bounds.right, bounds.bottom, bounds.top])):
        raise ValueError(
            f"Invalid bounds from raster {img_path}: {bounds}. "
            "Bounds contain inf or nan values."
        )
    
    LOGGER.info(
        "[polygons-dummy] Raster info: width=%d, height=%d, bounds=(%.2f, %.2f, %.2f, %.2f), transform=%s",
        width_pixels, height_pixels, bounds.left, bounds.bottom, bounds.right, bounds.top, transform
    )
    
    # If CRS is not in the raster, try to extract it from the ENVI header
    if crs is None:
        LOGGER.warning(
            "[polygons-dummy] ENVI file has no CRS. Attempting to extract from header..."
        )
        try:
            from cross_sensor_cal.polygon_extraction import get_crs_from_hdr
            crs = get_crs_from_hdr(hdr_path)
            if crs is not None:
                LOGGER.info(
                    "[polygons-dummy] ✅ Extracted CRS from header: %s", crs
                )
                # Try to normalize to EPSG if it's a UTM projection
                try:
                    epsg_code = crs.to_epsg()
                    if epsg_code is not None:
                        LOGGER.info(
                            "[polygons-dummy] Normalized CRS to EPSG:%d", epsg_code
                        )
                        crs = rasterio.crs.CRS.from_epsg(epsg_code)
                except Exception:
                    # If normalization fails, use the extracted CRS as-is
                    pass
            else:
                LOGGER.warning(
                    "[polygons-dummy] ⚠️  Could not extract CRS from header. "
                    "Polygon will be created without CRS (may cause intersection issues)."
                )
        except Exception as e:
            LOGGER.warning(
                "[polygons-dummy] ⚠️  Failed to extract CRS from header: %s", e
            )
    
    # Normalize CRS to EPSG if possible (handles custom PROJCS definitions)
    if crs is not None:
        try:
            epsg_code = crs.to_epsg()
            if epsg_code is not None:
                # Check if it's a UTM Zone 13N (EPSG:32613) or similar
                # If the CRS is a custom PROJCS but represents UTM 13N, normalize it
                if epsg_code == 32613:
                    LOGGER.info(
                        "[polygons-dummy] Normalizing CRS to EPSG:32613 (UTM Zone 13N)"
                    )
                    crs = rasterio.crs.CRS.from_epsg(32613)
                else:
                    LOGGER.info(
                        "[polygons-dummy] Using CRS EPSG:%d", epsg_code
                    )
                    crs = rasterio.crs.CRS.from_epsg(epsg_code)
        except Exception:
            # If normalization fails, use the CRS as-is
            pass
    
    if pixel_size is not None:
        # Create polygon covering exact pixel dimensions from center
        pix_width, pix_height = pixel_size
        center_row = height_pixels // 2
        center_col = width_pixels // 2
        
        # Calculate pixel bounds centered on the image
        # For 4x4: start at center - 2, end at center + 2 (exclusive), giving exactly 4 pixels
        row_start = center_row - pix_height // 2
        row_end = row_start + pix_height
        col_start = center_col - pix_width // 2
        col_end = col_start + pix_width
        
        # Clamp to image bounds
        row_start = max(0, row_start)
        row_end = min(height_pixels, row_end)
        col_start = max(0, col_start)
        col_end = min(width_pixels, col_end)
        
        # Convert pixel coordinates to map coordinates
        # Get corners of the pixel region
        # Use "ul" (upper left) for start corner and "lr" (lower right) for end corner
        # Note: row_end and col_end are exclusive (like Python ranges), so use them directly
        try:
            minx, maxy = xy(transform, row_start, col_start, offset="ul")
            maxx, miny = xy(transform, row_end, col_end, offset="lr")
        except Exception as e:
            LOGGER.error(
                "[polygons-dummy] ❌ Failed to convert pixel coordinates to map coordinates: %s",
                e
            )
            raise ValueError(
                f"Failed to convert pixel coordinates to map coordinates. "
                f"Transform may be invalid. Error: {e}"
            ) from e
        
        # Validate coordinates are finite
        if not all(np.isfinite([minx, miny, maxx, maxy])):
            LOGGER.error(
                "[polygons-dummy] ❌ Invalid coordinates: minx=%.2f, miny=%.2f, maxx=%.2f, maxy=%.2f",
                minx, miny, maxx, maxy
            )
            raise ValueError(
                f"Invalid polygon coordinates (inf or nan). "
                f"minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}. "
                f"This usually indicates an invalid transform or CRS."
            )
        
        # Ensure min < max
        if minx > maxx:
            minx, maxx = maxx, minx
        if miny > maxy:
            miny, maxy = maxy, miny
        
        LOGGER.info(
            "[polygons-dummy] Creating %dx%d pixel polygon at center (rows %d-%d, cols %d-%d)",
            pix_width, pix_height, row_start, row_end, col_start, col_end,
        )
        LOGGER.info(
            "[polygons-dummy] Polygon map coordinates: minx=%.2f, miny=%.2f, maxx=%.2f, maxy=%.2f",
            minx, miny, maxx, maxy
        )
    else:
        # Calculate buffer in map units
        width = bounds.right - bounds.left
        height = bounds.top - bounds.bottom
        x_buffer = width * buffer_pct
        y_buffer = height * buffer_pct
        
        # Create a box with buffer from edges
        minx = bounds.left + x_buffer
        miny = bounds.bottom + y_buffer
        maxx = bounds.right - x_buffer
        maxy = bounds.top - y_buffer
    
    # Create polygon
    polygon_geom = box(minx, miny, maxx, maxy)
    
    # Ensure CRS is set (required for spatial operations)
    if crs is None:
        LOGGER.error(
            "[polygons-dummy] ❌ Cannot create polygon without CRS. "
            "ENVI file has no CRS information. Please check your ENVI header file."
        )
        raise ValueError(
            "Cannot create polygon without CRS. "
            "The ENVI file must have CRS information in the header "
            "(coordinate system string or map info with UTM zone)."
        )
    
    # Create GeoDataFrame
    desc = (
        f"{pixel_size[0]}x{pixel_size[1]} pixel polygon" if pixel_size
        else f"Dummy polygon covering {100 - buffer_pct*200:.1f}% of image"
    )
    gdf = geopandas.GeoDataFrame(
        {
            "polygon_id": [1],
            "name": ["dummy_polygon"],
            "description": [desc],
        },
        geometry=[polygon_geom],
        crs=crs,
    )
    
    LOGGER.info(
        "[polygons-dummy] Polygon CRS: %s (EPSG: %s)",
        crs, crs.to_epsg() if crs else "None"
    )
    
    # Save to GeoJSON
    polygons_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(polygons_path, driver="GeoJSON")
    
    LOGGER.info(
        "[polygons-dummy] ✅ Created dummy polygon → %s (bounds: %.2f, %.2f, %.2f, %.2f)",
        polygons_path,
        minx, miny, maxx, maxy,
    )
    
    return polygons_path


def validate_coordinate_match(
    polygons,
    img_path: Path,
    hdr_path: Path,
    *,
    tolerance_m: float = 10000.0,  # Increased default to 10km to handle large flightlines
) -> tuple[bool, str]:
    """Validate that polygon and ENVI coordinates match.
    
    Parameters
    ----------
    polygons
        GeoDataFrame containing polygons to validate
    img_path : Path
        Path to ENVI image file
    hdr_path : Path
        Path to ENVI header file
    tolerance_m : float, default 100.0
        Tolerance in meters for coordinate validation
    
    Returns
    -------
    tuple[bool, str]
        (is_valid, message) - True if coordinates match, False otherwise with error message
    """
    rasterio = require_rasterio()
    
    with rasterio.open(img_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
    
    # Extract CRS from header if needed
    if raster_crs is None:
        try:
            from cross_sensor_cal.polygon_extraction import get_crs_from_hdr
            raster_crs = get_crs_from_hdr(hdr_path)
        except Exception:
            pass
    
    # Check if CRS is available
    if raster_crs is None:
        return False, "ENVI file has no CRS information. Cannot validate coordinate match."
    
    if polygons.crs is None:
        return False, "Polygons have no CRS information. Cannot validate coordinate match."
    
    # Normalize CRS to EPSG for comparison
    try:
        raster_epsg = raster_crs.to_epsg()
        poly_epsg = polygons.crs.to_epsg()
    except Exception:
        return False, "Cannot convert CRS to EPSG for comparison."
    
    # Check if CRS matches
    if raster_epsg != poly_epsg:
        # Try transforming polygons to raster CRS for validation
        try:
            polygons_aligned = polygons.to_crs(raster_crs)
        except Exception as e:
            return False, f"CRS mismatch: Raster is EPSG:{raster_epsg}, Polygons are EPSG:{poly_epsg}. Failed to transform: {e}"
        
        # Check if bounds are reasonable after transformation
        poly_bounds = polygons_aligned.total_bounds
        raster_bounds_list = [raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top]
        
        # Calculate distance between bounds centers
        poly_center_x = (poly_bounds[0] + poly_bounds[2]) / 2
        poly_center_y = (poly_bounds[1] + poly_bounds[3]) / 2
        raster_center_x = (raster_bounds_list[0] + raster_bounds_list[2]) / 2
        raster_center_y = (raster_bounds_list[1] + raster_bounds_list[3]) / 2
        
        distance = ((poly_center_x - raster_center_x) ** 2 + (poly_center_y - raster_center_y) ** 2) ** 0.5
        
        # Log detailed CRS and coordinate information
        LOGGER.info(
            f"[coordinate-validation] CRS: Raster=EPSG:{raster_epsg}, Polygons=EPSG:{poly_epsg} (transformed)"
        )
        LOGGER.info(
            f"[coordinate-validation] Polygon bounds: ({poly_bounds[0]:.1f}, {poly_bounds[1]:.1f}, {poly_bounds[2]:.1f}, {poly_bounds[3]:.1f})"
        )
        LOGGER.info(
            f"[coordinate-validation] Raster bounds: ({raster_bounds_list[0]:.1f}, {raster_bounds_list[1]:.1f}, {raster_bounds_list[2]:.1f}, {raster_bounds_list[3]:.1f})"
        )
        LOGGER.info(
            f"[coordinate-validation] Distance between centers: {distance:.1f}m (tolerance: {tolerance_m}m)"
        )
        
        # Check if bounds overlap (more lenient check)
        bounds_overlap = (
            poly_bounds[0] < raster_bounds_list[2] and
            poly_bounds[2] > raster_bounds_list[0] and
            poly_bounds[1] < raster_bounds_list[3] and
            poly_bounds[3] > raster_bounds_list[1]
        )
        
        if bounds_overlap:
            return True, f"CRS transformed from EPSG:{poly_epsg} to EPSG:{raster_epsg}. Bounds overlap (distance: {distance:.1f}m)."
        
        if distance > tolerance_m:
            return False, (
                f"CRS transformed but coordinates are far apart: "
                f"Polygon center ({poly_center_x:.1f}, {poly_center_y:.1f}) is "
                f"{distance:.1f}m from raster center ({raster_center_x:.1f}, {raster_center_y:.1f}). "
                f"Tolerance: {tolerance_m}m. "
                f"Bounds do not overlap. Check if polygons and raster are from the same geographic area."
            )
        
        return True, f"CRS transformed from EPSG:{poly_epsg} to EPSG:{raster_epsg}. Coordinates match within tolerance ({distance:.1f}m)."
    
    # CRS matches - check if bounds overlap reasonably
    poly_bounds = polygons.total_bounds
    raster_bounds_list = [raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top]
    
    # Check if bounds overlap
    bounds_overlap = (
        poly_bounds[0] < raster_bounds_list[2] and
        poly_bounds[2] > raster_bounds_list[0] and
        poly_bounds[1] < raster_bounds_list[3] and
        poly_bounds[3] > raster_bounds_list[1]
    )
    
    if not bounds_overlap:
        # Calculate distance between bounds
        poly_center_x = (poly_bounds[0] + poly_bounds[2]) / 2
        poly_center_y = (poly_bounds[1] + poly_bounds[3]) / 2
        raster_center_x = (raster_bounds_list[0] + raster_bounds_list[2]) / 2
        raster_center_y = (raster_bounds_list[1] + raster_bounds_list[3]) / 2
        
        distance = ((poly_center_x - raster_center_x) ** 2 + (poly_center_y - raster_center_y) ** 2) ** 0.5
        
        return False, (
            f"Bounds do not overlap. Polygon center ({poly_center_x:.1f}, {poly_center_y:.1f}) is "
            f"{distance:.1f}m from raster center ({raster_center_x:.1f}, {raster_center_y:.1f}). "
            f"Polygon bounds: ({poly_bounds[0]:.1f}, {poly_bounds[1]:.1f}, {poly_bounds[2]:.1f}, {poly_bounds[3]:.1f}), "
            f"Raster bounds: ({raster_bounds_list[0]:.1f}, {raster_bounds_list[1]:.1f}, {raster_bounds_list[2]:.1f}, {raster_bounds_list[3]:.1f})"
        )
    
    return True, f"Coordinates match. Both use EPSG:{raster_epsg}. Bounds overlap."


def filter_polygons_by_overlap(
    polygons_path: str | Path,
    flight_paths: FlightlinePaths,
    reference_product: str = "brdfandtopo_corrected_envi",
    min_overlap_pct: float = 0.0,
    *,
    output_path: Path | str | None = None,
    validate_coords: bool = True,
    search_buffer_m: float = 0.0,
) -> tuple[Path, pd.DataFrame]:
    """Filter polygons from GeoJSON that overlap the flightline.
    
    Parameters
    ----------
    polygons_path : str | Path
        Path to GeoJSON file containing polygons
    flight_paths : FlightlinePaths
        Flightline paths object
    reference_product : str, default "brdfandtopo_corrected_envi"
        Reference product to use for determining flightline bounds
    min_overlap_pct : float, default 0.0
        Minimum percentage of polygon area that must overlap with flightline.
        Set to 0.0 to include any polygon that touches the flightline.
        Set to 1.0 to only include polygons completely inside the flightline.
    output_path : Path | str | None, optional
        Path to save filtered GeoJSON. If None, saves to flightline directory.
    search_buffer_m : float, default 0.0
        Buffer distance in meters to expand the flightline search area.
        Polygons within this distance of the flightline will be included even if
        they don't directly overlap. Useful when polygons are nearby but don't
        intersect due to precision issues or small gaps.
    
    Returns
    -------
    tuple[Path, pd.DataFrame]
        Path to filtered GeoJSON file and DataFrame with overlap statistics
    """
    geopandas = require_geopandas()
    rasterio = require_rasterio()
    
    polygons_path = Path(polygons_path)
    if not polygons_path.exists():
        raise FileNotFoundError(f"Polygon file not found: {polygons_path}")
    
    img_path, hdr_path = _resolve_reference_raster(flight_paths, reference_product)
    if not img_path.exists():
        raise FileNotFoundError(
            f"Reference image for {reference_product!r} not found: {img_path}"
        )
    
    # Read polygons
    polygons = geopandas.read_file(polygons_path)
    if polygons.empty:
        raise ValueError(f"No polygons found in {polygons_path}")
    
    LOGGER.info(f"[polygons-filter] Loaded {len(polygons)} polygons from {polygons_path}")
    LOGGER.info(f"[polygons-filter] Search buffer: {search_buffer_m}m")
    
    # Extract site code from flightline ID (e.g., "NEON_D13_NIWO_DP1_20200720_173638_reflectance" -> "NIWO")
    flight_id = flight_paths.flight_id
    site_code = None
    site_match = re.search(r'NEON_D\d+_([A-Z0-9]{4})_', flight_id)
    if site_match:
        site_code = site_match.group(1)
        LOGGER.info(f"[polygons-filter] Extracted site code from flightline: {site_code}")
    
    # Filter by site code if available in polygon attributes
    if site_code and "aop_site" in polygons.columns:
        initial_count = len(polygons)
        available_sites = sorted(polygons["aop_site"].unique().tolist())
        polygons = polygons[polygons["aop_site"] == site_code].copy()
        LOGGER.info(
            f"[polygons-filter] Filtered by site '{site_code}': {initial_count} -> {len(polygons)} polygons"
        )
        if polygons.empty:
            raise ValueError(
                f"No polygons found for site '{site_code}' in {polygons_path}. "
                f"Available sites: {available_sites}"
            )
        
        # Log polygon metadata for debugging
        if "imagery_year" in polygons.columns:
            years = sorted(polygons["imagery_year"].dropna().unique().tolist())
            LOGGER.info(f"[polygons-filter] Polygon imagery years: {years}")
        if "location" in polygons.columns:
            locations = polygons["location"].value_counts().head(10).to_dict()
            LOGGER.info(f"[polygons-filter] Top polygon locations: {locations}")
        
        # Extract date from flightline ID for potential date-based filtering
        # Format: NEON_D13_NIWO_DP1_20200720_173638_reflectance -> 20200720
        flight_date_match = re.search(r'_(\d{8})_', flight_id)
        if flight_date_match:
            flight_date_str = flight_date_match.group(1)
            flight_year = int(flight_date_str[:4])
            flight_month = int(flight_date_str[4:6])
            LOGGER.info(
                f"[polygons-filter] Flightline date: {flight_date_str} (year={flight_year}, month={flight_month})"
            )
            
            # Optionally filter by imagery_year if available
            # Note: This is commented out because polygons might be from different years
            # but still valid for the flightline. Uncomment if needed.
            # if "imagery_year" in polygons.columns:
            #     initial_count = len(polygons)
            #     polygons = polygons[polygons["imagery_year"] == flight_year].copy()
            #     LOGGER.info(
            #         f"[polygons-filter] Filtered by imagery_year={flight_year}: {initial_count} -> {len(polygons)} polygons"
            #     )
    elif site_code:
        LOGGER.warning(
            f"[polygons-filter] Site code '{site_code}' extracted but 'aop_site' column not found in polygons. "
            "Skipping site-based filtering."
        )
    
    # Validate coordinate matching (with more lenient tolerance for large flightlines)
    if validate_coords:
        is_valid, message = validate_coordinate_match(polygons, img_path, hdr_path, tolerance_m=50000.0)
        if not is_valid:
            LOGGER.warning(f"[polygons-filter] ⚠️  Coordinate validation warning: {message}")
            LOGGER.warning(
                "[polygons-filter] Continuing anyway - bounds overlap check will be performed during extraction. "
                "If no pixels intersect, the extraction will fail with a clear error."
            )
            # Don't raise error, just warn - let the actual extraction determine if polygons overlap
        else:
            LOGGER.info(f"[polygons-filter] ✅ Coordinate validation passed: {message}")
    
    # Get raster bounds and CRS
    with rasterio.open(img_path) as src:
        transform = src.transform
        dataset_crs = src.crs
        bounds = src.bounds
    
    # Build transform from ENVI header if needed
    if transform is None or not hasattr(transform, '__call__'):
        LOGGER.warning("[polygons-filter] Building transform from ENVI header...")
        try:
            from cross_sensor_cal.envi import _parse_envi_header_tolerant
            from cross_sensor_cal.io.neon import _map_info_core
            
            header = _parse_envi_header_tolerant(hdr_path)
            map_info = header.get("map info")
            
            if map_info:
                if isinstance(map_info, str):
                    map_info_str = map_info.strip().strip("{}")
                    map_info_list = [item.strip() for item in map_info_str.split(",")]
                elif isinstance(map_info, list):
                    map_info_list = [str(item).strip() for item in map_info]
                else:
                    map_info_list = []
                
                if len(map_info_list) >= 7:
                    ref_x, ref_y, ref_easting, ref_northing, pixel_x, pixel_y = _map_info_core(
                        map_info_list
                    )
                    ulx = ref_easting - pixel_x * (ref_x - 0.5)
                    uly = ref_northing + abs(pixel_y) * (ref_y - 0.5)
                    yres = -abs(pixel_y)
                    transform = rasterio.Affine(pixel_x, 0.0, ulx, 0.0, yres, uly)
        except Exception as e:
            LOGGER.error(f"[polygons-filter] Failed to build transform: {e}")
            raise
    
    # Extract CRS from header if needed
    if dataset_crs is None:
        try:
            from cross_sensor_cal.polygon_extraction import get_crs_from_hdr
            dataset_crs = get_crs_from_hdr(hdr_path)
        except Exception:
            pass
    
    # Normalize CRS
    if dataset_crs is not None:
        try:
            epsg_code = dataset_crs.to_epsg()
            if epsg_code is not None:
                dataset_crs = rasterio.crs.CRS.from_epsg(epsg_code)
        except Exception:
            pass
    
    # Log polygon bounds before transformation
    if len(polygons) > 0:
        poly_bounds_before = polygons.total_bounds
        LOGGER.info(
            f"[polygons-filter] Polygon bounds before CRS transformation: "
            f"({poly_bounds_before[0]:.2f}, {poly_bounds_before[1]:.2f}, "
            f"{poly_bounds_before[2]:.2f}, {poly_bounds_before[3]:.2f}) "
            f"[CRS: {polygons.crs}]"
        )
    
    # Align polygon CRS with raster CRS
    if polygons.crs is None and dataset_crs is not None:
        polygons = polygons.set_crs(dataset_crs)
        LOGGER.info(f"[polygons-filter] Set polygon CRS to {dataset_crs}")
    elif dataset_crs is not None and polygons.crs != dataset_crs:
        LOGGER.info(
            f"[polygons-filter] Transforming polygons from {polygons.crs} to {dataset_crs}"
        )
        polygons = polygons.to_crs(dataset_crs)
        # Log bounds after transformation
        if len(polygons) > 0:
            poly_bounds_after = polygons.total_bounds
            LOGGER.info(
                f"[polygons-filter] Polygon bounds after CRS transformation: "
                f"({poly_bounds_after[0]:.2f}, {poly_bounds_after[1]:.2f}, "
                f"{poly_bounds_after[2]:.2f}, {poly_bounds_after[3]:.2f}) "
                f"[CRS: {polygons.crs}]"
            )
    
    # Create flightline bounding box (with optional buffer)
    from shapely.geometry import box
    LOGGER.info(
        f"[polygons-filter] search_buffer_m parameter value: {search_buffer_m}m"
    )
    if search_buffer_m > 0:
        LOGGER.info(
            f"[polygons-filter] Using search buffer of {search_buffer_m}m ({search_buffer_m/1000:.2f} km) to expand flightline search area"
        )
        flightline_bbox = box(
            bounds.left - search_buffer_m,
            bounds.bottom - search_buffer_m,
            bounds.right + search_buffer_m,
            bounds.top + search_buffer_m
        )
        LOGGER.info(
            f"[polygons-filter] Buffered flightline bbox: "
            f"({bounds.left - search_buffer_m:.2f}, {bounds.bottom - search_buffer_m:.2f}, "
            f"{bounds.right + search_buffer_m:.2f}, {bounds.top + search_buffer_m:.2f}) "
            f"[CRS: {dataset_crs}]"
        )
    else:
        flightline_bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        LOGGER.info(
            f"[polygons-filter] Flightline bbox: "
            f"({bounds.left:.2f}, {bounds.bottom:.2f}, {bounds.right:.2f}, {bounds.top:.2f}) "
            f"[CRS: {dataset_crs}]"
        )
    
    # Calculate overlap for each polygon
    overlap_stats = []
    filtered_polygons = []
    
    # Log some sample polygon bounds for debugging
    # Use buffered bounds for comparison if buffer is applied
    comparison_bounds = {
        'left': bounds.left - search_buffer_m if search_buffer_m > 0 else bounds.left,
        'bottom': bounds.bottom - search_buffer_m if search_buffer_m > 0 else bounds.bottom,
        'right': bounds.right + search_buffer_m if search_buffer_m > 0 else bounds.right,
        'top': bounds.top + search_buffer_m if search_buffer_m > 0 else bounds.top,
    }
    
    if len(polygons) > 0:
        sample_poly_bounds = polygons.total_bounds
        LOGGER.info(
            f"[polygons-filter] Total polygon bounds (after site filter): "
            f"({sample_poly_bounds[0]:.2f}, {sample_poly_bounds[1]:.2f}, "
            f"{sample_poly_bounds[2]:.2f}, {sample_poly_bounds[3]:.2f})"
        )
        if search_buffer_m > 0:
            LOGGER.info(
                f"[polygons-filter] Flightline bbox bounds (with {search_buffer_m}m buffer): "
                f"({comparison_bounds['left']:.2f}, {comparison_bounds['bottom']:.2f}, "
                f"{comparison_bounds['right']:.2f}, {comparison_bounds['top']:.2f})"
            )
        else:
            LOGGER.info(
                f"[polygons-filter] Flightline bbox bounds: "
                f"({comparison_bounds['left']:.2f}, {comparison_bounds['bottom']:.2f}, "
                f"{comparison_bounds['right']:.2f}, {comparison_bounds['top']:.2f})"
            )
        # Check if total bounds overlap (using buffered bounds if buffer is applied)
        total_bounds_overlap = (
            sample_poly_bounds[0] < comparison_bounds['right'] and
            sample_poly_bounds[2] > comparison_bounds['left'] and
            sample_poly_bounds[1] < comparison_bounds['top'] and
            sample_poly_bounds[3] > comparison_bounds['bottom']
        )
        if not total_bounds_overlap:
            if search_buffer_m > 0:
                LOGGER.warning(
                    f"[polygons-filter] ⚠️  Total polygon bounds do NOT overlap with buffered flightline bbox! "
                    f"Buffer: {search_buffer_m}m. Polygons may still be too far away."
                )
            else:
                LOGGER.warning(
                    "[polygons-filter] ⚠️  Total polygon bounds do NOT overlap with flightline bbox! "
                    "This suggests polygons are from different locations within the site. "
                    "Consider using polygon_search_buffer_m to expand the search area."
                )
        else:
            LOGGER.info(
                "[polygons-filter] ✅ Total polygon bounds overlap with flightline bbox"
                + (f" (with {search_buffer_m}m buffer)" if search_buffer_m > 0 else "")
            )
    
    # Debug: Check if flightline_bbox is valid
    if not flightline_bbox.is_valid:
        LOGGER.warning(
            "[polygons-filter] ⚠️  Flightline bbox is invalid! Attempting to fix..."
        )
        flightline_bbox = flightline_bbox.buffer(0)  # Fix invalid geometry
    
    intersection_count = 0
    for idx, row in polygons.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        
        # Ensure geometry is valid
        if not geom.is_valid:
            LOGGER.debug(f"[polygons-filter] Polygon {idx} is invalid, attempting to fix...")
            geom = geom.buffer(0)  # Fix invalid geometry
        
        # Calculate intersection with flightline
        try:
            intersection = geom.intersection(flightline_bbox)
        except Exception as e:
            LOGGER.warning(
                f"[polygons-filter] Failed to calculate intersection for polygon {idx}: {e}"
            )
            overlap_stats.append({
                "polygon_index": idx,
                "overlap_pct": 0.0,
                "intersects": False,
            })
            continue
        
        if intersection.is_empty or intersection.area == 0:
            overlap_pct = 0.0
        else:
            # Calculate percentage of polygon that overlaps flightline
            polygon_area = geom.area
            if polygon_area > 0:
                overlap_pct = (intersection.area / polygon_area) * 100.0
            else:
                overlap_pct = 0.0
        
        intersects = not (intersection.is_empty or intersection.area == 0)
        if intersects:
            intersection_count += 1
        
        overlap_stats.append({
            "polygon_index": idx,
            "overlap_pct": overlap_pct,
            "intersects": intersects,
        })
        
        # Filter by overlap threshold
        # Also require that the intersection is not empty (polygon actually touches flightline)
        if overlap_pct >= min_overlap_pct * 100.0 and intersects:
            filtered_polygons.append(row)
    
    LOGGER.info(
        f"[polygons-filter] Intersection check complete: {intersection_count} polygons intersect "
        f"out of {len(polygons)} total"
    )
    
    # Log details about first few polygons that don't intersect for debugging
    non_intersecting = [s for s in overlap_stats if not s["intersects"]]
    if len(non_intersecting) > 0 and len(filtered_polygons) == 0:
        LOGGER.warning(
            f"[polygons-filter] ⚠️  None of the {len(polygons)} polygons intersect the flightline. "
            f"Sample non-intersecting polygon bounds:"
        )
        # Show bounds of first few polygons
        for i, (idx, row) in enumerate(polygons.head(3).iterrows()):
            if row.geometry is not None and not row.geometry.is_empty:
                poly_bounds = row.geometry.bounds
                LOGGER.warning(
                    f"  Polygon {i+1}: bounds=({poly_bounds[0]:.2f}, {poly_bounds[1]:.2f}, "
                    f"{poly_bounds[2]:.2f}, {poly_bounds[3]:.2f})"
                )
    
    overlap_df = pd.DataFrame(overlap_stats)
    
    LOGGER.info(
        f"[polygons-filter] Overlap statistics:\n"
        f"  Total polygons: {len(polygons)}\n"
        f"  Polygons intersecting flightline: {overlap_df['intersects'].sum()}\n"
        f"  Polygons with >= {min_overlap_pct*100:.1f}% overlap: {len(filtered_polygons)}\n"
        f"  Average overlap: {overlap_df['overlap_pct'].mean():.2f}%"
    )
    
    if not filtered_polygons:
        # If no polygons intersect but total bounds overlap, try using a buffer approach
        # This handles cases where polygons are very close but don't intersect due to precision issues
        if len(polygons) > 0:
            sample_poly_bounds = polygons.total_bounds
            total_bounds_overlap = (
                sample_poly_bounds[0] < bounds.right and
                sample_poly_bounds[2] > bounds.left and
                sample_poly_bounds[1] < bounds.top and
                sample_poly_bounds[3] > bounds.bottom
            )
            
            if total_bounds_overlap:
                LOGGER.warning(
                    "[polygons-filter] ⚠️  No polygons intersect, but total bounds overlap! "
                    "This might be a precision issue. Trying with buffered bbox..."
                )
                # Try with a small buffer (1 meter) to handle precision issues
                buffered_bbox = flightline_bbox.buffer(1.0)
                filtered_polygons = []
                for idx, row in polygons.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    if not geom.is_valid:
                        geom = geom.buffer(0)
                    try:
                        intersection = geom.intersection(buffered_bbox)
                        if not intersection.is_empty and intersection.area > 0:
                            polygon_area = geom.area
                            if polygon_area > 0:
                                overlap_pct = (intersection.area / polygon_area) * 100.0
                                if overlap_pct >= min_overlap_pct * 100.0:
                                    filtered_polygons.append(row)
                    except Exception:
                        continue
                
                if filtered_polygons:
                    LOGGER.info(
                        f"[polygons-filter] ✅ Found {len(filtered_polygons)} polygons with buffered bbox"
                    )
                else:
                    LOGGER.error(
                        "[polygons-filter] ❌ Still no polygons found even with buffered bbox"
                    )
        
        if not filtered_polygons:
            # Provide detailed error message
            # Use buffered bounds in error message if buffer was applied
            error_bounds = comparison_bounds if search_buffer_m > 0 else {
                'left': bounds.left,
                'bottom': bounds.bottom,
                'right': bounds.right,
                'top': bounds.top,
            }
            error_details = [
                f"No polygons found with >= {min_overlap_pct*100:.1f}% overlap with flightline.",
                "",
            ]
            if search_buffer_m > 0:
                error_details.append(
                    f"Flightline bounds (with {search_buffer_m}m buffer): "
                    f"({error_bounds['left']:.2f}, {error_bounds['bottom']:.2f}, "
                    f"{error_bounds['right']:.2f}, {error_bounds['top']:.2f})"
                )
            else:
                error_details.append(
                    f"Flightline bounds: ({error_bounds['left']:.2f}, {error_bounds['bottom']:.2f}, "
                    f"{error_bounds['right']:.2f}, {error_bounds['top']:.2f})"
                )
            if len(polygons) > 0:
                poly_bounds = polygons.total_bounds
                error_details.append(
                    f"Polygon bounds: ({poly_bounds[0]:.2f}, {poly_bounds[1]:.2f}, "
                    f"{poly_bounds[2]:.2f}, {poly_bounds[3]:.2f})"
                )
                error_details.append(f"Total polygons checked: {len(polygons)}")
                error_details.append(f"Polygons intersecting: {intersection_count}")
                
                # Calculate distance between centers
                flightline_center_x = (bounds.left + bounds.right) / 2
                flightline_center_y = (bounds.bottom + bounds.top) / 2
                polygon_center_x = (poly_bounds[0] + poly_bounds[2]) / 2
                polygon_center_y = (poly_bounds[1] + poly_bounds[3]) / 2
                distance_x = abs(polygon_center_x - flightline_center_x)
                distance_y = abs(polygon_center_y - flightline_center_y)
                distance_total = (distance_x**2 + distance_y**2)**0.5
                
                error_details.append("")
                error_details.append(
                    f"Distance between centers: {distance_total/1000:.2f} km "
                    f"(X: {distance_x/1000:.2f} km, Y: {distance_y/1000:.2f} km)"
                )
                error_details.append("")
                error_details.append(
                    "⚠️  The polygons are from a different location within the site. "
                    "They do not overlap with this flightline's spatial extent."
                )
                error_details.append(
                    "This is expected if your polygon file contains polygons from multiple "
                    "flightlines or locations within the same site."
                )
                error_details.append("")
                error_details.append(
                    "Possible solutions:"
                )
                error_details.append(
                    "  1. Use a polygon file that contains polygons for this specific flightline"
                )
                error_details.append(
                    "  2. Process a different flightline that overlaps with these polygons"
                )
                error_details.append(
                    "  3. Filter your polygon file to only include polygons that overlap "
                    "with this flightline's bounds before running the pipeline"
                )
                if search_buffer_m > 0:
                    error_details.append("")
                    required_buffer = int(distance_total) + 1000  # Add 1km safety margin
                    error_details.append(
                        f"  4. Increase polygon_search_buffer_m (currently {search_buffer_m}m = {search_buffer_m/1000:.2f} km) "
                        f"to at least {required_buffer}m ({required_buffer/1000:.2f} km) to include polygons "
                        f"{distance_total/1000:.2f} km away"
                    )
                    error_details.append(
                        f"     Example: polygon_search_buffer_m={required_buffer}.0"
                    )
                else:
                    required_buffer = int(distance_total) + 1000
                    error_details.append("")
                    error_details.append(
                        f"  4. Use polygon_search_buffer_m={required_buffer}.0 ({required_buffer/1000:.2f} km) "
                        f"to include polygons {distance_total/1000:.2f} km away"
                    )
            raise ValueError("\n".join(error_details))
    
    # Create filtered GeoDataFrame
    # Save in the raster CRS to avoid transformation issues later
    filtered_gdf = geopandas.GeoDataFrame(filtered_polygons, crs=polygons.crs)
    
    # Ensure filtered polygons are in the same CRS as the raster before saving
    if dataset_crs is not None and filtered_gdf.crs != dataset_crs:
        LOGGER.info(
            f"[polygons-filter] Transforming filtered polygons to raster CRS {dataset_crs} for saving"
        )
        filtered_gdf = filtered_gdf.to_crs(dataset_crs)
    
    # Log filtered polygon bounds for debugging
    filtered_bounds = filtered_gdf.total_bounds
    LOGGER.info(
        f"[polygons-filter] Filtered polygon bounds: ({filtered_bounds[0]:.2f}, {filtered_bounds[1]:.2f}, "
        f"{filtered_bounds[2]:.2f}, {filtered_bounds[3]:.2f}) [minx, miny, maxx, maxy]"
    )
    LOGGER.info(
        f"[polygons-filter] Flightline bounds: ({bounds.left:.2f}, {bounds.bottom:.2f}, "
        f"{bounds.right:.2f}, {bounds.top:.2f}) [minx, miny, maxx, maxy]"
    )
    
    # Verify bounds actually overlap
    bounds_overlap = (
        filtered_bounds[0] < bounds.right and
        filtered_bounds[2] > bounds.left and
        filtered_bounds[1] < bounds.top and
        filtered_bounds[3] > bounds.bottom
    )
    if not bounds_overlap:
        # This is a critical error - filtered polygons should always overlap
        error_msg = (
            f"Filtered polygon bounds do NOT overlap with flightline bounds after filtering! "
            f"This indicates a serious issue with the overlap calculation or CRS transformation.\n"
            f"Filtered polygon bounds: ({filtered_bounds[0]:.2f}, {filtered_bounds[1]:.2f}, "
            f"{filtered_bounds[2]:.2f}, {filtered_bounds[3]:.2f})\n"
            f"Flightline bounds: ({bounds.left:.2f}, {bounds.bottom:.2f}, "
            f"{bounds.right:.2f}, {bounds.top:.2f})\n"
            f"Filtered polygon CRS: {filtered_gdf.crs}\n"
            f"Raster CRS: {dataset_crs}\n"
            f"Number of filtered polygons: {len(filtered_gdf)}"
        )
        LOGGER.error(f"[polygons-filter] ❌ {error_msg}")
        raise ValueError(error_msg)
    else:
        LOGGER.info(
            "[polygons-filter] ✅ Filtered polygon bounds overlap with flightline bounds"
        )
    
    # Save filtered GeoJSON
    if output_path is None:
        output_path = (
            flight_paths.flight_dir
            / f"{flight_paths.flight_id}_filtered_polygons.geojson"
        )
    else:
        output_path = Path(output_path)
    
    filtered_gdf.to_file(output_path, driver="GeoJSON")
    LOGGER.info(
        f"[polygons-filter] ✅ Saved {len(filtered_gdf)} filtered polygons → {output_path} "
        f"(CRS: {filtered_gdf.crs})"
    )
    
    return output_path, overlap_df


def visualize_polygons_on_envi(
    flight_paths: FlightlinePaths,
    polygons_path: str | Path,
    reference_product: str = "brdfandtopo_corrected_envi",
    output_path: Path | str | None = None,
    *,
    rgb_bands: tuple[float, float, float] | None = None,
    dpi: int = 300,
    validate_coords: bool = True,
) -> Path:
    """Create a visualization showing polygons overlaid on ENVI RGB image.
    
    Parameters
    ----------
    flight_paths : FlightlinePaths
        Flightline paths object
    polygons_path : str | Path
        Path to GeoJSON file containing polygons
    reference_product : str, default "brdfandtopo_corrected_envi"
        Reference product to use for RGB visualization
    output_path : Path | str | None, optional
        Path to save visualization PNG. If None, saves to flightline directory.
    rgb_bands : tuple[float, float, float] | None, optional
        Wavelengths (nm) for R, G, B bands. If None, uses default (660, 560, 490).
    dpi : int, default 300
        Resolution for output image
    
    Returns
    -------
    Path
        Path to saved visualization PNG
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    geopandas = require_geopandas()
    rasterio = require_rasterio()
    from cross_sensor_cal.envi import hdr_to_dict
    from cross_sensor_cal.header_utils import wavelengths_from_hdr
    
    polygons_path = Path(polygons_path)
    if not polygons_path.exists():
        raise FileNotFoundError(f"Polygon file not found: {polygons_path}")
    
    img_path, hdr_path = _resolve_reference_raster(flight_paths, reference_product)
    if not img_path.exists() or not hdr_path.exists():
        raise FileNotFoundError(
            f"Reference ENVI files not found: {img_path} / {hdr_path}"
        )
    
    # Read polygons
    polygons = geopandas.read_file(polygons_path)
    if polygons.empty:
        raise ValueError(f"No polygons found in {polygons_path}")
    
    # Validate coordinate matching (with more lenient tolerance for large flightlines)
    if validate_coords:
        is_valid, message = validate_coordinate_match(polygons, img_path, hdr_path, tolerance_m=50000.0)
        if not is_valid:
            LOGGER.warning(f"[polygons-viz] ⚠️  Coordinate validation warning: {message}")
            LOGGER.warning("[polygons-viz] Continuing anyway - visualization will show actual overlap.")
        else:
            LOGGER.info(f"[polygons-viz] ✅ Coordinate validation passed: {message}")
    
    # Read ENVI header for wavelengths
    header = hdr_to_dict(hdr_path)
    wavelengths = wavelengths_from_hdr(header)
    
    # Determine RGB bands
    if rgb_bands is None:
        rgb_bands = (660.0, 560.0, 490.0)  # Default: red, green, blue
    
    # Find closest bands to target wavelengths
    def find_closest_band(target_wl: float) -> int:
        if len(wavelengths) == 0:
            return 0
        return int(np.argmin(np.abs(wavelengths - target_wl)))
    
    r_band = find_closest_band(rgb_bands[0])
    g_band = find_closest_band(rgb_bands[1])
    b_band = find_closest_band(rgb_bands[2])
    
    LOGGER.info(
        f"[polygons-viz] Using bands {r_band}, {g_band}, {b_band} "
        f"(wavelengths {wavelengths[r_band]:.1f}, {wavelengths[g_band]:.1f}, {wavelengths[b_band]:.1f} nm)"
    )
    
    # Read ENVI cube (just the RGB bands)
    with rasterio.open(img_path) as src:
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        
        # Read only RGB bands
        r_data = src.read(r_band + 1)  # 1-indexed
        g_data = src.read(g_band + 1)
        b_data = src.read(b_band + 1)
    
    # Normalize RGB for display (0-1 range)
    def normalize_band(data: np.ndarray, percentile: tuple[float, float] = (2, 98)) -> np.ndarray:
        valid_data = data[data > 0]
        if len(valid_data) == 0:
            # If no valid data, return zeros
            return np.zeros_like(data, dtype=float)
        # Calculate percentiles separately to ensure we get scalars
        p_low = float(np.percentile(valid_data, percentile[0]))
        p_high = float(np.percentile(valid_data, percentile[1]))
        if p_high == p_low:
            # Avoid division by zero
            return np.ones_like(data, dtype=float) if p_high > 0 else np.zeros_like(data, dtype=float)
        normalized = np.clip((data - p_low) / (p_high - p_low), 0, 1)
        return normalized
    
    r_norm = normalize_band(r_data)
    g_norm = normalize_band(g_data)
    b_norm = normalize_band(b_data)
    
    # Stack into RGB image
    rgb_image = np.dstack([r_norm, g_norm, b_norm])
    
    # Align polygon CRS with raster CRS
    if polygons.crs is None and crs is not None:
        polygons = polygons.set_crs(crs)
    elif crs is not None and polygons.crs != crs:
        LOGGER.info(f"[polygons-viz] Transforming polygons from {polygons.crs} to {crs}")
        polygons = polygons.to_crs(crs)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    
    # Show RGB image using imshow with extent
    from rasterio.transform import xy
    minx, maxy = xy(transform, 0, 0, offset="ul")
    maxx, miny = xy(transform, height, width, offset="lr")
    extent = [minx, maxx, miny, maxy]
    
    ax.imshow(
        rgb_image,
        extent=extent,
        interpolation='bilinear',
        origin='upper',
    )
    
    # Overlay polygons with colors
    num_polygons = len(polygons)
    colors = plt.cm.tab20(np.linspace(0, 1, num_polygons))
    
    for idx, (_, row) in enumerate(polygons.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        
        # Get polygon attributes for label
        label = f"Polygon {idx + 1}"
        if "polygon_id" in row:
            label = f"ID: {row['polygon_id']}"
        elif "id" in row:
            label = f"ID: {row['id']}"
        
        # Plot polygon boundary
        geopandas.GeoSeries([geom], crs=polygons.crs).plot(
            ax=ax,
            facecolor='none',
            edgecolor=colors[idx],
            linewidth=2,
            alpha=0.8,
            label=label,
        )
        
        # Fill polygon with semi-transparent color
        geopandas.GeoSeries([geom], crs=polygons.crs).plot(
            ax=ax,
            facecolor=colors[idx],
            edgecolor='none',
            alpha=0.3,
        )
    
    ax.set_title(
        f"Polygons overlaid on {reference_product}\n"
        f"{flight_paths.flight_id}\n"
        f"{num_polygons} polygons",
        fontsize=14,
        fontweight='bold',
    )
    ax.set_xlabel("Easting (m)", fontsize=12)
    ax.set_ylabel("Northing (m)", fontsize=12)
    
    # Add legend if not too many polygons
    if num_polygons <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = (
            flight_paths.flight_dir
            / f"{flight_paths.flight_id}_polygons_overlay.png"
        )
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"[polygons-viz] ✅ Saved visualization → {output_path}")
    return output_path


def _quote_path(path: Path) -> str:
    """Escape a filesystem path for embedding inside DuckDB SQL strings."""

    return str(path).replace("'", "''")


def _quote_identifier(name: str) -> str:
    """Return a double-quoted SQL identifier."""

    return '"' + name.replace('"', '""') + '"'


def _sanitise_alias(name: str) -> str:
    """Return a DuckDB-safe identifier derived from ``name``."""

    alias = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not alias:
        alias = "tbl"
    if alias[0].isdigit():
        alias = f"t_{alias}"
    return alias


def _resolve_reference_raster(
    flight_paths: FlightlinePaths, reference_product: str
) -> tuple[Path, Path]:
    """Return ``(img, hdr)`` paths for the requested reference product."""

    reference_product = reference_product.strip()
    if reference_product == "brdfandtopo_corrected_envi":
        return flight_paths.corrected_img, flight_paths.corrected_hdr
    if reference_product == "envi":
        return flight_paths.envi_img, flight_paths.envi_hdr
    if reference_product.endswith("_envi"):
        sensor_name = reference_product[: -len("_envi")]
        sensor_paths = flight_paths.sensor_product(sensor_name)
        return sensor_paths.img, sensor_paths.hdr
    raise ValueError(
        "Unsupported reference_product value: "
        f"{reference_product!r}. Expected one of 'envi', "
        "'brdfandtopo_corrected_envi', or '<sensor>_envi'."
    )


def _available_product_parquets(flight_paths: FlightlinePaths) -> Dict[str, Path]:
    """Return known per-product parquet paths for ``flight_paths``.
    
    This function discovers ALL parquet files in the flightline directory,
    including originals, corrected, resampled (Landsat, MicaSense, etc.),
    and undarkened parquets.
    """
    flight_dir = flight_paths.flight_dir
    
    # Start with known paths from FlightlinePaths
    products: Dict[str, Path] = {}
    
    # Add original ENVI parquet if it exists
    if flight_paths.envi_parquet.exists():
        products["envi"] = flight_paths.envi_parquet
    
    # Add corrected parquet if it exists
    if flight_paths.corrected_parquet.exists():
        products["brdfandtopo_corrected_envi"] = flight_paths.corrected_parquet
    
    # Add sensor products from FlightlinePaths
    for sensor_name, sensor_paths in flight_paths.sensor_products.items():
        if sensor_paths.parquet.exists():
            products[f"{sensor_name}_envi"] = sensor_paths.parquet
    
    # Discover ALL other parquet files in the directory
    # This includes resampled files, undarkened files, etc.
    all_parquets = sorted(flight_dir.glob("*.parquet"))
    
    # Exclude merged and QA files
    exclude_patterns = {
        "_merged_pixel_extraction.parquet",
        "_qa_metrics.parquet",
        "_polygon_pixel_index.parquet",
        "_polygons.parquet",
        "_polygons_merged",
    }
    
    for parquet_path in all_parquets:
        # Skip if already in products
        if parquet_path in products.values():
            continue
        
        # Skip excluded patterns
        if any(ex in parquet_path.name for ex in exclude_patterns):
            continue
        
        # Determine product name from filename
        stem = parquet_path.stem
        
        # Handle undarkened files
        if "_undarkened_envi" in stem:
            # Extract sensor name: <flight>_<sensor>_undarkened_envi
            parts = stem.split("_undarkened_envi")[0].split("_")
            if len(parts) >= 2:
                sensor = parts[-1]  # Last part before _undarkened
                product_name = f"{sensor}_undarkened_envi"
            else:
                product_name = f"{stem}_undarkened_envi"
        # Handle regular sensor files: <flight>_<sensor>_envi
        elif stem.endswith("_envi"):
            # Check if it's a sensor file (has underscore before _envi)
            if "_" in stem[:-5]:  # stem without "_envi"
                parts = stem.rsplit("_envi", 1)[0].split("_")
                if len(parts) >= 2:
                    sensor = parts[-1]  # Last part before _envi
                    product_name = f"{sensor}_envi"
                else:
                    product_name = stem
            else:
                # Already handled as "envi" above
                continue
        else:
            # Unknown format, use stem as product name
            product_name = stem
        
        # Only add if not already present
        if product_name not in products:
            products[product_name] = parquet_path
    
    return products


def _describe_parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    """Return column names for a parquet file using DuckDB DESCRIBE."""

    sql = f"DESCRIBE SELECT * FROM read_parquet('{_quote_path(path)}')"
    return [row[0] for row in con.execute(sql).fetchall()]


def _write_dataframe_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist a DataFrame to parquet via DuckDB to avoid pyarrow dependency."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("df_view", df)
        con.execute(
            "COPY (SELECT * FROM df_view) TO '"
            + _quote_path(path)
            + "' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        con.close()


def build_polygon_pixel_index(
    flight_paths: FlightlinePaths,
    polygons_path: str | Path,
    output_path: Path | str | None = None,
    *,
    reference_product: str = "brdfandtopo_corrected_envi",
    overwrite: bool = False,
    all_touched: bool = False,
) -> Path:
    """Create a pixel→polygon lookup table for ``flight_paths``.

    The resulting Parquet table includes one row per pixel that intersects any
    polygon.  Columns comprise the canonical pixel identifiers plus polygon
    attributes and geometry stored as WKB.
    """

    polygons_path = Path(polygons_path)
    if not polygons_path.exists():
        raise FileNotFoundError(polygons_path)

    if output_path is None:
        output_path = (
            flight_paths.flight_dir
            / f"{flight_paths.flight_id}_polygon_pixel_index.parquet"
        )
    else:
        output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        LOGGER.info("[polygons-index] Reusing existing index → %s", output_path)
        return output_path

    geopandas = require_geopandas()
    rasterio = require_rasterio()
    from rasterio.features import rasterize
    from rasterio.transform import xy

    img_path, hdr_path = _resolve_reference_raster(flight_paths, reference_product)
    if not img_path.exists():
        raise FileNotFoundError(
            f"Reference image for {reference_product!r} not found: {img_path}"
        )
    if not hdr_path.exists():
        raise FileNotFoundError(
            f"Reference header for {reference_product!r} not found: {hdr_path}"
        )

    polygons = geopandas.read_file(polygons_path)
    if polygons.empty:
        raise ValueError(f"No polygons were found in {polygons_path}")
    
    LOGGER.info(
        f"[polygons-index] Loaded {len(polygons)} polygons from {polygons_path.name} "
        f"(CRS: {polygons.crs})"
    )

    # Validate coordinate matching before processing (with more lenient tolerance)
    is_valid, message = validate_coordinate_match(polygons, img_path, hdr_path, tolerance_m=50000.0)
    if not is_valid:
        LOGGER.warning(f"[polygons-index] ⚠️  Coordinate validation warning: {message}")
        LOGGER.warning(
            "[polygons-index] Continuing anyway - rasterization will determine if polygons actually intersect pixels. "
            "If no pixels intersect, the function will raise a clear error."
        )
        # Don't raise error, just warn - let rasterization determine actual overlap
    else:
        LOGGER.info(f"[polygons-index] ✅ Coordinate validation passed: {message}")

    with rasterio.open(img_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        dataset_crs = src.crs
        crs_epsg = dataset_crs.to_epsg() if dataset_crs else None
    
    # If transform is None, try to build it from ENVI header map info
    if transform is None or not hasattr(transform, '__call__'):
        LOGGER.warning(
            "[polygons-index] Raster has no transform. Attempting to build from ENVI header..."
        )
        try:
            from cross_sensor_cal.envi import _parse_envi_header_tolerant
            from cross_sensor_cal.io.neon import _map_info_core
            
            header = _parse_envi_header_tolerant(hdr_path)
            map_info = header.get("map info")
            
            if map_info:
                if isinstance(map_info, str):
                    # Parse map info string
                    map_info_str = map_info.strip()
                    if map_info_str.startswith("{"):
                        map_info_str = map_info_str[1:]
                    if map_info_str.endswith("}"):
                        map_info_str = map_info_str[:-1]
                    map_info_list = [item.strip() for item in map_info_str.split(",")]
                elif isinstance(map_info, list):
                    map_info_list = [str(item).strip() for item in map_info]
                else:
                    map_info_list = []
                
                if len(map_info_list) >= 7:
                    ref_x, ref_y, ref_easting, ref_northing, pixel_x, pixel_y = _map_info_core(
                        map_info_list
                    )
                    # Build transform: (ulx, pixel_width, 0, uly, 0, pixel_height)
                    # pixel_y is typically negative for north-up images
                    ulx = ref_easting - pixel_x * (ref_x - 0.5)
                    uly = ref_northing + abs(pixel_y) * (ref_y - 0.5)
                    yres = -abs(pixel_y)  # Negative for north-up
                    transform = rasterio.Affine(pixel_x, 0.0, ulx, 0.0, yres, uly)
                    LOGGER.info(
                        "[polygons-index] ✅ Built transform from map info: ulx=%.2f, uly=%.2f, xres=%.2f, yres=%.2f",
                        ulx, uly, pixel_x, yres
                    )
                else:
                    raise ValueError(f"Map info has insufficient elements: {len(map_info_list)}")
            else:
                raise ValueError("No map info found in ENVI header")
        except Exception as e:
            LOGGER.error(
                "[polygons-index] ❌ Failed to build transform from ENVI header: %s", e
            )
            raise ValueError(
                f"Invalid transform from raster {img_path} and could not build from ENVI header. "
                f"Transform is None or not callable. Error: {e}"
            ) from e
    
    # If CRS is not in the raster, try to extract it from the ENVI header
    if dataset_crs is None:
        LOGGER.warning(
            "[polygons-index] ENVI file has no CRS. Attempting to extract from header..."
        )
        try:
            from cross_sensor_cal.polygon_extraction import get_crs_from_hdr
            dataset_crs = get_crs_from_hdr(hdr_path)
            if dataset_crs is not None:
                crs_epsg = dataset_crs.to_epsg()
                LOGGER.info(
                    "[polygons-index] ✅ Extracted CRS from header: %s (EPSG: %s)",
                    dataset_crs, crs_epsg
                )
            else:
                LOGGER.warning(
                    "[polygons-index] ⚠️  Could not extract CRS from header. "
                    "CRS alignment will be skipped (may cause intersection issues)."
                )
        except Exception as e:
            LOGGER.warning(
                "[polygons-index] ⚠️  Failed to extract CRS from header: %s", e
            )
    
    # Normalize CRS to EPSG if possible (handles custom PROJCS definitions)
    if dataset_crs is not None:
        try:
            epsg_code = dataset_crs.to_epsg()
            if epsg_code is not None:
                LOGGER.info(
                    "[polygons-index] Normalizing dataset CRS to EPSG:%d", epsg_code
                )
                dataset_crs = rasterio.crs.CRS.from_epsg(epsg_code)
                crs_epsg = epsg_code
        except Exception:
            # If normalization fails, use the CRS as-is
            pass
    
    # Normalize polygon CRS to EPSG if possible
    if polygons.crs is not None:
        try:
            poly_epsg = polygons.crs.to_epsg()
            if poly_epsg is not None:
                LOGGER.info(
                    "[polygons-index] Normalizing polygon CRS to EPSG:%d", poly_epsg
                )
                polygons = polygons.set_crs(rasterio.crs.CRS.from_epsg(poly_epsg))
        except Exception:
            # If normalization fails, use the CRS as-is
            pass
    
    # Log CRS information for debugging
    LOGGER.info(
        "[polygons-index] Dataset CRS: %s, Polygon CRS: %s",
        dataset_crs if dataset_crs else "None",
        polygons.crs if polygons.crs is not None else "None",
    )

    if polygons.crs is None and dataset_crs is not None:
        LOGGER.warning(
            "[polygons-index] Polygon source has no CRS; assuming %s", dataset_crs
        )
        polygons = polygons.set_crs(dataset_crs)
    elif dataset_crs is not None and polygons.crs != dataset_crs:
        LOGGER.info(
            "[polygons-index] Transforming polygons from %s to %s",
            polygons.crs, dataset_crs
        )
        polygons = polygons.to_crs(dataset_crs)
    elif polygons.crs is None and dataset_crs is None:
        LOGGER.error(
            "[polygons-index] ❌ Both polygon and dataset have no CRS! "
            "Cannot perform spatial intersection. Please ensure your ENVI file has "
            "CRS information in the header (coordinate system string or map info)."
        )
        raise ValueError(
            "Both polygon and ENVI dataset have no CRS. "
            "Cannot perform spatial intersection without matching coordinate systems."
        )

    polygons = polygons.reset_index(drop=True).copy()
    if "polygon_id" in polygons and polygons["polygon_id"].is_unique:
        polygon_ids = polygons["polygon_id"].astype("int64", copy=False)
    else:
        polygon_ids = pd.Series(
            np.arange(1, len(polygons) + 1, dtype="int64"), name="polygon_id"
        )
        polygons["polygon_id"] = polygon_ids

    # Prepare rasterisation shapes (skip empties)
    shapes = []
    for geom, pid in zip(polygons.geometry, polygons["polygon_id"]):
        if geom is None or geom.is_empty:
            continue
        shapes.append((geom, int(pid)))

    if not shapes:
        raise ValueError("All polygons were empty; nothing to index")

    LOGGER.info(
        "[polygons-index] Rasterising %s polygons against %s", len(shapes), img_path
    )
    
    # Log polygon bounds for debugging
    if len(polygons) > 0:
        poly_bounds = polygons.total_bounds
        LOGGER.info(
            "[polygons-index] Polygon bounds: (%.2f, %.2f, %.2f, %.2f) [minx, miny, maxx, maxy]",
            poly_bounds[0], poly_bounds[1], poly_bounds[2], poly_bounds[3]
        )
        LOGGER.info(
            "[polygons-index] Polygon CRS: %s", polygons.crs
        )
        # Calculate raster bounds properly using transform
        from rasterio.transform import xy
        minx_raster, maxy_raster = xy(transform, 0, 0, offset="ul")
        maxx_raster, miny_raster = xy(transform, height, width, offset="lr")
        LOGGER.info(
            "[polygons-index] Raster bounds: (%.2f, %.2f, %.2f, %.2f) [minx, miny, maxx, maxy]",
            minx_raster, miny_raster, maxx_raster, maxy_raster
        )
        LOGGER.info(
            "[polygons-index] Raster CRS: %s", dataset_crs
        )
        # Check if bounds overlap
        bounds_overlap = (
            poly_bounds[0] < maxx_raster and
            poly_bounds[2] > minx_raster and
            poly_bounds[1] < maxy_raster and
            poly_bounds[3] > miny_raster
        )
        if not bounds_overlap:
            LOGGER.error(
                "[polygons-index] ❌ Polygon bounds do NOT overlap with raster bounds! "
                "This indicates the filtered polygons may not be correctly aligned. "
                "Distance between centers: X=%.1fm, Y=%.1fm",
                abs((poly_bounds[0] + poly_bounds[2]) / 2 - (minx_raster + maxx_raster) / 2),
                abs((poly_bounds[1] + poly_bounds[3]) / 2 - (miny_raster + maxy_raster) / 2)
            )
            # Don't raise error yet - let rasterization try, but log the issue
        else:
            LOGGER.info(
                "[polygons-index] ✅ Polygon bounds overlap with raster bounds"
            )

    polygon_grid = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=all_touched,
    )

    mask = polygon_grid > 0
    if not mask.any():
        # Provide detailed error message with bounds
        if len(polygons) > 0:
            poly_bounds = polygons.total_bounds
            from rasterio.transform import xy
            minx_raster, miny_raster = xy(transform, 0, 0, offset="ul")
            maxx_raster, maxy_raster = xy(transform, height, width, offset="lr")
            raise ValueError(
                f"No pixels intersected the supplied polygons.\n"
                f"Polygon bounds: ({poly_bounds[0]:.2f}, {poly_bounds[1]:.2f}, "
                f"{poly_bounds[2]:.2f}, {poly_bounds[3]:.2f})\n"
                f"Raster bounds: ({minx_raster:.2f}, {miny_raster:.2f}, "
                f"{maxx_raster:.2f}, {maxy_raster:.2f})\n"
                f"Polygon CRS: {polygons.crs}\n"
                f"Raster CRS: {dataset_crs}\n"
                f"This usually indicates a CRS mismatch or polygon outside raster extent."
            )
        raise ValueError("No pixels intersected the supplied polygons")

    rows, cols = np.nonzero(mask)
    polygon_ids = polygon_grid[rows, cols].astype("int64", copy=False)
    xs, ys = xy(transform, rows, cols, offset="center")

    df = pd.DataFrame(
        {
            "pixel_id": rows.astype("int64") * width + cols.astype("int64"),
            "row": rows.astype("int32"),
            "col": cols.astype("int32"),
            "x": np.asarray(xs, dtype="float64"),
            "y": np.asarray(ys, dtype="float64"),
            "polygon_id": polygon_ids,
        }
    )
    df["flight_id"] = flight_paths.flight_id
    df["polygon_source"] = str(polygons_path)
    df["reference_product"] = reference_product
    if dataset_crs is not None:
        try:
            df["raster_crs"] = dataset_crs.to_string()
        except Exception:  # pragma: no cover - defensive
            df["raster_crs"] = str(dataset_crs)
    if crs_epsg is not None:
        df["epsg"] = pd.Series(crs_epsg, index=df.index, dtype="Int64")

    df = ensure_coord_columns(df, transform=transform, crs_epsg=crs_epsg or 0)

    attribute_columns = [
        col for col in polygons.columns if col != polygons.geometry.name
    ]
    polygon_attrs = polygons[attribute_columns].copy()
    polygon_attrs["polygon_geometry_wkb"] = polygons.geometry.to_wkb()
    df = df.merge(polygon_attrs, on="polygon_id", how="left")

    _write_dataframe_parquet(df, output_path)
    LOGGER.info(
        "[polygons-index] ✅ Wrote polygon pixel index → %s (%s rows)",
        output_path,
        len(df),
    )
    return output_path


def extract_polygon_parquet_from_envi(
    envi_img: Path,
    envi_hdr: Path,
    polygon_index_path: Path,
    output_parquet_path: Path,
    *,
    chunk_size: int = 50_000,
    overwrite: bool = False,
) -> Path:
    """Extract polygon-filtered parquet directly from ENVI file.
    
    This function reads an ENVI file and extracts only pixels that intersect
    with the polygon, writing directly to a parquet file. This avoids creating
    the full parquet file first.
    
    Parameters
    ----------
    envi_img : Path
        Path to ENVI .img file
    envi_hdr : Path
        Path to ENVI .hdr file
    polygon_index_path : Path
        Path to polygon pixel index parquet (from build_polygon_pixel_index)
    output_parquet_path : Path
        Output path for the polygon-filtered parquet
    chunk_size : int, default 50_000
        Chunk size for reading ENVI file
    overwrite : bool, default False
        Whether to overwrite existing parquet file
    
    Returns
    -------
    Path
        Path to the created polygon parquet file
    """
    from cross_sensor_cal.parquet_export import (
        read_envi_in_chunks,
        _write_parquet_chunks,
    )
    from cross_sensor_cal.exports.schema_utils import infer_stage_from_name
    import inspect
    
    output_parquet_path = Path(output_parquet_path)
    
    if output_parquet_path.exists() and not overwrite:
        LOGGER.info(
            "[polygons-extract-envi] Reusing existing polygon parquet → %s",
            output_parquet_path,
        )
        return output_parquet_path
    
    # Read polygon pixel IDs from index
    con = duckdb.connect()
    try:
        pixel_ids_df = con.execute(
            f"SELECT DISTINCT pixel_id FROM read_parquet('{_quote_path(polygon_index_path)}')"
        ).df()
        polygon_pixel_ids = set(pixel_ids_df["pixel_id"].tolist())
        LOGGER.info(
            "[polygons-extract-envi] Polygon contains %d unique pixels",
            len(polygon_pixel_ids),
        )
    finally:
        con.close()
    
    if not polygon_pixel_ids:
        raise ValueError("Polygon index contains no pixels")
    
    # Read ENVI in chunks and filter by polygon pixel_ids
    parquet_name = output_parquet_path.name
    stage_key = infer_stage_from_name(parquet_name)
    
    chunk_iter = read_envi_in_chunks(
        Path(envi_img),
        Path(envi_hdr),
        parquet_name,
        chunk_size=chunk_size,
    )
    
    # Get context from original iterator before filtering
    # read_envi_in_chunks should always return an object with context (either wrapped or with attribute)
    context = {}
    try:
        if hasattr(chunk_iter, "context"):
            context = getattr(chunk_iter, "context", {})
            if context is None:
                context = {}
        else:
            LOGGER.warning(
                "[polygons-extract-envi] chunk_iter from read_envi_in_chunks has no context attribute. "
                "This may cause issues. Type: %s",
                type(chunk_iter).__name__
            )
            context = {}
    except Exception as e:
        LOGGER.warning(
            "[polygons-extract-envi] Failed to extract context from chunk_iter: %s. Using empty dict.",
            e
        )
        context = {}
    
    LOGGER.debug(
        "[polygons-extract-envi] Extracted context: %s (type: %s)",
        type(context).__name__,
        list(context.keys()) if isinstance(context, dict) else "N/A"
    )
    
    # Filter chunks to only include polygon pixels
    def _filtered_iterator():
        for df_chunk in chunk_iter:
            if df_chunk.empty:
                continue
            # Filter to only polygon pixels
            filtered = df_chunk[df_chunk["pixel_id"].isin(polygon_pixel_ids)]
            if not filtered.empty:
                yield filtered
    
    filtered_iter = _filtered_iterator()
    
    # Always wrap filtered iterator to preserve context attribute
    # This ensures _write_parquet_chunks can access .context if needed
    class _FilteredIteratorWrapper:
        def __init__(self, gen, ctx):
            self._gen = gen
            # Ensure context is always a dict, never None
            self.context = ctx if isinstance(ctx, dict) else {}
        
        def __iter__(self):
            return self
        
        def __next__(self):
            return next(self._gen)
    
    # Always wrap, even if context is empty
    filtered_iter = _FilteredIteratorWrapper(filtered_iter, context)
    
    # Verify the wrapper has context
    if not hasattr(filtered_iter, "context"):
        raise RuntimeError(
            "FilteredIteratorWrapper was created but doesn't have context attribute. "
            "This is a bug in the code."
        )
    
    write_kwargs: dict[str, object] = {}
    if context is not None:
        write_kwargs["context"] = context
    
    parameters = inspect.signature(_write_parquet_chunks).parameters
    if "row_group_size" in parameters:
        write_kwargs["row_group_size"] = chunk_size
    
    # Verify filtered_iter has context before calling _write_parquet_chunks
    if not hasattr(filtered_iter, "context"):
        raise RuntimeError(
            f"filtered_iter (type: {type(filtered_iter).__name__}) does not have context attribute. "
            "This should not happen after wrapping."
        )
    
    LOGGER.debug(
        "[polygons-extract-envi] Calling _write_parquet_chunks with filtered_iter (type: %s, has_context: %s)",
        type(filtered_iter).__name__,
        hasattr(filtered_iter, "context")
    )
    
    try:
        _write_parquet_chunks(output_parquet_path, filtered_iter, stage_key, **write_kwargs)
    except AttributeError as e:
        if "'generator' object has no attribute 'context'" in str(e):
            LOGGER.error(
                "[polygons-extract-envi] ❌ Context error in _write_parquet_chunks. "
                "filtered_iter type: %s, has_context: %s, context value: %s",
                type(filtered_iter).__name__,
                hasattr(filtered_iter, "context"),
                getattr(filtered_iter, "context", "MISSING")
            )
            import traceback
            LOGGER.error("[polygons-extract-envi] Full traceback:\n%s", traceback.format_exc())
        raise
    
    LOGGER.info(
        "[polygons-extract-envi] ✅ Wrote polygon parquet → %s",
        output_parquet_path,
    )
    
    return output_parquet_path


def extract_all_polygon_parquets_from_envi(
    flight_paths: FlightlinePaths,
    polygon_index_path: Path,
    *,
    chunk_size: int = 50_000,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Extract polygon parquets directly from all ENVI files in flightline.
    
    This discovers all ENVI files and extracts polygon-filtered parquets
    directly from them, skipping full parquet creation.
    
    Parameters
    ----------
    flight_paths : FlightlinePaths
        Flightline paths object
    polygon_index_path : Path
        Path to polygon pixel index parquet
    chunk_size : int, default 50_000
        Chunk size for reading ENVI files
    overwrite : bool, default False
        Whether to overwrite existing polygon parquets
    
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping product names to polygon parquet paths
    """
    flight_dir = flight_paths.flight_dir
    outputs: Dict[str, Path] = {}
    
    # Find all ENVI files
    envi_files = sorted(flight_dir.glob("*.img"))
    envi_files = [
        p for p in envi_files
        if "mask" not in p.stem.lower()
        and "angle" not in p.stem.lower()
        and "qa" not in p.stem.lower()
        and "quality" not in p.stem.lower()
    ]
    
    LOGGER.info(
        "[polygons-extract-all] Found %d ENVI files to extract",
        len(envi_files),
    )
    
    for img_path in envi_files:
        hdr_path = img_path.with_suffix(".hdr")
        if not hdr_path.exists():
            LOGGER.warning(
                "[polygons-extract-all] Skipping %s (no .hdr file)",
                img_path.name,
            )
            continue
        
        # Determine product name and output path
        stem = img_path.stem
        if "_undarkened_envi" in stem:
            product_name = stem.replace("_undarkened_envi", "_undarkened_envi")
        elif "_envi" in stem:
            product_name = stem
        else:
            product_name = stem
        
        output_path = flight_dir / f"{product_name}_polygons.parquet"
        
        try:
            extract_polygon_parquet_from_envi(
                img_path,
                hdr_path,
                polygon_index_path,
                output_path,
                chunk_size=chunk_size,
                overwrite=overwrite,
            )
            outputs[product_name] = output_path
        except Exception as exc:
            LOGGER.warning(
                "[polygons-extract-all] Failed to extract %s: %s",
                img_path.name,
                exc,
            )
    
    return outputs


def extract_polygon_parquets_for_flightline(
    flight_paths: FlightlinePaths,
    polygon_index_path: Path | str,
    products: list[str] | None = None,
    *,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Filter per-product Parquet tables down to polygon pixels."""

    polygon_index_path = Path(polygon_index_path)
    if not polygon_index_path.exists():
        raise FileNotFoundError(polygon_index_path)

    product_paths = _available_product_parquets(flight_paths)
    if products is None:
        target_products = list(product_paths.keys())
    else:
        target_products = products

    outputs: Dict[str, Path] = {}
    con = duckdb.connect()
    try:
        for product in target_products:
            parquet_path = product_paths.get(product)
            if parquet_path is None:
                raise ValueError(f"Unknown product key: {product}")
            if not parquet_path.exists():
                LOGGER.warning(
                    "[polygons-extract] Missing parquet for %s → %s", product, parquet_path
                )
                continue

            out_path = parquet_path.with_name(f"{parquet_path.stem}_polygons.parquet")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not overwrite:
                LOGGER.info(
                    "[polygons-extract] Reusing %s polygons parquet → %s",
                    product,
                    out_path,
                )
                outputs[product] = out_path
                continue

            LOGGER.info(
                "[polygons-extract] Filtering %s → %s", parquet_path.name, out_path.name
            )
            sql = (
                "COPY ("
                "SELECT p.* FROM read_parquet('"
                + _quote_path(parquet_path)
                + "') p "
                "INNER JOIN read_parquet('"
                + _quote_path(polygon_index_path)
                + "') idx USING (pixel_id)"
                ") TO '"
                + _quote_path(out_path)
                + "' (FORMAT PARQUET)"
            )
            con.execute(sql)
            outputs[product] = out_path
            LOGGER.info(
                "[polygons-extract] ✅ Wrote %s polygons parquet → %s", product, out_path
            )
    finally:
        con.close()

    return outputs


def merge_polygon_parquets_for_flightline(
    flight_paths: FlightlinePaths,
    polygon_index_path: Path | str,
    product_polygon_parquets: Mapping[str, Path],
    output_path: Path | str | None = None,
    *,
    overwrite: bool = False,
    row_group_size: int = 25_000,
) -> Path:
    """Merge polygon-filtered Parquet tables into one spectral library."""

    polygon_index_path = Path(polygon_index_path)
    if not polygon_index_path.exists():
        raise FileNotFoundError(polygon_index_path)

    if output_path is None:
        output_path = (
            flight_paths.flight_dir
            / f"{flight_paths.flight_id}_polygons_merged_pixel_extraction.parquet"
        )
    else:
        output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        LOGGER.info(
            "[polygons-merge] Reusing existing polygon merge → %s", output_path
        )
        return output_path

    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")

    filtered_products = {
        key: Path(path)
        for key, path in product_polygon_parquets.items()
        if Path(path).exists()
    }
    if not filtered_products:
        raise ValueError("No polygon parquet tables were provided for merging")

    LOGGER.info(
        "[polygons-merge] Start merge for %s with %d products",
        flight_paths.flight_id,
        len(filtered_products),
    )

    con = duckdb.connect()
    try:
        index_columns = _describe_parquet_columns(con, polygon_index_path)
        select_terms = [
            f"idx.{_quote_identifier(col)} AS {_quote_identifier(col)}"
            for col in index_columns
        ]
        joins: list[str] = []
        seen_columns = set(index_columns)

        for product, parquet_path in filtered_products.items():
            alias = _sanitise_alias(product)
            columns = _describe_parquet_columns(con, parquet_path)
            keep_cols = [
                col
                for col in columns
                if col != "pixel_id" and col not in seen_columns
            ]
            for col in keep_cols:
                select_terms.append(
                    f"{alias}.{_quote_identifier(col)} AS {_quote_identifier(col)}"
                )
                seen_columns.add(col)
            joins.append(
                "LEFT JOIN read_parquet('"
                + _quote_path(parquet_path)
                + f"') {alias} USING (pixel_id)"
            )

        select_sql = (
            "SELECT "
            + ", ".join(select_terms)
            + " FROM read_parquet('"
            + _quote_path(polygon_index_path)
            + "') idx "
            + " ".join(joins)
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        LOGGER.info("[polygons-merge] Writing merged parquet → %s", output_path)
        copy_sql = (
            "COPY ("
            + select_sql
            + ") TO '"
            + _quote_path(output_path)
            + "' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE "
            + str(int(row_group_size))
            + ")"
        )
        con.execute(copy_sql)
    finally:
        con.close()

    LOGGER.info("[polygons-merge] ✅ Polygon spectral library → %s", output_path)
    return output_path


def run_polygon_pipeline_for_flightline(
    flight_paths: FlightlinePaths,
    polygons_path: str | Path,
    *,
    products: list[str] | None = None,
    reference_product: str = "brdfandtopo_corrected_envi",
    overwrite: bool = False,
) -> Dict[str, object]:
    """Execute the polygon pipeline end-to-end for one flightline."""

    polygon_index_path = build_polygon_pixel_index(
        flight_paths,
        polygons_path,
        reference_product=reference_product,
        overwrite=overwrite,
    )
    product_polygon_parquets = extract_polygon_parquets_for_flightline(
        flight_paths,
        polygon_index_path,
        products=products,
        overwrite=overwrite,
    )
    merged_path = merge_polygon_parquets_for_flightline(
        flight_paths,
        polygon_index_path,
        product_polygon_parquets,
        overwrite=overwrite,
    )
    return {
        "polygon_index_path": polygon_index_path,
        "product_polygon_parquets": product_polygon_parquets,
        "polygon_merged_parquet": merged_path,
    }


__all__ = [
    "build_polygon_pixel_index",
    "extract_polygon_parquets_for_flightline",
    "merge_polygon_parquets_for_flightline",
    "run_polygon_pipeline_for_flightline",
    "create_dummy_polygon",
    "filter_polygons_by_overlap",
    "visualize_polygons_on_envi",
    "validate_coordinate_match",
]

