import math
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

from ._optional import require_geopandas, require_rasterio


def _require_pyarrow():
    """Lazy import pyarrow with a helpful error message."""

    try:
        import pyarrow as pa

        return pa
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in optional envs
        raise ModuleNotFoundError(
            "The 'pyarrow' package is required for polygon extraction IO. "
            "Install it with: pip install pyarrow"
        ) from exc


def _require_pyarrow_parquet():
    """Lazy import pyarrow.parquet via :func:`_require_pyarrow`."""

    pa = _require_pyarrow()
    return pa.parquet

from .file_types import DataFile, NEONReflectanceENVIFile, NEONReflectanceBRDFCorrectedENVIFile, \
    NEONReflectanceResampledENVIFile, SpectralDataParquetFile


_BAND_RE = re.compile(r"^(ENVI|Masked|Original)_band_(\d+)$")


def _leading_band_columns(colnames):
    """Return the leading run of band columns (in stored order)."""

    bands = []
    for column in colnames:
        match = _BAND_RE.match(column)
        if match:
            bands.append(column)
        else:
            break
    return bands


def _band_indices(band_cols):
    """Return the integer band indices extracted from band column names."""

    indices = []
    for column in band_cols:
        match = _BAND_RE.match(column)
        if match:
            indices.append(int(match.group(2)))
    return indices


def validate_bands_in_dir(
    directory,
    expected_bands: int = 426,
    sample_rows: int = 4000,
    recursive: bool = True,
    value_range: tuple | None = (0.0, 1.0),
    eps_all_zero: float = 1e-12,
    show_preview: bool = True,
    meta_cols=("Raster_File", "CRS", "Chunk_Number", "Pixel_X", "Pixel_Y"),
):
    """Validate band layout and sample values for each Parquet file in ``directory``."""

    directory = Path(directory)
    files = sorted(directory.rglob("*.parquet") if recursive else directory.glob("*.parquet"))
    if not files:
        print(f"[INFO] No .parquet files in {directory}")
        return pd.DataFrame()

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)

    rows = []
    for parquet_file in files:
        record = {
            "file": str(parquet_file),
            "ok_open": False,
            "num_row_groups": None,
            "num_rows_meta": None,
            "band_count": None,
            "first_cols_head": None,
            "first_band": None,
            "last_band": None,
            "band_index_min": None,
            "band_index_max": None,
            "bands_ok": None,
            "finite_ok": None,
            "all_zero_bands": [],
            "range_fail_bands": [],
            "error": None,
        }

        try:
            pq = _require_pyarrow_parquet()
            pq_file = pq.ParquetFile(parquet_file)
            record["ok_open"] = True
            record["num_row_groups"] = pq_file.num_row_groups
            record["num_rows_meta"] = pq_file.metadata.num_rows if pq_file.metadata else None

            schema_columns = pq_file.schema_arrow.names
            record["first_cols_head"] = schema_columns[:8]

            band_columns = _leading_band_columns(schema_columns)
            record["band_count"] = len(band_columns)

            indices = _band_indices(band_columns)
            if indices:
                record["band_index_min"] = int(np.min(indices))
                record["band_index_max"] = int(np.max(indices))
            record["first_band"] = band_columns[0] if band_columns else None
            record["last_band"] = band_columns[-1] if band_columns else None

            record["bands_ok"] = record["band_count"] == expected_bands

            meta_columns = [column for column in meta_cols if column in schema_columns]
            columns_to_read = band_columns + meta_columns
            table = pq_file.read_row_group(0, columns=columns_to_read)
            dataframe = table.to_pandas()
            if len(dataframe) > sample_rows:
                dataframe = dataframe.head(sample_rows)
            band_dataframe = dataframe[band_columns].astype("float32") if band_columns else pd.DataFrame()

            record["finite_ok"] = bool(np.isfinite(band_dataframe.to_numpy()).all()) if not band_dataframe.empty else False

            all_zero_bands = []
            if not band_dataframe.empty:
                max_abs = band_dataframe.abs().max(axis=0)
                for column, maximum in max_abs.items():
                    if not (maximum > eps_all_zero):
                        all_zero_bands.append(column)
            record["all_zero_bands"] = all_zero_bands

            range_fail_bands = []
            if value_range is not None and not band_dataframe.empty:
                lo, hi = value_range
                too_low = (band_dataframe < lo).any(axis=0)
                too_high = (band_dataframe > hi).any(axis=0)
                for column in band_columns:
                    if (column in too_low.index and bool(too_low[column])) or (
                        column in too_high.index and bool(too_high[column])
                    ):
                        range_fail_bands.append(column)
            record["range_fail_bands"] = range_fail_bands

            if show_preview and not dataframe.empty:
                preview_columns = meta_columns + band_columns[:5]
                preview_df = dataframe[preview_columns].head(1)
                print(f"\n=== {parquet_file.name} ===")
                print(
                    "leading band cols: "
                    f"{record['band_count']} (expected {expected_bands}) | "
                    f"first: {record['first_band']} | last: {record['last_band']} | "
                    f"indices: [{record['band_index_min']}, {record['band_index_max']}] | "
                    f"row_groups: {record['num_row_groups']} | rows(meta): {record['num_rows_meta']}"
                )
                print(preview_df.to_string(index=False))

        except Exception as exc:  # pragma: no cover - diagnostic print for unexpected failure
            record["error"] = str(exc)

        rows.append(record)

    result = pd.DataFrame(rows)

    print("\n===== BAND LAYOUT VALIDATION =====")
    print(f"Directory: {directory}")
    print(f"Files: {len(result)}")
    print(f"Open errors: {(result['ok_open'] == False).sum()}")
    print(f"Band-count mismatches: {(result['bands_ok'] == False).sum()} (expected {expected_bands})")
    print(f"Finite check failures: {(result['finite_ok'] == False).sum()}")
    print(
        "Files with all-zero bands (any): "
        f"{(result['all_zero_bands'].apply(lambda x: len(x) > 0)).sum()}"
    )
    if value_range is not None:
        print(
            "Range check failures (any band outside "
            f"{value_range}): {(result['range_fail_bands'].apply(lambda x: len(x) > 0)).sum()}"
        )

    compact = result[
        [
            "file",
            "band_count",
            "bands_ok",
            "first_band",
            "last_band",
            "band_index_min",
            "band_index_max",
            "num_row_groups",
            "num_rows_meta",
            "finite_ok",
            "all_zero_bands",
            "range_fail_bands",
            "error",
        ]
    ].copy()
    print(compact.head(50).to_string(index=False))
    return result


def _leading_band_columns_sorted(colnames):
    bands = []
    for c in colnames:
        match = _BAND_RE.match(c)
        if match:
            bands.append((c, int(match.group(2))))
        else:
            break
    bands.sort(key=lambda x: x[1])
    return [c for c, _ in bands], [i for _, i in bands]


def _process_single_raster(
    raster_file: DataFile,
    polygon_path: Optional[Path],
    *,
    overwrite: bool = False,
):
    spectral_parquet_file = SpectralDataParquetFile.from_raster_file(raster_file)
    if spectral_parquet_file.path.exists() and not overwrite:
        print(
            f"[INFO] Skipping extraction for {raster_file.path.name};"
            f" output {spectral_parquet_file.path} already exists."
        )
        return spectral_parquet_file.path

    print(f"[DEBUG] Writing to {spectral_parquet_file.path}")
    process_raster_in_chunks(
        raster_file,
        polygon_path,
        spectral_parquet_file,
        overwrite=overwrite,
    )

    return spectral_parquet_file.path


def control_function_for_extraction(
    directory,
    polygon_path: Optional[Path],
    max_workers: Optional[int] = None,
    *,
    overwrite: bool = False,
):
    """
    Finds and processes raster files in a directory.
    Processes data in chunks and saves output to Parquet.

    When more than one raster file is found, the extractions run in parallel using a
    process pool. The ``max_workers`` argument can be used to control the number of
    concurrent processes. By default, existing extraction outputs are reused; pass
    ``overwrite=True`` to regenerate them.
    """
    raster_files = get_all_priority_rasters(directory, 'envi')
    plot_directories: Set[Path] = set()

    if not raster_files:
        print(f"[DEBUG] No matching raster files found in {directory}.")
        return

    if len(raster_files) == 1:
        try:
            output_path = _process_single_raster(
                raster_files[0],
                polygon_path,
                overwrite=overwrite,
            )
            if output_path:
                plot_directories.add(output_path.parent)
        except Exception as e:
            print(f"[ERROR] Error while processing raster file {raster_files[0].file_path}: {e}")
        else:
            _validate_extracted_directories(plot_directories)
            _generate_spectral_plots(plot_directories)
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_raster,
                raster_file,
                polygon_path,
                overwrite=overwrite,
            ): raster_file
            for raster_file in raster_files
        }
        for future in as_completed(futures):
            raster_file = futures[future]
            try:
                output_path = future.result()
                if output_path:
                    plot_directories.add(output_path.parent)
            except Exception as e:
                print(f"[ERROR] Error while processing raster file {raster_file.file_path}: {e}")
    _validate_extracted_directories(plot_directories)
    _generate_spectral_plots(plot_directories)


def _validate_extracted_directories(plot_directories: Set[Path]) -> None:
    for directory in sorted(plot_directories):
        try:
            validate_bands_in_dir(
                directory,
                expected_bands=426,
                value_range=(0.0, 10000.0),
                sample_rows=4000,
                recursive=True,
                show_preview=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostic print for unexpected failure
            print(f"[ERROR] Failed to validate bands for {directory}: {exc}")


def _generate_spectral_plots(plot_directories: Set[Path]) -> None:
    if not plot_directories:
        return

    for directory in sorted(plot_directories):
        try:
            output_dir = directory.parent / f"{directory.name}_spectral_plots"
            plot_spectra_from_parquet_dir(
                directory,
                output_dir=output_dir,
                recursive=False,
            )
        except Exception as exc:
            print(f"[ERROR] Failed to generate spectral plots for {directory}: {exc}")


def plot_spectra_from_parquet_dir(
    directory,
    output_dir: Optional[str] = None,
    recursive: bool = True,
    max_lines_per_file: int = 5000,
    median_sample_rows: int = 50000,
    sentinel_at_or_below: float = -9990,
    mask_integer_lo: int = 1,
    mask_integer_hi: int = 255,
    dpi: int = 200,
):
    """Generate spectral plots for each Parquet file within ``directory``."""

    directory = Path(directory)
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = directory.parent / f"{directory.name}_spectral_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(directory.rglob("*.parquet") if recursive else directory.glob("*.parquet"))
    if not files:
        print(f"[INFO] No .parquet files found under: {directory}")
        return []

    generated = []
    rng = np.random.default_rng(42)

    for parquet_path in files:
        try:
            pq = _require_pyarrow_parquet()
            parquet_file = pq.ParquetFile(parquet_path)
        except Exception as exc:
            print(f"[WARN] Skipping {parquet_path.name}: cannot open ({exc})")
            continue

        columns = parquet_file.schema_arrow.names
        band_columns, band_indices = _leading_band_columns_sorted(columns)
        if not band_columns:
            print(f"[WARN] {parquet_path.name}: no leading band columns found; skipping.")
            continue

        total_rows = (
            parquet_file.metadata.num_rows
            if parquet_file.metadata and parquet_file.metadata.num_rows is not None
            else None
        )
        if total_rows is None or total_rows == 0:
            print(f"[WARN] {parquet_path.name}: no rows; skipping.")
            continue

        stride = max(1, math.ceil(total_rows / max_lines_per_file)) if max_lines_per_file else 1
        probability = min(1.0, median_sample_rows / total_rows) if median_sample_rows else 1.0

        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        x_values = np.array(band_indices, dtype=np.float32)

        samples = []
        plotted = 0
        processed_rows = 0

        for row_group_index in range(parquet_file.num_row_groups):
            try:
                table = parquet_file.read_row_group(row_group_index, columns=band_columns)
            except Exception as exc:
                print(
                    f"[WARN] {parquet_path.name}: failed to read row group {row_group_index} ({exc}); continuing."
                )
                continue

            dataframe = table.to_pandas()
            array = dataframe.to_numpy(dtype=np.float32, copy=False)

            array = np.where(array <= sentinel_at_or_below, np.nan, array)
            integer_mask = (
                (array >= mask_integer_lo)
                & (array <= mask_integer_hi)
                & (array == np.floor(array))
            )
            array = np.where(integer_mask, np.nan, array)

            rows_in_group = array.shape[0]

            if max_lines_per_file:
                indices = np.arange(rows_in_group)
                plot_mask = ((indices + processed_rows) % stride == 0)
                rows_to_plot = array[plot_mask]
                for row in rows_to_plot:
                    if np.all(np.isnan(row)):
                        continue
                    ax.plot(x_values, row, linewidth=0.5, alpha=0.15)
                    plotted += 1

            if median_sample_rows:
                if probability >= 1.0:
                    sampled = array
                else:
                    sample_size = int(np.ceil(rows_in_group * probability))
                    if sample_size > 0 and rows_in_group > 0:
                        selected = rng.choice(rows_in_group, size=min(sample_size, rows_in_group), replace=False)
                        sampled = array[selected]
                    else:
                        sampled = None
                if sampled is not None and sampled.size > 0:
                    samples.append(sampled)

            processed_rows += rows_in_group

        if median_sample_rows and samples:
            sample_array = np.vstack(samples)
            if sample_array.shape[0] > median_sample_rows:
                selection = rng.choice(sample_array.shape[0], size=median_sample_rows, replace=False)
                sample_array = sample_array[selection]
            median_values = np.nanmedian(sample_array, axis=0)
            ax.plot(x_values, median_values, linewidth=2.0, alpha=1.0, label="Median (50th)")
            ax.legend()

        ax.set_xlabel("Band index")
        ax.set_ylabel("Reflectance")
        ax.set_title(parquet_path.name)
        fig.tight_layout()

        output_path = out_dir / f"{parquet_path.stem}_spectra.png"
        try:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            generated.append(output_path)
        except Exception as exc:
            print(f"[WARN] Failed to save plot for {parquet_path.name}: {exc}")
        finally:
            plt.close(fig)

        print(f"[OK] {parquet_path.name}: plotted ~{plotted} pixel lines -> {output_path}")

    return generated


def select_best_files(files: List[DataFile]) -> List[DataFile]:
    grouped = defaultdict(list)
    for f in files:
        key = (f.domain, f.site, f.date, f.time)
        grouped[key].append(f)

    selected = []
    for group in grouped.values():
        # Sort using priority: masked + BRDF > masked > unmasked BRDF > unmasked
        best = sorted(
            group,
            key=lambda f: (
                not getattr(f, "is_masked", False),
                "brdf" not in f.path.name.lower(),
                getattr(f, "suffix", "") != "envi"
            )
        )[0]
        selected.append(best)

    return selected


def find_best_resampled_files(directory: Path, suffix: str) -> List[NEONReflectanceResampledENVIFile]:
    all_resampled = NEONReflectanceResampledENVIFile.find_all_sensors_in_directory(directory, suffix)
    return select_best_files(all_resampled)


def get_all_priority_rasters(base_dir: Path, suffix: str = 'envi') -> List[DataFile]:
    """Return all raster files in ``base_dir`` prioritising BRDF corrected files first.

    Previously this function attempted to select a single "best" raster per
    (domain, site, date, time) combination which meant only one file was
    processed when multiple valid rasters were present in the directory.  The
    polygon extraction workflow now requires that every available raster be
    processed, so we gather all matches while still preferring BRDF corrected
    rasters when duplicates exist.
    """

    # 1. Get BRDF-corrected reflectance files
    brdf_files = NEONReflectanceBRDFCorrectedENVIFile.find_in_directory(base_dir, suffix)

    # 2. Get original reflectance files
    raw_files = NEONReflectanceENVIFile.find_in_directory(base_dir)

    # 3. Get all resampled files
    resampled_files = NEONReflectanceResampledENVIFile.find_all_sensors_in_directory(base_dir, suffix)

    ordered_groups: List[List[DataFile]] = [brdf_files, raw_files, resampled_files]
    seen_paths = set()
    prioritised_files: List[DataFile] = []

    for group in ordered_groups:
        for data_file in group:
            if data_file.path not in seen_paths:
                prioritised_files.append(data_file)
                seen_paths.add(data_file.path)

    return prioritised_files



def get_crs_from_hdr(hdr_path):
    """
    Reads an ENVI .hdr file and extracts the CRS as a Proj string or EPSG code.

    Parameters:
    - hdr_path (str): Path to the .hdr file.

    Returns:
    - crs (rasterio.crs.CRS or None): CRS object if found, else None.
    """
    try:
        # Open file with 'latin1' encoding and ignore errors to bypass problematic bytes.
        with open(hdr_path, 'r', encoding='latin1', errors='ignore') as f:
            lines = f.readlines()

        for line in lines:
            lower_line = line.lower()
            if "coordinate system string" in lower_line:
                proj_str = re.search(r'coordinate system string = (.*)', line, re.IGNORECASE)
                if proj_str:
                    wkt = proj_str.group(1).strip()
                    if wkt:
                        rasterio = require_rasterio()
                        return rasterio.crs.CRS.from_wkt(wkt)  # Convert WKT to CRS

            elif "map info" in lower_line:
                map_info = re.search(r'map info = {(.*?)}', line, re.IGNORECASE)
                if map_info:
                    values = map_info.group(1).split(',')
                    try:
                        utm_zone = int(values[7])
                    except (IndexError, ValueError):
                        continue
                    hemisphere = values[8].strip().lower()
                    datum = values[9].strip() if len(values) > 9 else ""
                    if datum == "WGS-84":
                        rasterio = require_rasterio()
                        return (
                            rasterio.crs.CRS.from_epsg(32600 + utm_zone)
                            if hemisphere == "north"
                            else rasterio.crs.CRS.from_epsg(32700 + utm_zone)
                        )

        return None  # Return None if no CRS is found

    except Exception as e:
        print(f"[ERROR] Could not extract CRS from {hdr_path}: {e}")
        return None


def process_raster_in_chunks(
    raster_file: DataFile,
    polygon_path: Optional[Path],
    output_parquet_file: DataFile,
    chunk_size=100000,
    *,
    overwrite: bool = False,
):
    """
    Processes a raster file in chunks, intersects pixels with a polygon, and writes extracted
    spectral and spatial data to a Parquet file. Existing outputs are reused unless
    ``overwrite`` is ``True``.
    """

    raster_path = raster_file.path
    output_parquet_path = output_parquet_file.path
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if output_parquet_path.exists():
        if overwrite:
            output_parquet_path.unlink()
        else:
            print(
                f"[INFO] Output {output_parquet_path} already exists; skipping extraction."
                " Use overwrite=True to regenerate."
            )
            return
    hdr_path = raster_path.with_suffix(".hdr")

    rasterio = require_rasterio()

    with rasterio.open(raster_path) as src:
        crs_from_hdr = None
        if hdr_path.exists():
            crs_from_hdr = get_crs_from_hdr(hdr_path)

        dataset_crs = src.crs if src.crs is not None else crs_from_hdr
        if src.crs is None and crs_from_hdr is not None:
            print(f"[INFO] Using CRS from .hdr file: {crs_from_hdr}")

        polygons = None
        polygon_values = None
        polygon_attributes = None

        if polygon_path is not None:
            gpd = require_geopandas()
            polygons = gpd.read_file(polygon_path)

            if polygons.crs is None:
                print(f"[WARNING] {polygon_path} has no CRS. Assigning from .hdr file if available.")
                polygons = polygons.set_crs(dataset_crs if dataset_crs else "EPSG:4326")

            if dataset_crs is not None and polygons.crs != dataset_crs:
                polygons = polygons.to_crs(dataset_crs)

            polygon_values = rasterio.features.rasterize(
                [(geom, idx + 1) for idx, geom in enumerate(polygons.geometry)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                dtype="int32"
            ).ravel()

            polygon_attributes = polygons.reset_index().rename(columns={'index': 'Polygon_ID'})

        total_bands = src.count
        height, width = src.height, src.width
        num_chunks = (height * width // chunk_size) + (1 if (height * width) % chunk_size else 0)

        # Smart prefixing
        if getattr(raster_file, "is_masked", False):
            band_prefix = "Masked_band_"
        elif getattr(raster_file, "suffix", "") == "envi":
            band_prefix = "ENVI_band_"
        else:
            band_prefix = "Original_band_"

        print(f"[INFO] Processing {raster_path.name} with {total_bands} bands as {band_prefix}")

        pq_module = _require_pyarrow_parquet()
        pa_module = _require_pyarrow()
        pq_writer = None
        try:
            with tqdm(total=num_chunks, desc=f"Processing {raster_path.name}", unit="chunk") as pbar:
                for i in range(num_chunks):
                    row_start = (i * chunk_size) // width
                    row_end = min(((i + 1) * chunk_size) // width + 1, height)

                    data = src.read(window=((row_start, row_end), (0, width)))
                    data_chunk = data.reshape(total_bands, -1).T

                    row_indices, col_indices = np.meshgrid(
                        np.arange(row_start, row_end),
                        np.arange(width),
                        indexing='ij'
                    )
                    row_indices_flat = row_indices.flatten()
                    col_indices_flat = col_indices.flatten()

                    valid_mask = ~np.any(data_chunk == -9999, axis=1)
                    data_chunk = data_chunk[valid_mask]

                    valid_rows = row_indices_flat[valid_mask]
                    valid_cols = col_indices_flat[valid_mask]
                    pixel_ids = valid_rows * width + valid_cols

                    transform = src.transform
                    x_coords = transform.a * valid_cols + transform.b * valid_rows + transform.c
                    y_coords = transform.d * valid_cols + transform.e * valid_rows + transform.f

                    if polygon_values is not None:
                        polygon_chunk = polygon_values[row_start * width:row_end * width][valid_mask]
                    else:
                        polygon_chunk = None

                    chunk_df = pd.DataFrame(data_chunk, columns=[f'Band_{b + 1}' for b in range(total_bands)])
                    chunk_df["Raster_File"] = raster_path.name
                    chunk_df["Polygon_File"] = str(polygon_path) if polygon_path is not None else None
                    chunk_df["Chunk_Number"] = i
                    chunk_df["Pixel_ID"] = pixel_ids
                    chunk_df["Pixel_X"] = x_coords
                    chunk_df["Pixel_Y"] = y_coords
                    if polygon_chunk is not None:
                        chunk_df["Polygon_ID"] = pd.Series(polygon_chunk, dtype="Int64")
                    else:
                        chunk_df["Polygon_ID"] = pd.Series(pd.NA, index=chunk_df.index, dtype="Int64")
                    if dataset_crs:
                        chunk_df["CRS"] = dataset_crs.to_string()

                    chunk_df.rename(columns={f"Band_{b + 1}": f"{band_prefix}{b + 1}" for b in range(total_bands)},
                                    inplace=True)

                    if polygon_attributes is not None:
                        chunk_df = pd.merge(chunk_df, polygon_attributes, on='Polygon_ID', how='left')

                    table = pa_module.Table.from_pandas(chunk_df, preserve_index=False)
                    if pq_writer is None:
                        pq_writer = pq_module.ParquetWriter(output_parquet_path, table.schema)
                    pq_writer.write_table(table)

                    pbar.update(1)
        finally:
            if pq_writer is not None:
                pq_writer.close()
