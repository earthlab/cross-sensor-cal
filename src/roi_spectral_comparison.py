from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from shapely.geometry import box 

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

try:  # ``spectral`` is an optional dependency used for ENVI headers.
    from spectral.io import envi
except ImportError:  # pragma: no cover - spectral is present in production envs.
    envi = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RoiResult:
    """Container for per-ROI spectral statistics."""

    dataframe: pd.DataFrame
    wavelengths: Dict[str, np.ndarray]


def _read_wavelengths(image_path: Path) -> Optional[np.ndarray]:
    """Attempt to read wavelength information that accompanies ``image_path``.

    Parameters
    ----------
    image_path:
        Path to the raster file.  The function looks for an accompanying
        ``.hdr`` ENVI header and, if found, extracts the ``wavelength`` field.

    Returns
    -------
    Optional[np.ndarray]
        An array of wavelength values in nanometres when available.  ``None``
        is returned if no header can be read or the ``spectral`` package is not
        installed.
    """

    if envi is None:
        return None

    hdr_path = image_path.with_suffix(".hdr")
    if not hdr_path.exists():
        return None

    header = envi.read_envi_header(str(hdr_path))
    wavelengths = header.get("wavelength")
    if not wavelengths:
        return None

    try:
        return np.array([float(value) for value in wavelengths], dtype=np.float32)
    except (TypeError, ValueError):  # pragma: no cover - defensive, header inconsistencies.
        return None


def _prepare_rois(roi_path: Path) -> gpd.GeoDataFrame:
    """Load ROIs from ``roi_path`` and ensure they contain valid geometries."""

    rois = gpd.read_file(roi_path)
    if rois.empty:
        raise ValueError(f"No polygon features found in {roi_path}")
    if "geometry" not in rois.columns:
        raise ValueError("ROI file does not contain a geometry column")
    rois = rois[~rois.geometry.is_empty & rois.geometry.notnull()].copy()
    if rois.empty:
        raise ValueError("All ROI geometries are empty or null")
    rois.reset_index(drop=True, inplace=True)
    return rois


def _build_roi_labels(rois: gpd.GeoDataFrame, label_column: Optional[str]) -> List[str]:
    if label_column:
        if label_column not in rois.columns:
            raise ValueError(f"ROI column '{label_column}' not found")
        labels = rois[label_column].astype(str).tolist()
    else:
        labels = [f"ROI_{idx}" for idx in rois.index]
    return labels

def extract_roi_spectra(
    image_paths: Sequence[Path | str],
    roi_path: Path | str,
    *,
    label_column: Optional[str] = None,
    statistics: Iterable[str] = ("mean", "median"),
    invalid_values: Iterable[float] = (-9999.0,),
    all_touched: bool = False,
    nonoverlap_fill: float = 9999.0,  # sentinel for no-overlap / zero-valid-pixels
) -> RoiResult:
    image_paths = [Path(path) for path in image_paths]
    roi_path = Path(roi_path)
    if not image_paths:
        raise ValueError("No image paths were provided")

    rois = _prepare_rois(roi_path)
    roi_labels = _build_roi_labels(rois, label_column)

    stats_requested = {stat.lower() for stat in statistics}
    valid_stats = {"mean", "median", "std"}
    unsupported = stats_requested - valid_stats
    if unsupported:
        raise ValueError(f"Unsupported statistics requested: {sorted(unsupported)}")

    results: List[Dict[str, object]] = []
    wavelengths_by_image: Dict[str, np.ndarray] = {}
    invalid_values = list(invalid_values)

    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with rasterio.open(image_path) as dataset:
            dataset_crs = dataset.crs
            dataset_name = image_path.name
            dataset_rois = rois
            if dataset_crs is not None and rois.crs is not None and rois.crs != dataset_crs:
                dataset_rois = rois.to_crs(dataset_crs)

            # Prepare helpers
            raster_poly = box(*dataset.bounds)
            wavelengths = _read_wavelengths(image_path)
            if wavelengths is not None and len(wavelengths) == dataset.count:
                wavelengths_by_image[dataset_name] = wavelengths
            else:
                wavelengths = None
            nodata_value = dataset.nodata
            band_numbers = np.arange(1, dataset.count + 1, dtype=int)

            for roi_index, (label, (_, roi_row)) in enumerate(zip(roi_labels, dataset_rois.iterrows())):
                geom = roi_row.geometry
                if geom is None or geom.is_empty or not geom.is_valid:
                    # Treat invalid/empty as no-overlap
                    for band_idx, band_number in enumerate(band_numbers):
                        wl = float(wavelengths[band_idx]) if wavelengths is not None else np.nan
                        for stat_name in stats_requested:
                            results.append({
                                "image": dataset_name,
                                "image_path": str(image_path),
                                "roi_index": roi_index,
                                "roi_label": label,
                                "band": int(band_number),
                                "wavelength_nm": wl,
                                "statistic": stat_name,
                                "value": float(nonoverlap_fill),
                                "pixel_count": 0,
                            })
                    continue

                # ---- EARLY BOUNDS CHECK: skip mask if no geometric overlap ----
                if not geom.intersects(raster_poly):
                    for band_idx, band_number in enumerate(band_numbers):
                        wl = float(wavelengths[band_idx]) if wavelengths is not None else np.nan
                        for stat_name in stats_requested:
                            results.append({
                                "image": dataset_name,
                                "image_path": str(image_path),
                                "roi_index": roi_index,
                                "roi_label": label,
                                "band": int(band_number),
                                "wavelength_nm": wl,
                                "statistic": stat_name,
                                "value": float(nonoverlap_fill),
                                "pixel_count": 0,
                            })
                    continue

                # ---- MASK with try/except to catch any residual no-overlap errors ----
                try:
                    data, _ = mask(
                        dataset,
                        [geom],
                        crop=True,          # keep crop for speed
                        filled=False,
                        all_touched=all_touched,
                    )
                except ValueError as e:
                    # rasterio raises when shapes don’t overlap (with crop=True)
                    msg = str(e).lower()
                    if "do not overlap" in msg or "outside bounds" in msg or "intersection is empty" in msg:
                        for band_idx, band_number in enumerate(band_numbers):
                            wl = float(wavelengths[band_idx]) if wavelengths is not None else np.nan
                            for stat_name in stats_requested:
                                results.append({
                                    "image": dataset_name,
                                    "image_path": str(image_path),
                                    "roi_index": roi_index,
                                    "roi_label": label,
                                    "band": int(band_number),
                                    "wavelength_nm": wl,
                                    "statistic": stat_name,
                                    "value": float(nonoverlap_fill),
                                    "pixel_count": 0,
                                })
                        continue
                    else:
                        # unexpected error -> re-raise
                        raise

                # If we get here, we have some window. Now mask/compute stats.
                if data.size == 0:
                    for band_idx, band_number in enumerate(band_numbers):
                        wl = float(wavelengths[band_idx]) if wavelengths is not None else np.nan
                        for stat_name in stats_requested:
                            results.append({
                                "image": dataset_name,
                                "image_path": str(image_path),
                                "roi_index": roi_index,
                                "roi_label": label,
                                "band": int(band_number),
                                "wavelength_nm": wl,
                                "statistic": stat_name,
                                "value": float(nonoverlap_fill),
                                "pixel_count": 0,
                            })
                    continue

                data = np.ma.masked_invalid(data)
                if nodata_value is not None:
                    data = np.ma.masked_equal(data, nodata_value)
                for invalid in invalid_values:
                    data = np.ma.masked_equal(data, invalid)

                flattened = data.reshape(dataset.count, -1)
                valid_counts = np.sum(~np.ma.getmaskarray(flattened), axis=1)

                if not np.any(valid_counts):
                    for band_idx, band_number in enumerate(band_numbers):
                        wl = float(wavelengths[band_idx]) if wavelengths is not None else np.nan
                        for stat_name in stats_requested:
                            results.append({
                                "image": dataset_name,
                                "image_path": str(image_path),
                                "roi_index": roi_index,
                                "roi_label": label,
                                "band": int(band_number),
                                "wavelength_nm": wl,
                                "statistic": stat_name,
                                "value": float(nonoverlap_fill),
                                "pixel_count": 0,
                            })
                    continue

                filled = flattened.filled(np.nan)
                with np.errstate(invalid="ignore"):
                    stat_values: Dict[str, np.ndarray] = {}
                    if "mean" in stats_requested:
                        stat_values["mean"] = np.nanmean(filled, axis=1)
                    if "median" in stats_requested:
                        stat_values["median"] = np.nanmedian(filled, axis=1)
                    if "std" in stats_requested:
                        stat_values["std"] = np.nanstd(filled, axis=1)

                for band_idx, band_number in enumerate(band_numbers):
                    wl = float(wavelengths[band_idx]) if wavelengths is not None and band_idx < len(wavelengths) else np.nan
                    for stat_name, stat_array in stat_values.items():
                        val = float(stat_array[band_idx]) if np.isfinite(stat_array[band_idx]) else np.nan
                        results.append({
                            "image": dataset_name,
                            "image_path": str(image_path),
                            "roi_index": roi_index,
                            "roi_label": label,
                            "band": int(band_number),
                            "wavelength_nm": wl,
                            "statistic": stat_name,
                            "value": val,
                            "pixel_count": int(valid_counts[band_idx]),
                        })

    dataframe = pd.DataFrame(results)
    if not dataframe.empty:
        dataframe.sort_values(["roi_index", "image", "band", "statistic"], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)

    return RoiResult(dataframe=dataframe, wavelengths=wavelengths_by_image)

def plot_roi_spectral_comparison(
    result: RoiResult,
    *,
    statistic: str = "mean",
    output_dir: Path | str | None = None,
    show: bool = False,
) -> List[Path]:
    """Create comparison plots for each ROI from :func:`extract_roi_spectra`.

    Parameters
    ----------
    result:
        The :class:`RoiResult` produced by :func:`extract_roi_spectra`.
    statistic:
        Name of the statistic to plot (``"mean"`` by default).  The statistic
        must be present in ``result.dataframe``.
    output_dir:
        When provided, PNG figures are written into this directory.  If omitted
        the figures are not saved.
    show:
        If ``True`` call :func:`matplotlib.pyplot.show` once all figures are
        generated.

    Returns
    -------
    list of :class:`~pathlib.Path`
        Paths to generated plot images.
    """

    dataframe = result.dataframe
    if dataframe.empty:
        return []

    statistic = statistic.lower()
    if statistic not in dataframe["statistic"].unique():
        raise ValueError(f"Statistic '{statistic}' not present in results")

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    plots: List[Path] = []
    x_column = "wavelength_nm"
    if dataframe[x_column].isna().all():
        x_column = "band"

    for roi_label, group in dataframe[dataframe["statistic"] == statistic].groupby("roi_label"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for image_name, image_df in group.groupby("image"):
            ax.plot(image_df[x_column], image_df["value"], label=image_name)

        xlabel = "Wavelength (nm)" if x_column == "wavelength_nm" else "Band"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Reflectance ({statistic})")
        ax.set_title(f"ROI: {roi_label}")
        ax.legend()
        fig.tight_layout()

        if output_path is not None:
            filename = f"{roi_label.replace(' ', '_')}_{statistic}.png"
            destination = output_path / filename
            fig.savefig(destination, dpi=200, bbox_inches="tight")
            plots.append(destination)

        if show:
            plt.show(block=False)
        else:
            plt.close(fig)

    if show:
        plt.show()

    return plots


__all__ = ["extract_roi_spectra", "plot_roi_spectral_comparison", "RoiResult"]

from pathlib import Path

TABLE_MOUNTAIN_DATA = Path(__file__).resolve().parent.parent / "data" / "Table_mountain_data"

# Define paths to your raster images
image_paths = [
    TABLE_MOUNTAIN_DATA / "HLS_L30_Boulder_09162021.tif",
    TABLE_MOUNTAIN_DATA / "NEON_D10_R10C_DP1.30006.001_L002-1_20210915_directional_resampled_Landsat_8_OLI_envi.img",
    TABLE_MOUNTAIN_DATA / "NEON_D10_R10C_DP1.30006.001_L003-1_20210915_directional_resampled_Landsat_8_OLI_envi.img",
]

# Define path to your ROI shapefile or GeoJSON
roi_path = TABLE_MOUNTAIN_DATA / "ROI_TM_NEON_LST.geojson"

# Run the extraction
result = extract_roi_spectra(
    image_paths=image_paths,
    roi_path=roi_path,
    label_column="id",        # or None if you don’t have one
    statistics=("mean", "std"),   # you can include "median" too
)

# Plot and save results
plots = plot_roi_spectral_comparison(
    result,
    statistic="mean",             # choose one of the extracted stats
    output_dir="spectra_plots",   # directory to save figures
    show=False                    # change to True to display interactively
)

print("Saved plots:", plots)
