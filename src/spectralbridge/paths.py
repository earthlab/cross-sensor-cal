from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Dict

from .file_types import NEONReflectanceENVIFile, NEONReflectanceFile


@dataclass(frozen=True)
class SensorProductPaths:
    """Convenience wrapper for per-sensor ENVI/Parquet/QA artefacts."""

    base_folder: Path
    flight_id: str
    sensor_suffix: str

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "base_folder", Path(self.base_folder))

    @property
    def flight_dir(self) -> Path:
        return self.base_folder / self.flight_id

    @property
    def stem(self) -> str:
        return f"{self.flight_id}_{self.sensor_suffix}_envi"

    @property
    def img(self) -> Path:
        return self.flight_dir / f"{self.stem}.img"

    @property
    def hdr(self) -> Path:
        return self.flight_dir / f"{self.stem}.hdr"

    @property
    def parquet(self) -> Path:
        return self.flight_dir / f"{self.stem}.parquet"

    @property
    def qa_png(self) -> Path:
        return self.flight_dir / f"{self.stem}_qa.png"

    @property
    def qa_pdf(self) -> Path:
        return self.flight_dir / f"{self.stem}_qa.pdf"

    @property
    def qa_json(self) -> Path:
        return self.flight_dir / f"{self.stem}_qa.json"


@dataclass(frozen=True)
class FlightlinePaths:
    """Centralised path helper ensuring per-flightline isolation."""

    base_folder: Path
    flight_id: str

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        object.__setattr__(self, "base_folder", Path(self.base_folder))

    @property
    def flight_dir(self) -> Path:
        return self.base_folder / self.flight_id

    @property
    def h5(self) -> Path:
        return self.base_folder / f"{self.flight_id}.h5"

    @property
    def envi_img(self) -> Path:
        return self._raw_envi_base().with_suffix(".img")

    @property
    def envi_hdr(self) -> Path:
        return self._raw_envi_base().with_suffix(".hdr")

    @property
    def envi_parquet(self) -> Path:
        return self._raw_envi_base().with_suffix(".parquet")

    @property
    def corrected_img(self) -> Path:
        return self.flight_dir / f"{self.flight_id}_brdfandtopo_corrected_envi.img"

    @property
    def corrected_hdr(self) -> Path:
        return self.flight_dir / f"{self.flight_id}_brdfandtopo_corrected_envi.hdr"

    @property
    def corrected_json(self) -> Path:
        return self.flight_dir / f"{self.flight_id}_brdfandtopo_corrected_envi.json"

    @property
    def corrected_parquet(self) -> Path:
        return self.flight_dir / f"{self.flight_id}_brdfandtopo_corrected_envi.parquet"

    @property
    def brdf_model(self) -> Path:
        return self.flight_dir / f"{self.flight_id}_brdf_model.json"

    @property
    def qa_png(self) -> Path:
        return self.flight_dir / f"{self.flight_id}_qa.png"

    @property
    def qa_pdf(self) -> Path:
        return self.flight_dir / f"{self.flight_id}_qa.pdf"

    @property
    def qa_json(self) -> Path:
        return self.flight_dir / f"{self.flight_id}_qa.json"

    @property
    def qa_metrics_parquet(self) -> Path:
        return self.flight_dir / f"{self.flight_id}_qa_metrics.parquet"

    @property
    def merged_parquet(self) -> Path:
        return self.flight_dir / f"{self.flight_id}_merged_pixel_extraction.parquet"

    @property
    def sensor_products(self) -> Dict[str, SensorProductPaths]:
        mapping: Dict[str, SensorProductPaths] = {}
        for sensor_name in [
            "landsat_tm",
            "landsat_etm+",
            "landsat_oli",
            "landsat_oli2",
            "micasense",
            "micasense_to_match_tm_etm+",
            "micasense_to_match_oli_oli2",
        ]:
            mapping[sensor_name] = SensorProductPaths(
                base_folder=self.base_folder,
                flight_id=self.flight_id,
                sensor_suffix=sensor_name,
            )
        return mapping

    def sensor_product(self, sensor_name: str) -> SensorProductPaths:
        return SensorProductPaths(
            base_folder=self.base_folder,
            flight_id=self.flight_id,
            sensor_suffix=sensor_name,
        )

    def _raw_envi_base(self) -> Path:
        """
        Resolve the canonical stem for the uncorrected ENVI export.

        Legacy NEON flightlines that omit the tile identifier in the filename
        produce ENVI outputs whose stem includes that synthetic tile
        (e.g. ` …_20200720T163210_20200720_163210_reflectance_envi `). Newer
        flightlines with explicit tiles already match ` flight_id `.

        When parsing fails for any reason, we fall back to the historic
        ` <flight_id>_envi ` stem to preserve compatibility.
        """
        try:
            reflectance = NEONReflectanceFile.from_filename(f"{self.flight_id}.h5")
        except ValueError:
            return self.flight_dir / f"{self.flight_id}_envi"

        tile: str | None = getattr(reflectance, "tile", None)

        if not tile:
            return self.flight_dir / f"{self.flight_id}_envi"

        # For legacy datetime format files, the tile is synthetic (e.g., "20200720T163210")
        # Detect synthetic tile: if tile contains 'T' and matches date+time pattern
        is_legacy_datetime = tile and 'T' in tile and re.match(r'^\d{8}T\d{6}$', tile)
        
        if is_legacy_datetime:
            # For legacy datetime format, check if files exist with the OLD format first
            # (with T-separator and duplicate date/time), then fall back to NEW format
            date = getattr(reflectance, "date", None)
            time = getattr(reflectance, "time", None)
            descriptor = getattr(reflectance, "descriptor", "reflectance")
            
            # OLD format: NEON_{domain}_{site}_DP1_{date}T{time}_{date}_{time}_reflectance_envi.img
            # This is what files from previous runs might have
            product_part = "DP1"
            old_parts = ["NEON", reflectance.domain, reflectance.site, product_part]
            if date and time:
                old_parts.append(f"{date}T{time}")  # T-separator format
            if date:
                old_parts.append(date)
            if time:
                old_parts.append(time)
            if descriptor and "directional" in descriptor:
                old_parts.append("directional")
            old_parts.append("reflectance")
            old_parts.append("envi")
            old_filename = "_".join(old_parts) + ".img"
            old_path_stem = self.flight_dir / old_filename.replace(".img", "")
            old_img = old_path_stem.with_suffix(".img")
            old_hdr = old_path_stem.with_suffix(".hdr")
            
            # Check if old format file exists - try/except to handle any path issues
            try:
                old_img_exists = old_img.exists()
                old_hdr_exists = old_hdr.exists()
                if old_img_exists and old_hdr_exists:
                    old_img_is_file = old_img.is_file()
                    old_hdr_is_file = old_hdr.is_file()
                    if old_img_is_file and old_hdr_is_file:
                        old_img_size = old_img.stat().st_size
                        old_hdr_size = old_hdr.stat().st_size
                        if old_img_size > 0 and old_hdr_size > 0:
                            return old_path_stem
            except (OSError, ValueError):
                # If there's any issue checking the old format, fall through to new format
                pass
            
            # NEW format: NEON_{domain}_{site}_DP1_{date}_{time}_reflectance_envi.img
            # This is what neon_to_envi_no_hytools() now creates
            new_parts = ["NEON", reflectance.domain, reflectance.site, product_part]
            if date:
                new_parts.append(date)
            if time:
                new_parts.append(time)
            if descriptor and "directional" in descriptor:
                new_parts.append("directional")
            new_parts.append("reflectance")
            new_parts.append("envi")
            new_filename = "_".join(new_parts) + ".img"
            return self.flight_dir / new_filename.replace(".img", "")

        product = getattr(reflectance, "product", None) or "DP1"

        date = getattr(reflectance, "date", None)
        time = getattr(reflectance, "time", None)
        directional = getattr(reflectance, "directional", False)
        descriptor = getattr(reflectance, "descriptor", None)

        try:
            envi_file = NEONReflectanceENVIFile.from_components(
                domain=reflectance.domain,
                site=reflectance.site,
                product=product,
                tile=tile,
                date=date,
                time=time,
                directional=directional,
                descriptor=descriptor,
                folder=self.flight_dir,
            )
        except (ValueError, AttributeError):
            return self.flight_dir / f"{self.flight_id}_envi"

        return envi_file.path.with_suffix("")


def scene_prefix_from_dir(flightline_dir: Path) -> str:
    flightline_dir = Path(flightline_dir)
    # Prefer *_envi.img; else *.h5; else folder name
    imgs = sorted(flightline_dir.glob("*_envi.img"))
    if imgs:
        stem = imgs[0].stem
        return stem[:-5] if stem.endswith("_envi") else stem
    h5s = sorted(flightline_dir.glob("*.h5"))
    if h5s:
        return h5s[0].stem
    return flightline_dir.name


_site_re = re.compile(r"NEON_[A-Z0-9]+_([A-Z]{4})_DP1_")


def site_from_prefix(prefix: str) -> str | None:
    m = _site_re.search(prefix)
    return m.group(1) if m else None


def normalize_brdf_model_path(flightline_dir: Path) -> Path | None:
    """
    Ensure the BRDF model JSON in ` flightline_dir ` matches the scene prefix:
        <prefix>_brdf_model.json

    If a legacy file like 'NIWO_brdf_model.json' exists, rename it.
    Returns the normalized Path (or None if nothing found).
    """
    flightline_dir = Path(flightline_dir)
    prefix = scene_prefix_from_dir(flightline_dir)
    target = flightline_dir / f"{prefix}_brdf_model.json"
    if target.exists():
        return target

    # legacy site-level name e.g., NIWO_brdf_model.json
    site = site_from_prefix(prefix)
    if site:
        legacy = flightline_dir / f"{site}_brdf_model.json"
        if legacy.exists():
            shutil.move(str(legacy), str(target))
            return target
    # also accept any stray *_brdf_model.json and normalize to target
    for p in flightline_dir.glob("*_brdf_model.json"):
        if p != target:
            shutil.move(str(p), str(target))
            return target
    return None


__all__ = [
    "FlightlinePaths",
    "SensorProductPaths",
    "scene_prefix_from_dir",
    "site_from_prefix",
    "normalize_brdf_model_path",
]
