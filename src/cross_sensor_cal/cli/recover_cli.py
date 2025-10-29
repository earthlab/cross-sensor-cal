import argparse
import logging
from pathlib import Path

from cross_sensor_cal.pipelines.pipeline import stage_export_envi_from_h5

logger = logging.getLogger(__name__)


def main() -> None:
    """Recover raw ENVI exports when corrected products already exist."""

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=(
            "Backfill raw ENVI for flightlines where corrected exists but raw is missing."
        )
    )
    parser.add_argument("--base-folder", required=True, type=Path)
    parser.add_argument("--brightness-offset", type=float, default=0.0)
    args = parser.parse_args()

    base_folder: Path = args.base_folder

    for h5_path in sorted(base_folder.glob("*.h5")):
        flight_stem = h5_path.stem
        work_dir = base_folder / flight_stem
        if not work_dir.is_dir():
            continue

        raw_img = work_dir / f"{flight_stem}_envi.img"
        corrected_img = work_dir / f"{flight_stem}_brdfandtopo_corrected_envi.img"

        if corrected_img.exists() and not raw_img.exists():
            logger.info("♻️  Recovering raw for %s", flight_stem)
            stage_export_envi_from_h5(
                base_folder=base_folder,
                product_code="DP1.30006.001",
                flight_stem=flight_stem,
                brightness_offset=args.brightness_offset,
                parallel_mode=False,
                recover_missing_raw=True,
            )

    logger.info("✅ Recovery pass complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
