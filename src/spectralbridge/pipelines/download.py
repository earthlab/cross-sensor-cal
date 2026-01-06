from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..envi_download import download_neon_flight_lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cscal-download",
        description="Download NEON flight line HDF5 files into a workspace directory.",
    )
    parser.add_argument("site", help="NEON site code, e.g., SOAP")
    parser.add_argument(
        "--product",
        default="DP1.30006.001",
        help="NEON product code to download (default: %(default)s)",
    )
    parser.add_argument("--year-month", required=True, help="Year-month identifier, e.g., 2021-06")
    parser.add_argument(
        "--flight",
        action="append",
        dest="flight_lines",
        required=True,
        help="Flight line identifier to download (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Directory where downloads are stored (default: %(default)s)",
    )
    return parser


def run_download(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output / args.site
    output_dir.mkdir(parents=True, exist_ok=True)

    download_neon_flight_lines(
        site_code=args.site,
        year_month=args.year_month,
        flight_lines=args.flight_lines,
        out_dir=str(output_dir),
        product_code=args.product,
    )


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    run_download()
