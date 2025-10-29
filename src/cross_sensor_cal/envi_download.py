"""Helpers for downloading NEON hyperspectral flight line products."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import requests

try:  # pragma: no cover - tqdm is optional in minimal environments
    from tqdm import tqdm
except Exception:  # pragma: no cover - fall back to no progress bar
    tqdm = None

logger = logging.getLogger(__name__)

_NEON_API_BASE = "https://data.neonscience.org/api/v0"
_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB chunks for better throughput


def list_neon_products() -> None:
    """Print the available NEON products (utility/debug helper)."""

    response = requests.get(f"{_NEON_API_BASE}/products", timeout=60)
    response.raise_for_status()
    products = response.json().get("data", [])
    for product in products:
        print(product.get("productCode"), "-", product.get("productName"))


def _find_matching_file(records: Iterable[dict], flight_line: str) -> dict | None:
    """Return the file record matching ``flight_line`` (preferring .h5 entries)."""

    for record in records:
        name = record.get("name", "")
        if not name:
            continue
        if not name.lower().endswith(".h5"):
            continue
        if flight_line not in name:
            continue
        return record
    return None


def download_neon_file(
    site_code: str,
    product_code: str,
    year_month: str,
    flight_line: str,
    out_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    session: requests.Session | None = None,
) -> tuple[Path, bool]:
    """Download a single NEON flight line HDF5.

    Parameters
    ----------
    site_code, product_code, year_month, flight_line
        Identify the desired NEON hyperspectral flight line.
    out_dir
        Directory where downloads should be written when ``output_path`` is not
        provided. Created if necessary.
    output_path
        Optional explicit destination path for the downloaded file. When
        provided, the file will always be written to this location.
    session
        Optional ``requests.Session`` to reuse HTTP connections.

    Returns
    -------
    (path, was_downloaded)
        ``path`` is the on-disk ``Path`` to the flight line. ``was_downloaded``
        is ``True`` if a new download occurred, ``False`` if the existing file
        was reused.

    Raises
    ------
    FileNotFoundError
        If the requested flight line could not be located in the NEON API
        response.
    RuntimeError
        If an HTTP error occurs or the downloaded file is empty.
    """

    if not year_month:
        raise ValueError("year_month is required")

    session = session or requests.Session()

    base_dir = Path(output_path).parent if output_path is not None else Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    data_url = f"{_NEON_API_BASE}/data/{product_code}/{site_code}/{year_month}"
    try:
        response = session.get(data_url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network issues
        raise RuntimeError(
            f"Failed to query NEON data API for {site_code}/{year_month}: {exc}"
        ) from exc

    files = response.json().get("data", {}).get("files", [])
    record = _find_matching_file(files, flight_line)
    if record is None:
        raise FileNotFoundError(
            f"No HDF5 file found for {flight_line} at {site_code} {year_month}."
        )

    file_name = record.get("name", "")
    url = record.get("url")
    if not url:
        raise RuntimeError(f"NEON API response missing download URL for {file_name}.")

    destination = Path(output_path) if output_path is not None else base_dir / file_name

    try:
        if destination.exists() and destination.stat().st_size > 0:
            return destination, False
    except OSError as exc:  # pragma: no cover - filesystem permissions
        raise RuntimeError(f"Cannot access existing file {destination}: {exc}") from exc

    temp_path = destination.with_suffix(destination.suffix + ".download")
    try:
        with session.get(url, stream=True, timeout=60) as download_resp:
            download_resp.raise_for_status()
            total_bytes_header = download_resp.headers.get("Content-Length")
            try:
                total_bytes = int(total_bytes_header) if total_bytes_header else 0
            except ValueError:
                total_bytes = 0

            bar = None
            if tqdm is not None:
                bar_total = total_bytes if total_bytes > 0 else None
                bar = tqdm(
                    total=bar_total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {flight_line}",
                    leave=False,
                    disable=False,
                )

            try:
                with temp_path.open("wb") as fh:
                    for chunk in download_resp.iter_content(
                        chunk_size=_DOWNLOAD_CHUNK_BYTES
                    ):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        if bar is not None:
                            bar.update(len(chunk))
            finally:
                if bar is not None:
                    bar.close()

        temp_path.replace(destination)
    except requests.RequestException as exc:  # pragma: no cover - network issues
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed downloading {file_name}: {exc}") from exc
    except Exception:
        if temp_path.exists():  # pragma: no cover - cleanup best-effort
            temp_path.unlink(missing_ok=True)
        raise

    if destination.stat().st_size <= 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file {destination} is empty.")

    return destination, True


def download_neon_flight_lines(
    site_code: str,
    year_month: str,
    flight_lines: Sequence[str] | str,
    out_dir: str | Path,
    product_code: str = "DP1.30006.001",
) -> list[Path]:
    """Download one or more NEON flight lines into ``out_dir``.

    Returns a list of ``Path`` objects pointing to the downloaded (or reused)
    files.
    """

    if isinstance(flight_lines, str):
        flight_lines = [flight_lines]

    results: list[Path] = []
    for flight_line in flight_lines:
        path, was_downloaded = download_neon_file(
            site_code,
            product_code,
            year_month,
            flight_line,
            out_dir,
        )
        results.append(path)
        action = "downloaded" if was_downloaded else "already present"
        print(f"{flight_line}: {action} at {path}")

    return results
