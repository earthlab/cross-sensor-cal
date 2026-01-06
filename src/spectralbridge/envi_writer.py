"""
Portions of this module are adapted from HyTools: Hyperspectral image
processing library (GPLv3).
HyTools Authors: Adam Chlus, Zhiwei Ye, Philip Townsend.
This adapted version is simplified for NEON-only use in cross-sensor-cal.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


def _to_python_scalar(value: Any) -> Union[int, float, str]:
    """Convert numpy scalar types into native Python numbers or strings."""
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _format_envi_list(values: Sequence[Any]) -> str:
    """Format a sequence of values into an ENVI-style list string."""
    formatted: List[str] = []
    for value in values:
        scalar = _to_python_scalar(value)
        if isinstance(scalar, float):
            formatted.append(repr(float(scalar)))
        else:
            formatted.append(str(scalar))
    return "{" + ", ".join(formatted) + "}"


def build_envi_header_text(header_dict: Dict) -> str:
    """
    Build the ENVI .hdr file text from a header_dict.

    Attribution:
        Adapted from hytools.io.envi.write_envi_header() and
        hytools.io.envi.envi_header_from_neon() (GPLv3).
        HyTools Authors: Adam Chlus, Zhiwei Ye, Philip Townsend.
        This simplified version is NEON-only and assumes BSQ float32 output.

    Parameters
    ----------
    header_dict : dict
        A dictionary with at least the following keys:
            'samples'            int
            'lines'              int
            'bands'              int
            'interleave'         str   (must be 'bsq' or 'BSQ')
            'data type'          int   (ENVI code 4 -> float32)
            'byte order'         int   (0 for little endian)
            'wavelength'         list[float]
            'fwhm'               list[float]
            'wavelength units'   str
            'map info'           list or str
            'projection'         str

        You may also include optional keys like:
            'description'
            'transform'
            'ulx'
            'uly'

    Returns
    -------
    str
        A string containing the full .hdr contents to be written to disk.

    Behavior
    --------
        - You MUST write a header that ENVI (and typical GDAL ENVI readers)
          understand.
        - The header should begin with "ENVI" on the first line.
        - Then list key/value lines in ENVI style, for example:

            samples = 1000
            lines   = 500
            bands   = 426
            data type = 4
            interleave = bsq
            byte order = 0
            map info = {UTM, 1.0, 1.0, 500000.0, 4420000.0, 1.0, -1.0, 13, North, WGS-84}
            projection = <...WKT or similar...>
            wavelength = { 400.0, 401.0, ... }
            fwhm = { ... }
            wavelength units = Nanometers

        - You MUST serialize lists (like wavelength, fwhm) into ENVI-style
          curly-brace lists separated by commas.
        - You MUST serialize "map info" consistently with how hytools writes it.
          "map info" in ENVI is written as:

              map info = {UTM, 1.0, 1.0, <ulx>, <uly>, <xres>, <yres>, <utm zone>, North, WGS-84}

          If header_dict['map info'] is already such a list, just join it.

        - Include 'description' in the header if header_dict has it. Wrap it
          ENVI-style like:
              description = { BRDF+topographic corrected reflectance ... }

    Implementation details:
        - Make sure numeric types are converted to Python floats/ints
          before string join, so numpy scalars don't show up weird.
    """

    lines: List[str] = ["ENVI"]

    description = header_dict.get("description")
    if description is not None:
        lines.append(f"description = {{ {description} }}")

    def _req_int(key: str) -> int:
        value = _to_python_scalar(header_dict[key])
        return int(value)

    def _req_str(key: str) -> str:
        value = header_dict[key]
        return str(value)

    samples = _req_int("samples")
    n_lines = _req_int("lines")
    bands = _req_int("bands")
    data_type = _req_int("data type")
    interleave = _req_str("interleave").lower()
    byte_order = _req_int("byte order")

    lines.extend(
        [
            f"samples = {samples}",
            f"lines   = {n_lines}",
            f"bands   = {bands}",
            f"data type = {data_type}",
            f"interleave = {interleave}",
            f"byte order = {byte_order}",
        ]
    )

    map_info = header_dict.get("map info")
    if map_info is None:
        raise KeyError("header_dict must include 'map info'")

    if isinstance(map_info, str):
        map_info_line = map_info.strip()
        if not map_info_line.startswith("{"):
            map_info_line = "{" + map_info_line + "}"
    else:
        map_info_line = _format_envi_list(map_info)  # type: ignore[arg-type]

    lines.append(f"map info = {map_info_line}")

    projection = _req_str("projection")
    lines.append(f"projection = {projection}")

    wavelength = header_dict.get("wavelength")
    if wavelength is None:
        raise KeyError("header_dict must include 'wavelength'")
    lines.append(f"wavelength = {_format_envi_list(wavelength)}")

    fwhm = header_dict.get("fwhm")
    if fwhm is not None:
        lines.append(f"fwhm = {_format_envi_list(fwhm)}")

    wavelength_units = header_dict.get("wavelength units")
    if wavelength_units is None:
        raise KeyError("header_dict must include 'wavelength units'")
    lines.append(f"wavelength units = {wavelength_units}")

    if "reflectance scale factor" in header_dict:
        scale = _to_python_scalar(header_dict["reflectance scale factor"])
        lines.append(f"reflectance scale factor = {scale}")

    if "transform" in header_dict:
        transform = header_dict["transform"]
        lines.append(f"; transform = {_format_envi_list(transform)}")
    if "ulx" in header_dict:
        lines.append(f"; ulx = {float(_to_python_scalar(header_dict['ulx']))}")
    if "uly" in header_dict:
        lines.append(f"; uly = {float(_to_python_scalar(header_dict['uly']))}")

    return "\n".join(lines) + "\n"


class EnviWriter:
    """
    EnviWriter assembles a corrected hyperspectral cube into a BSQ ENVI file.

    Portions adapted from hytools.io.envi (GPLv3).
    HyTools Authors: Adam Chlus, Zhiwei Ye, Philip Townsend.
    This version is NEON-only (no GLT, no NetCDF).

    Usage pattern:

        header = cube.build_envi_header()
        # optionally edit header['description'] before writing

        writer = EnviWriter(out_stem=Path("/path/to/output_stem"),
                            header_dict=header)

        for (ys, ye, xs, xe, chunk) in cube.iter_chunks(...):
            corrected = ... # run apply_topo_correct(), apply_brdf_correct(), etc.
            corrected = corrected.astype("float32", copy=False)
            writer.write_chunk(corrected, ys, xs)

        writer.close()

    After close(), you'll have:
        /path/to/output_stem.img
        /path/to/output_stem.hdr
    """

    def __init__(self, out_stem: Path, header_dict: Dict) -> None:
        """
        Parameters
        ----------
        out_stem : Path
            Output path WITHOUT extension.
            We will create:
                <out_stem>.img
                <out_stem>.hdr

        header_dict : dict
            The header dict produced by NeonCube.build_envi_header(), possibly
            updated to include a human-readable description like
            "BRDF+topographic corrected reflectance".

        Behavior
        --------
        - Extract:
              samples = header_dict["samples"]
              lines   = header_dict["lines"]
              bands   = header_dict["bands"]

        - Allocate a memory-mapped array shaped:
              (bands, lines, samples)
          dtype float32, order='C'.

          This is a BSQ layout: band-major on disk.

        - Store that memmap in self._mm so we can fill it incrementally.
        - Store header_dict and out_stem so we can write the .hdr in close().

        Implementation detail:
            Use numpy.memmap with mode='w+'.
            The filename MUST be f"{out_stem}.img".
        """

        self.out_stem = Path(out_stem)
        self.header_dict = dict(header_dict)

        interleave = str(self.header_dict.get("interleave", "")).lower()
        if interleave != "bsq":
            raise RuntimeError("EnviWriter only supports BSQ interleave")

        try:
            self.samples = int(_to_python_scalar(self.header_dict["samples"]))
            self.lines = int(_to_python_scalar(self.header_dict["lines"]))
            self.bands = int(_to_python_scalar(self.header_dict["bands"]))
        except KeyError as exc:
            raise KeyError("header_dict must contain 'samples', 'lines', and 'bands'") from exc

        img_path = self.out_stem.with_suffix(".img")
        shape = (self.bands, self.lines, self.samples)
        self._mm = np.memmap(img_path, dtype="float32", mode="w+", shape=shape, order="C")

    def write_chunk(self, corrected_chunk: np.ndarray, ys: int, xs: int) -> None:
        """
        Write a corrected reflectance chunk into the BSQ memmap.

        Parameters
        ----------
        corrected_chunk : np.ndarray
            Shape (y_size, x_size, bands), float32.
            This chunk is aligned with cube.data[ys:ys+y_size, xs:xs+x_size, :].

        ys : int
            Row offset (top of chunk in full image coordinates).

        xs : int
            Column offset (left of chunk in full image coordinates).

        Behavior
        --------
        For each band index b:
            self._mm[b,
                     ys:ys+y_size,
                     xs:xs+x_size] = corrected_chunk[:, :, b]

        You MUST do band-major assignment to match BSQ layout.

        This method should NOT flush to disk; flushing happens in close().

        Assumptions:
            - corrected_chunk is already float32
            - corrected_chunk has no-data values already handled
        """

        if corrected_chunk.dtype != np.float32:
            raise RuntimeError("corrected_chunk must have dtype float32")

        if corrected_chunk.ndim != 3:
            raise RuntimeError("corrected_chunk must be a 3D array")

        y_size, x_size, bands_local = corrected_chunk.shape
        if bands_local != self.bands:
            raise RuntimeError(
                f"Chunk band dimension {bands_local} does not match expected {self.bands}"
            )

        y_slice = slice(ys, ys + y_size)
        x_slice = slice(xs, xs + x_size)

        for band_idx in range(self.bands):
            self._mm[band_idx, y_slice, x_slice] = corrected_chunk[:, :, band_idx]

    def close(self) -> None:
        """
        Finalize the ENVI output:

        Behavior
        --------
        - Flush/close the memmap.
        - Write the .hdr file next to the .img file using the
          build_envi_header_text() function.

        Implementation detail:
            - .img path is f"{self.out_stem}.img"
            - .hdr path is f"{self.out_stem}.hdr"
            - build the header string via build_envi_header_text(self.header_dict)
              and write it as ASCII/UTF-8 text.
        """

        self._mm.flush()
        header_text = build_envi_header_text(self.header_dict)
        hdr_path = self.out_stem.with_suffix(".hdr")
        hdr_path.write_text(header_text, encoding="utf-8")
