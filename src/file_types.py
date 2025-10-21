from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional, List
from enum import Enum

try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Backport for Python <3.11>


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────

class SensorType(str, Enum):
    LANDSAT_5_TM = "Landsat_5_TM"
    LANDSAT_7_ETM_PLUS = "Landsat_7_ETM+"
    LANDSAT_8_OLI = "Landsat_8_OLI"
    LANDSAT_9_OLI_2 = "Landsat_9_OLI-2"
    MICASENSE = "MicaSense"
    MICASENSE_MATCH_TM_ETM = "MicaSense-to-match_TM_and_ETM+"
    MICASENSE_MATCH_OLI = "MicaSense-to-match_OLI_and_OLI-2"


# ──────────────────────────────────────────────────────────────────────────────
# Base classes / mixins
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataFile:
    """Base class for NEON data files."""

    path: Path
    domain: str | None = None
    site: str | None = None
    date: str | None = None
    time: str | None = None
    product: str | None = None
    sensor: str | None = None
    masked: bool = False
    convolution: bool = False
    _CANON_RE: re.Pattern = field(
        default=re.compile(
            r"^NEON_(?P<domain>D\d{2})_(?P<site>[A-Z0-9]+)_DP1"
            r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
            r"(?:(?P<tile>L\d{3}-\d)_)?"
            r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        ),
        init=False,
        repr=False,
    )

    pattern: ClassVar[re.Pattern] = re.compile("")  # Override in subclasses

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            self.path = Path(self.path)

        fname = self.path.name
        match = self._CANON_RE.match(fname)
        if match:
            groups = match.groupdict()
            self.domain = self.domain or groups.get("domain")
            self.site = self.site or groups.get("site")
            self.date = self.date or groups.get("date")
            time_val = groups.get("time")
            if time_val:
                self.time = self.time or time_val
            product_code = groups.get("product")
            if product_code:
                formatted = f"DP1.{product_code}"
                if not self.product or self.product == "DP1":
                    self.product = formatted

        if not self.product:
            prod_match = re.search(r"DP\d\.\d{5}\.\d{3}", fname)
            if prod_match:
                self.product = prod_match.group(0)

        if not self.product:
            self.product = "DP1"

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def file_path(self) -> Path:
        """Return the concrete :class:`Path` for this data file."""
        return self.path

    @classmethod
    def match(cls, filename: str) -> Optional[re.Match]:
        return cls.pattern.match(filename)

    @classmethod
    def from_filename(cls, path: Path) -> Self:
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        return cls(path, **match.groupdict())  # type: ignore[arg-type]

    @classmethod
    def find_in_directory(cls, directory: Path) -> List[Self]:
        """Recursively find all matching files in a directory."""
        return [
            cls.from_filename(p)
            for p in directory.rglob("*")
            if p.is_file() and not p.name.startswith('.') and cls.match(p.name)
        ]


def _normalize_product_value(product: Optional[str]) -> Optional[str]:
    if not product:
        return None
    return product if product.startswith("DP") else f"DP1.{product}"


def _sensor_from_stem(path: Path) -> Optional[str]:
    stem = path.stem
    match = re.search(r"_resampled(?:_mask)?_(?P<sensor>[^_]+(?:_[^_]+)*)_envi(?:_masked)?$", stem)
    if match:
        return match.group("sensor")
    return None


class MaskedFileMixin:
    """Mixin to handle optional '_masked' suffixes in filenames."""

    @classmethod
    def masked_pattern(cls) -> re.Pattern:
        """
        Adjust the class's regex pattern to allow optional '_masked' before file extensions.
        Assumes the original pattern ends with: \.img$ or \.hdr$
        """
        base_pattern = cls.pattern.pattern
        modified_pattern = re.sub(r"(\\.img$|\\.hdr$)", r"(?:_masked)?\1", base_pattern)
        return re.compile(modified_pattern)

    @property
    def is_masked(self) -> bool:
        """Return True if the filename indicates a masked file."""
        return self.path.stem.endswith("_masked")

    @classmethod
    def match(cls, filename: str) -> Optional[re.Match]:
        """Override match to use the masked-aware pattern."""
        return cls.masked_pattern().match(filename)

    @property
    def masked_path(self) -> Path:
        """Return a new Path with '_masked' inserted before the file extension."""
        if self.is_masked:
            return self.path
        return self.path.with_name(f"{self.path.stem}_masked{self.path.suffix}")

    def masked_version(self) -> Self:
        """Return a new instance pointing to the masked version of the file."""
        return self.__class__.from_filename(self.masked_path)  # type: ignore[call-arg]


# ──────────────────────────────────────────────────────────────────────────────
# H5 reflectance
# ──────────────────────────────────────────────────────────────────────────────

class NEONReflectanceFile(DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})"
        r"(?:_(?P<time>\d{6}))?"            # Optional time
        r"(?:_directional)?"                # Optional "_directional"
        r"_reflectance(?:[-_])?(?P<suffix>[a-z0-9]+)?\.h5$"
    )

    def __init__(
        self,
        path: Path,
        domain: str,
        site: str,
        date: str,
        product: Optional[str] = None,
        time: Optional[str] = None,
        suffix: Optional[str] = None,
        tile: Optional[str] = None,
        directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.product = f"DP1.{product}" if product else "DP1"
        self.time = time
        self.tile = tile
        self.suffix = suffix
        self.directional = directional

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceFile":
        match = cls.match(path.name)
        if not match:
            # fallback (keeps code flowing even if a legacy H5 name slips in)
            return cls(
                path,
                domain="D00",
                site="SITE",
                date="00000000",
                product=None,
                time=None,
                suffix=None,
                tile=None,
                directional="_directional" in path.name
            )
        gd = match.groupdict()
        return cls(path, directional="_directional" in path.name, **gd)  # type: ignore[arg-type]

    @classmethod
    def from_components(
        cls,
        domain: str,
        site: str,
        date: str,
        folder: Path,
        time: Optional[str] = None,
        suffix: Optional[str] = None,
        tile: Optional[str] = None,
        directional: bool = False,
        product: str = "30006.001"
    ) -> "NEONReflectanceFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        suffix_part = f"_{suffix}" if suffix else ""
        filename = (
            f"NEON_{domain}_{site}_DP1.{product}_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance{suffix_part}.h5"
        )
        return cls.from_filename(folder / filename)


# ──────────────────────────────────────────────────────────────────────────────
# ENVI reflectance (uncorrected)
# ──────────────────────────────────────────────────────────────────────────────

class NEONReflectanceENVIFile(DataFile):
    """
    Base (uncorrected) NEON reflectance ENVI image, e.g.
      NEON_D10_R10C_DP1.30006.001_L003-1_20210915_directional_reflectance_envi.img
      NEON_D10_R10C_DP1.30006.001_L003-1_20210915_reflectance_envi.img
      NEON_D10_R10C_DP1.30006.001_L003-1_20210915_000000_directional_reflectance_envi.img
    """
    pattern = re.compile(
        r"""
        ^NEON_
        (?P<domain>D\d{2})_
        (?P<site>[A-Z0-9]{4})_
        (?P<product>DP\d(?:\.\d{5}\.\d{3})?)_
        (?P<tile>L\d{3}-\d)_
        (?P<date>\d{8})
        (?:_(?P<time>\d{6}))?                 # optional _HHMMSS
        _(?:(?P<directional>directional)_)?   # optional "directional_"
        reflectance_envi
        \.(?P<ext>img|hdr)$
        """,
        re.VERBOSE
    )

    @classmethod
    def parse_filename(cls, path: Path):
        m = cls.match(path.name)
        if not m:
            return None
        gd = m.groupdict()
        return {
            "domain": gd["domain"],
            "site": gd["site"],
            "product": gd["product"],
            "tile": gd["tile"],
            "date": gd["date"],
            "time": gd.get("time"),
            "directional": bool(gd.get("directional")),
            "ext": gd["ext"],
        }

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceENVIFile":
        meta = cls.parse_filename(path)
        if not meta:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        obj = cls(path)
        obj.domain = meta["domain"]
        obj.site = meta["site"]
        obj.product = meta["product"]
        obj.tile = meta["tile"]
        obj.date = meta["date"]
        obj.time = meta["time"]
        obj.directional = meta["directional"]
        return obj

    @classmethod
    def from_components(
        cls,
        *,
        domain: str,
        site: str,
        product: str = "DP1.30006.001",
        tile: str,
        date: str,
        time: Optional[str] = None,
        directional: bool = True,
        folder: Path
    ) -> "NEONReflectanceENVIFile":
        parts = ["NEON", domain, site, product, tile, date]
        if time:
            parts.append(time)
        if directional:
            parts.append("directional")
        name = "_".join(parts) + "_reflectance_envi.img"
        return cls.from_filename(folder / name)

    @classmethod
    def find_in_directory(cls, root: Path) -> List["NEONReflectanceENVIFile"]:
        candidates = list(root.rglob("*_reflectance_envi.img"))
        out: List["NEONReflectanceENVIFile"] = []
        for p in candidates:
            try:
                out.append(cls.from_filename(p))
            except ValueError:
                continue
        return out


class NEONReflectanceENVIHDRFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_(?P<product>DP\d(?:\.\d{5}\.\d{3})?)_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?_reflectance_envi\.hdr$"
    )

    def __init__(
        self, path: Path, domain: str, site: str, product: str, date: str,
        time: Optional[str] = None, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.product = product
        self.date = date
        self.time = time
        self.tile = tile
        self.directional = directional

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceENVIHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        return cls(path, directional="_directional" in path.name, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, product: str, date: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None, directional: bool = False
    ) -> "NEONReflectanceENVIHDRFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_{product}_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance_envi.hdr"
        )
        return cls.from_filename(folder / filename)


# --- Backwards-compat alias (tests import the misspelled name) ---
# Some callers import NEONReflectanceENVHDRFile (missing the 'I' in ENVI).
# Keep this alias so both spellings resolve to the same class.
NEONReflectanceENVHDRFile = NEONReflectanceENVIHDRFile

# If you maintain an __all__, expose both:
try:
    __all__.append("NEONReflectanceENVHDRFile")
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# ENVI ancillary
# ──────────────────────────────────────────────────────────────────────────────

class NEONReflectanceAncillaryENVIFile(DataFile):
    """
    Ancillary ENVI image stack, e.g.
      NEON_D10_R10C_DP1.30006.001_L003-1_20210915_directional_reflectance_ancillary_envi.img
      NEON_D10_R10C_DP1.30006.001_L003-1_20210915_reflectance_ancillary_envi.img
    """
    pattern = re.compile(
        r"""
        ^NEON_
        (?P<domain>D\d{2})_
        (?P<site>[A-Z0-9]{4})_
        (?P<product>DP\d(?:\.\d{5}\.\d{3})?)_
        (?P<tile>L\d{3}-\d)_
        (?P<date>\d{8})
        (?:_(?P<time>\d{6}))?
        _(?:(?P<directional>directional)_)?   # optional
        reflectance_ancillary_envi
        \.(?P<ext>img|hdr)$
        """,
        re.VERBOSE
    )

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceAncillaryENVIFile":
        m = cls.match(path.name)
        if not m:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        gd = m.groupdict()
        obj = cls(path)
        obj.domain      = gd["domain"]
        obj.site        = gd["site"]
        obj.product     = gd["product"]
        obj.tile        = gd["tile"]
        obj.date        = gd["date"]
        obj.time        = gd.get("time")
        obj.directional = bool(gd.get("directional"))
        return obj

    @classmethod
    def from_components(
        cls,
        *,
        domain: str,
        site: str,
        product: str = "DP1.30006.001",
        tile: str,
        date: str,
        time: Optional[str] = None,
        directional: bool = True,
        folder: Path,
    ) -> "NEONReflectanceAncillaryENVIFile":
        """
        Build a filename like:
        NEON_{domain}_{site}_{product}_{tile}_{date}[_HHMMSS][_directional]_reflectance_ancillary_envi.img
        """
        parts = ["NEON", domain, site, product, tile, date]
        if time:
            parts.append(time)
        if directional:
            parts.append("directional")
        name = "_".join(parts) + "_reflectance_ancillary_envi.img"
        return cls.from_filename(folder / name)

    @classmethod
    def find_in_directory(cls, root: Path) -> List["NEONReflectanceAncillaryENVIFile"]:
        cands = list(root.rglob("*_reflectance_ancillary_envi.img")) + list(root.rglob("*_reflectance_ancillary_envi.hdr"))
        out: List["NEONReflectanceAncillaryENVIFile"] = []
        for p in cands:
            try:
                out.append(cls.from_filename(p))
            except ValueError:
                continue
        return out



class NEONReflectanceAncillaryENVIHDRFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_(?P<product>DP\d(?:\.\d{5}\.\d{3})?)_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?_reflectance_ancillary_envi\.hdr$"
    )

    def __init__(
        self, path: Path, domain: str, site: str, product: str, date: str,
        time: Optional[str] = None, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.product = product
        self.date = date
        self.time = time
        self.tile = tile
        self.directional = directional

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceAncillaryENVIHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        return cls(path, directional="_directional" in path.name, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, product: str, date: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None, directional: bool = False
    ) -> "NEONReflectanceAncillaryENVIHDRFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_{product}_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance_ancillary_envi.hdr"
        )
        return cls.from_filename(folder / filename)


# ──────────────────────────────────────────────────────────────────────────────
# Config JSON
# ──────────────────────────────────────────────────────────────────────────────

class NEONReflectanceConfigFile(DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_(?P<product>DP\d(?:\.\d{5}\.\d{3})?)_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?"
        r"(?:_000000)?"                               # optional placeholder time
        r"_reflectance_envi_config_envi\.json$"
    )

    def __init__(
        self, path: Path, domain: str, site: str, product: str, date: str,
        time: Optional[str], tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.product = product
        self.date = date
        self.time = time
        self.tile = tile
        self.directional = directional

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceConfigFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        return cls(path, directional="_directional" in path.name, **groups)

    @classmethod
    def from_components(
        cls,
        *,
        domain: str,
        site: str,
        product: str,
        tile: str,
        date: str,
        time: Optional[str],
        directional: bool,
        folder: Path
    ) -> "NEONReflectanceConfigFile":
        parts = ["NEON", domain, site, product, tile, date]
        if time:
            parts.append(time)
        if directional:
            parts.append("directional")
        name = "_".join(parts) + "_reflectance_envi_config_envi.json"
        return cls.from_filename(folder / name)


# ──────────────────────────────────────────────────────────────────────────────
# BRDF-corrected outputs
# ──────────────────────────────────────────────────────────────────────────────

class NEONReflectanceBRDFCorrectedENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?"
        r"_reflectance_brdfandtopo_corrected_(?P<suffix>[a-z0-9_]{3,32})\.img$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, product: Optional[str],
                 time: Optional[str], suffix: str, tile: Optional[str] = None,
                 directional: bool = False):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.product = f"DP1.{product}" if product else "DP1"
        self.time = time
        self.suffix = suffix
        self.tile = tile
        self.directional = directional

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceBRDFCorrectedENVIFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        gd = match.groupdict()
        return cls(path, directional="_directional" in path.name, **gd)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, suffix: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False, product: str = "30006.001"
    ) -> "NEONReflectanceBRDFCorrectedENVIFile":
        suffix_clean = suffix.replace("corrected", "").replace("__", "_").strip("_")
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1.{product}_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance_brdfandtopo_corrected_{suffix_clean}.img"
        )
        return cls.from_filename(folder / filename)

    @classmethod
    def find_in_directory(cls, root: Path, include_masks: bool = False) -> List["NEONReflectanceBRDFCorrectedENVIFile"]:
        pats = [r"*reflectance_brdfandtopo_corrected_*.img"]
        if include_masks:
            pats.append(r"*reflectance_brdfandtopo_corrected_mask_*.img")
        cands: List[Path] = []
        for pat in pats:
            cands.extend(root.rglob(pat))
        out: List["NEONReflectanceBRDFCorrectedENVIFile"] = []
        for p in cands:
            try:
                out.append(cls.from_filename(p))
            except ValueError:
                continue
        return out


class NEONReflectanceBRDFCorrectedENVIHDRFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?"
        r"_reflectance_brdfandtopo_corrected_(?P<suffix>[a-z0-9_]{3,32})\.hdr$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, product: Optional[str],
                 time: Optional[str], suffix: str, tile: Optional[str] = None,
                 directional: bool = False):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.product = f"DP1.{product}" if product else "DP1"
        self.time = time
        self.suffix = suffix
        self.tile = tile
        self.directional = directional

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceBRDFCorrectedENVIHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        gd = match.groupdict()
        return cls(path, directional="_directional" in path.name, **gd)

    @classmethod
    def find_in_directory(cls, root: Path) -> List["NEONReflectanceBRDFCorrectedENVIHDRFile"]:
        cands = list(root.rglob("*reflectance_brdfandtopo_corrected_*.hdr"))
        out: List["NEONReflectanceBRDFCorrectedENVIHDRFile"] = []
        for p in cands:
            try:
                out.append(cls.from_filename(p))
            except ValueError:
                continue
        return out


class NEONReflectanceBRDFMaskENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?"
        r"_reflectance_brdfandtopo_corrected_mask_(?P<suffix>[a-z0-9_]{3,32})\.img$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str,
                 product: Optional[str], time: Optional[str],
                 suffix: str, tile: Optional[str] = None,
                 directional: bool = False):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.product = f"DP1.{product}" if product else "DP1"
        self.time = time
        self.suffix = suffix
        self.tile = tile
        self.directional = directional

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceBRDFMaskENVIFile":
        m = cls.pattern.match(path.name)
        if not m:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        gd = m.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **gd)

    @classmethod
    def from_components(
        cls,
        domain: str,
        site: str,
        date: str,
        suffix: str,
        folder: Path,
        *,
        product: str = "30006.001",
        time: Optional[str] = None,
        tile: Optional[str] = None,
        directional: bool = False,
    ) -> "NEONReflectanceBRDFMaskENVIFile":
        # match your corrected-image builder: strip “corrected”, collapse underscores, trim
        suffix_clean = suffix.replace("corrected", "").replace("__", "_").strip("_")
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        dir_part = "_directional" if directional else ""
        name = (
            f"NEON_{domain}_{site}_DP1.{product}_{tile_part}{date}{time_part}"
            f"{dir_part}_reflectance_brdfandtopo_corrected_mask_{suffix_clean}.img"
        )
        path = folder / name
        # We can instantiate directly (avoids strict regex dependency during writes)
        return cls(path, domain, site, date, product, time, suffix_clean, tile, directional)

    @classmethod
    def find_in_directory(cls, root: Path) -> List["NEONReflectanceBRDFMaskENVIFile"]:
        cands = list(root.rglob("*reflectance_brdfandtopo_corrected_mask_*.img"))
        out: List["NEONReflectanceBRDFMaskENVIFile"] = []
        for p in cands:
            try:
                out.append(cls.from_filename(p))
            except ValueError:
                continue
        return out



class NEONReflectanceBRDFMaskENVIHDRFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?"
        r"_reflectance_brdfandtopo_corrected_mask_(?P<suffix>[a-z0-9_]{3,32})\.hdr$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str,
                 product: Optional[str], time: Optional[str],
                 suffix: str, tile: Optional[str] = None,
                 directional: bool = False):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.product = f"DP1.{product}" if product else "DP1"
        self.time = time
        self.suffix = suffix
        self.tile = tile
        self.directional = directional

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceBRDFMaskENVIHDRFile":
        m = cls.pattern.match(path.name)
        if not m:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        gd = m.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **gd)

    @classmethod
    def from_components(
        cls,
        domain: str,
        site: str,
        date: str,
        suffix: str,
        folder: Path,
        *,
        product: str = "30006.001",
        time: Optional[str] = None,
        tile: Optional[str] = None,
        directional: bool = False,
    ) -> "NEONReflectanceBRDFMaskENVIHDRFile":
        suffix_clean = suffix.replace("corrected", "").replace("__", "_").strip("_")
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        dir_part = "_directional" if directional else ""
        name = (
            f"NEON_{domain}_{site}_DP1.{product}_{tile_part}{date}{time_part}"
            f"{dir_part}_reflectance_brdfandtopo_corrected_mask_{suffix_clean}.hdr"
        )
        path = folder / name
        return cls(path, domain, site, date, product, time, suffix_clean, tile, directional)

    @classmethod
    def find_in_directory(cls, root: Path) -> List["NEONReflectanceBRDFMaskENVIHDRFile"]:
        cands = list(root.rglob("*reflectance_brdfandtopo_corrected_mask_*.hdr"))
        out: List["NEONReflectanceBRDFMaskENVIHDRFile"] = []
        for p in cands:
            try:
                out.append(cls.from_filename(p))
            except ValueError:
                continue
        return out



# ──────────────────────────────────────────────────────────────────────────────
# Coefficients
# ──────────────────────────────────────────────────────────────────────────────

class NEONReflectanceCoefficientsFile(DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?"
        r"_reflectance_(?P<correction>[a-z]+)_coeffs_(?P<suffix>[a-z0-9_]+)\.json$"
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str, product: Optional[str],
        time: Optional[str], correction: str, suffix: str,
        tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.product = f"DP1.{product}" if product else "DP1"
        self.time = time
        self.correction = correction
        self.suffix = suffix
        self.tile = tile
        self.directional = directional

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceCoefficientsFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        return cls(path, directional="_directional" in path.name, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, correction: str, suffix: str,
        folder: Path, time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False, product: str = "30006.001"
    ) -> "NEONReflectanceCoefficientsFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1.{product}_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance_{correction}_coeffs_{suffix}.json"
        )
        return cls.from_filename(folder / filename)

    @classmethod
    def find_in_directory(
        cls, directory: Path, correction: Optional[str] = None, suffix: Optional[str] = None
    ) -> List["NEONReflectanceCoefficientsFile"]:
        files = super().find_in_directory(directory)
        return [
            f for f in files
            if (correction is None or f.correction == correction)
            and (suffix is None or f.suffix == suffix)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Resampled outputs
# ──────────────────────────────────────────────────────────────────────────────

class NEONReflectanceResampledENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?"
        r"_resampled_(?P<sensor>.+?)_(?P<suffix>[a-z0-9_]+)\.img$"
    )

    def __init__(
        self,
        path: Path,
        domain: str,
        site: str,
        date: str,
        product: Optional[str] = None,
        time: Optional[str] = None,
        sensor: Optional[str] = None,
        suffix: str = "envi",
        tile: Optional[str] = None,
        directional: bool = False,
    ):
        norm_product = _normalize_product_value(product)
        super().__init__(
            path=path,
            domain=domain,
            site=site,
            date=date,
            time=time,
            product=norm_product,
            sensor=sensor,
        )
        self.suffix = suffix
        self.tile = tile
        self.directional = directional
        self.masked = False

        if not self.sensor:
            inferred = _sensor_from_stem(self.path)
            if inferred:
                self.sensor = inferred

        if not norm_product and (not self.product or self.product == "DP1"):
            self.product = "resampled"

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceResampledENVIFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        product = groups.pop("product", None)
        return cls(
            path,
            product=product,
            directional="_directional" in path.name,
            **groups,
        )

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, sensor: str, suffix: str,
        folder: Path, time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False, product: str = "30006.001"
    ) -> "NEONReflectanceResampledENVIFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        sensor_safe = sensor.replace(" ", "_")
        filename = (
            f"NEON_{domain}_{site}_DP1.{product}_{tile_part}{date}{time_part}"
            f"{directional_part}_resampled_{sensor_safe}_{suffix}.img"
        )
        return cls.from_filename(folder / filename)

    @classmethod
    def find_in_directory(
        cls, directory: Path, sensor: Optional[str] = None, suffix: Optional[str] = None
    ) -> List["NEONReflectanceResampledENVIFile"]:
        files = super().find_in_directory(directory)
        return [
            f for f in files
            if (sensor is None or f.sensor == sensor)
            and (suffix is None or f.suffix == suffix)
        ]

    @classmethod
    def find_all_sensors_in_directory(
        cls, directory: Path, suffix: Optional[str] = None
    ) -> List["NEONReflectanceResampledENVIFile"]:
        files = super().find_in_directory(directory)
        return [f for f in files if (suffix is None or f.suffix == suffix)]


class NEONReflectanceResampledHDRFile(DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?"
        r"_resampled_(?P<sensor>.+?)_(?P<suffix>[a-z0-9_]+)\.hdr$"
    )

    def __init__(
        self,
        path: Path,
        domain: str,
        site: str,
        date: str,
        product: Optional[str] = None,
        time: Optional[str] = None,
        sensor: Optional[str] = None,
        suffix: str = "envi",
        tile: Optional[str] = None,
        directional: bool = False,
    ):
        norm_product = _normalize_product_value(product)
        super().__init__(
            path=path,
            domain=domain,
            site=site,
            date=date,
            time=time,
            product=norm_product,
            sensor=sensor,
        )
        self.suffix = suffix
        self.tile = tile
        self.directional = directional
        self.masked = False

        if not self.sensor:
            inferred = _sensor_from_stem(self.path)
            if inferred:
                self.sensor = inferred

        if not norm_product and (not self.product or self.product == "DP1"):
            self.product = "resampled"

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceResampledHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        product = groups.pop("product", None)
        return cls(
            path,
            product=product,
            directional="_directional" in path.name,
            **groups,
        )

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, sensor: str, suffix: str,
        folder: Path, time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False, product: str = "30006.001"
    ) -> "NEONReflectanceResampledHDRFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        sensor_safe = sensor.replace(" ", "_")
        filename = (
            f"NEON_{domain}_{site}_DP1.{product}_{tile_part}{date}{time_part}"
            f"{directional_part}_resampled_{sensor_safe}_{suffix}.hdr"
        )
        return cls.from_filename(folder / filename)

    @classmethod
    def find_in_directory(
        cls, directory: Path, sensor: Optional[str] = None, suffix: Optional[str] = None
    ) -> List["NEONReflectanceResampledHDRFile"]:
        files = super().find_in_directory(directory)
        return [
            f for f in files
            if (sensor is None or f.sensor == sensor)
            and (suffix is None or f.suffix == suffix)
        ]


class NEONReflectanceResampledMaskENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?"
        r"_resampled_mask_(?P<sensor>.+?)_(?P<suffix>[a-z0-9_]+)\.img$"
    )

    def __init__(
        self,
        path: Path,
        domain: str,
        site: str,
        date: str,
        product: Optional[str] = None,
        time: Optional[str] = None,
        sensor: Optional[str] = None,
        suffix: str = "envi",
        tile: Optional[str] = None,
        directional: bool = False,
    ):
        norm_product = _normalize_product_value(product)
        super().__init__(
            path=path,
            domain=domain,
            site=site,
            date=date,
            time=time,
            product=norm_product,
            sensor=sensor,
        )
        self.suffix = suffix
        self.tile = tile
        self.directional = directional
        self.masked = True

        if not self.sensor:
            inferred = _sensor_from_stem(self.path)
            if inferred:
                self.sensor = inferred

        if not norm_product and (not self.product or self.product == "DP1"):
            self.product = "resampled_mask"

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceResampledMaskENVIFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        product = groups.pop("product", None)
        return cls(
            path,
            product=product,
            directional="_directional" in path.name,
            **groups,
        )

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, sensor: str, suffix: str,
        folder: Path, time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False, product: str = "30006.001"
    ) -> "NEONReflectanceResampledMaskENVIFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        sensor_safe = sensor.replace(" ", "_")
        filename = (
            f"NEON_{domain}_{site}_DP1.{product}_{tile_part}{date}{time_part}"
            f"{directional_part}_resampled_mask_{sensor_safe}_{suffix}.img"
        )
        return cls.from_filename(folder / filename)

    @classmethod
    def find_in_directory(
        cls, directory: Path, sensor: Optional[str] = None, suffix: Optional[str] = None
    ) -> List["NEONReflectanceResampledMaskENVIFile"]:
        files = super().find_in_directory(directory)
        return [
            f for f in files
            if (sensor is None or f.sensor == sensor)
            and (suffix is None or f.suffix == suffix)
        ]


class NEONReflectanceResampledMaskHDRFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"
        r"(?:_directional)?"
        r"_resampled_mask_(?P<sensor>.+?)_(?P<suffix>[a-z0-9_]+)\.hdr$"
    )

    def __init__(
        self,
        path: Path,
        domain: str,
        site: str,
        date: str,
        product: Optional[str] = None,
        time: Optional[str] = None,
        sensor: Optional[str] = None,
        suffix: str = "envi",
        tile: Optional[str] = None,
        directional: bool = False,
    ):
        norm_product = _normalize_product_value(product)
        super().__init__(
            path=path,
            domain=domain,
            site=site,
            date=date,
            time=time,
            product=norm_product,
            sensor=sensor,
        )
        self.suffix = suffix
        self.tile = tile
        self.directional = directional
        self.masked = True

        if not self.sensor:
            inferred = _sensor_from_stem(self.path)
            if inferred:
                self.sensor = inferred

        if not norm_product and (not self.product or self.product == "DP1"):
            self.product = "resampled_mask"

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceResampledMaskHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        product = groups.pop("product", None)
        return cls(
            path,
            product=product,
            directional="_directional" in path.name,
            **groups,
        )

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, sensor: str, suffix: str,
        folder: Path, time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False, product: str = "30006.001"
    ) -> "NEONReflectanceResampledMaskHDRFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        sensor_safe = sensor.replace(" ", "_")
        filename = (
            f"NEON_{domain}_{site}_DP1.{product}_{tile_part}{date}{time_part}"
            f"{directional_part}_resampled_mask_{sensor_safe}_{suffix}.hdr"
        )
        return cls.from_filename(folder / filename)

    @classmethod
    def find_in_directory(
        cls, directory: Path, sensor: Optional[str] = None, suffix: Optional[str] = None
    ) -> List["NEONReflectanceResampledMaskHDRFile"]:
        files = super().find_in_directory(directory)
        return [
            f for f in files
            if (sensor is None or f.sensor == sensor)
            and (suffix is None or f.suffix == suffix)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# CSV + TIF helpers
# ──────────────────────────────────────────────────────────────────────────────

class SpectralDataParquetFile(DataFile):
    pattern = re.compile(r"^(?P<base>NEON_.*)_spectral_data\.parquet$")

    def __init__(self, path: Path, base: str):
        super().__init__(path)
        self.base = base

    @classmethod
    def from_raster_file(cls, raster_file: DataFile) -> "SpectralDataParquetFile":
        base = raster_file.path.stem
        filename = f"{base}_spectral_data.parquet"
        output_directory = raster_file.path.parent.parent / "full_extracted_pixels"
        output_directory.mkdir(parents=True, exist_ok=True)
        return cls(output_directory / filename, base=base)

    @classmethod
    def from_filename(cls, path: Path) -> "SpectralDataParquetFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"Filename does not match SpectralDataParquetFile pattern: {path}")
        return cls(path, base=match.group("base"))


class MaskedSpectralCSVFile(DataFile):
    """Represents masked spectral data CSV files like NEON_D13_NIWO_DP1_*_with_mask_and_all_spectra.csv"""
    pattern = re.compile(
        r"^NEON_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_DP1"
        r"(?:\.(?P<product>\d{5}\.\d{3}))?_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})_(?P<time>\d{6})_.*_with_mask_and_all_spectra\.csv$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str,
                 product: Optional[str] = None, tile: Optional[str] = None):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.product = f"DP1.{product}" if product else "DP1"
        self.tile = tile

    @classmethod
    def from_components(cls, domain: str, site: str, date: str, time: str,
                        base_name: str, folder: Path) -> "MaskedSpectralCSVFile":
        filename = f"{base_name}_with_mask_and_all_spectra.csv"
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time)


class EndmembersCSVFile(DataFile):
    """Represents endmembers CSV output from unmixing process"""
    pattern = re.compile(
        r"^endmembers_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_"
        r"(?P<date>\d{8})_(?P<time>\d{6})\.csv$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time

    @classmethod
    def from_signatures_file(cls, signatures_file: MaskedSpectralCSVFile,
                             output_dir: Path) -> "EndmembersCSVFile":
        filename = f"endmembers_{signatures_file.domain}_{signatures_file.site}_{signatures_file.date}_{signatures_file.time}.csv"
        path = output_dir / filename
        return cls(path, domain=signatures_file.domain, site=signatures_file.site,
                   date=signatures_file.date, time=signatures_file.time)


class UnmixingModelBestTIF(DataFile):
    """Represents model_best.tif output from unmixing"""
    pattern = re.compile(
        r"^model_best_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_"
        r"(?P<date>\d{8})_(?P<time>\d{6})\.tif$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time

    @classmethod
    def from_signatures_file(cls, signatures_file: MaskedSpectralCSVFile,
                             output_dir: Path) -> "UnmixingModelBestTIF":
        filename = f"model_best_{signatures_file.domain}_{signatures_file.site}_{signatures_file.date}_{signatures_file.time}.tif"
        path = output_dir / filename
        return cls(path, domain=signatures_file.domain, site=signatures_file.site,
                   date=signatures_file.date, time=signatures_file.time)


class UnmixingModelFractionsTIF(DataFile):
    """Represents model_fractions.tif output from unmixing"""
    pattern = re.compile(
        r"^model_fractions_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_"
        r"(?P<date>\d{8})_(?P<time>\d{6})\.tif$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time

    @classmethod
    def from_signatures_file(cls, signatures_file: MaskedSpectralCSVFile,
                             output_dir: Path) -> "UnmixingModelFractionsTIF":
        filename = f"model_fractions_{signatures_file.domain}_{signatures_file.site}_{signatures_file.date}_{signatures_file.time}.tif"
        path = output_dir / filename
        return cls(path, domain=signatures_file.domain, site=signatures_file.site,
                   date=signatures_file.date, time=signatures_file.time)


class UnmixingModelRMSETIF(DataFile):
    """Represents model_rmse.tif output from unmixing"""
    pattern = re.compile(
        r"^model_rmse_(?P<domain>D\d+?)_(?P<site>[A-Z0-9]+?)_"
        r"(?P<date>\d{8})_(?P<time>\d{6})\.tif$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time

    @classmethod
    def from_signatures_file(cls, signatures_file: MaskedSpectralCSVFile,
                             output_dir: Path) -> "UnmixingModelRMSETIF":
        filename = f"model_rmse_{signatures_file.domain}_{signatures_file.site}_{signatures_file.date}_{signatures_file.time}.tif"
        path = output_dir / filename
        return cls(path, domain=signatures_file.domain, site=signatures_file.site,
                   date=signatures_file.date, time=signatures_file.time)
