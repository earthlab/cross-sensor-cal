import re
from pathlib import Path
from typing import Optional, List, Type
from enum import Enum

try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Backport for Python <3.11




class SensorType(str, Enum):
    LANDSAT_5_TM = "Landsat_5_TM"
    LANDSAT_7_ETM_PLUS = "Landsat_7_ETM+"
    LANDSAT_8_OLI = "Landsat_8_OLI"
    LANDSAT_9_OLI_2 = "Landsat_9_OLI-2"
    MICASENSE = "MicaSense"
    MICASENSE_MATCH_TM_ETM = "MicaSense-to-match_TM_and_ETM+"
    MICASENSE_MATCH_OLI = "MicaSense-to-match_OLI_and_OLI-2"


class DataFile:
    """Base class for NEON data files."""
    pattern: re.Pattern = re.compile("")  # Override in subclasses

    def __init__(self, path: Path):
        self.path: Path = path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def file_path(self) -> str:
        return str(self.path)

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


class NEONReflectanceFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
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
        time: Optional[str] = None,
        suffix: Optional[str] = None,
        tile: Optional[str] = None,
        directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.tile = tile  # Optional
        self.suffix = suffix  # Optional for cases like "ancillary", "corrected", etc.
        self.directional = directional  # True if "_directional" was in filename

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceFile":
        match = cls.match(path.name)
        if not match:
            print(f"⚠️ WARNING: {cls.__name__} could not parse {path.name}, using fallback values.")
            return cls(
                path,
                domain="D00",
                site="UNK",
                date="00000000",
                time=None,
                suffix=None,
                tile=None,
                directional=False
            )
        groups = match.groupdict()
        # Check if '_directional' is in the filename
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str,
        folder: Path, time: Optional[str] = None, 
        suffix: Optional[str] = None, tile: Optional[str] = None, 
        directional: bool = False
    ) -> "NEONReflectanceFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        suffix_part = f"_{suffix}" if suffix else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance{suffix_part}.h5"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, suffix, tile, directional)



class NEONReflectanceENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})"
        r"(?:_(?P<time>\d{6}))?"             # Optional time
        r"(?:_directional)?"                 # Optional "_directional"
        r"_reflectance_envi\.img$"           # Matches ENVI .img files
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.tile = tile  # Optional
        self.directional = directional  # True if "_directional" was present

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceENVIFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceENVIFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance_envi.img"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, tile, directional)



class NEONReflectanceENVIHDRFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})"
        r"(?:_(?P<time>\d{6}))?"             # Optional time
        r"(?:_directional)?"                 # Optional "_directional"
        r"_reflectance_envi\.hdr$"           # Matches ENVI .hdr files
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.tile = tile  # Optional
        self.directional = directional  # True if "_directional" was present

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceENVIHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceENVIHDRFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance_envi.hdr"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, tile, directional)



class NEONReflectanceAncillaryENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})"
        r"(?:_(?P<time>\d{6}))?"                    # Optional time
        r"(?:_directional)?"                        # Optional "_directional"
        r"_reflectance_ancillary_envi\.img$"        # Matches ancillary ENVI .img files
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.tile = tile  # Optional
        self.directional = directional  # True if "_directional" is in filename

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceAncillaryENVIFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceAncillaryENVIFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance_ancillary_envi.img"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, tile, directional)




class NEONReflectanceAncillaryENVIHDRFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})"
        r"(?:_(?P<time>\d{6}))?"                    # Optional time
        r"(?:_directional)?"                        # Optional "_directional"
        r"_reflectance_ancillary_envi\.hdr$"        # Matches ancillary ENVI .hdr files
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.tile = tile  # Optional
        self.directional = directional  # True if "_directional" is in filename

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceAncillaryENVIHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceAncillaryENVIHDRFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance_ancillary_envi.hdr"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, tile, directional)



class NEONReflectanceConfigFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})"
        r"(?:_(?P<time>\d{6}))?"                     # Optional time
        r"(?:_directional)?"                         # Optional "_directional"
        r"_config_(?P<suffix>[a-z]{3,4})\.json$"     # Matches _config_*.json
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str,
        time: Optional[str], suffix: str, tile: Optional[str] = None,
        directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.suffix = suffix
        self.tile = tile  # Optional
        self.directional = directional  # True if "_directional" is present

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceConfigFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, suffix: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceConfigFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_config_{suffix}.json"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, suffix, tile, directional)

    @classmethod
    def find_in_directory(
        cls, directory: Path, suffix: Optional[str] = None
    ) -> List["NEONReflectanceConfigFile"]:
        files = super().find_in_directory(directory)
        return [f for f in files if suffix is None or f.suffix == suffix]


        

class NEONReflectanceBRDFCorrectedENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"              # Optional time
        r"(?:_directional)?"                                 # Optional "_directional"
        r"_brdf_corrected_(?P<suffix>[a-z]{3,4})\.img$"      # Matches _brdf_corrected_*.img
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str, time: Optional[str],
        suffix: str, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.suffix = suffix
        self.tile = tile  # Optional
        self.directional = directional  # True if "_directional" present

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceBRDFCorrectedENVIFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, suffix: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceBRDFCorrectedENVIFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_brdf_corrected_{suffix}.img"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, suffix, tile, directional)

    @classmethod
    def find_in_directory(
        cls, directory: Path, suffix: Optional[str] = None
    ) -> List["NEONReflectanceBRDFCorrectedENVIFile"]:
        files = super().find_in_directory(directory)
        return [f for f in files if suffix is None or f.suffix == suffix]




class NEONReflectanceBRDFCorrectedENVIHDRFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"               # Optional time
        r"(?:_directional)?"                                  # Optional "_directional"
        r"_brdf_corrected_(?P<suffix>[a-z]{3,4})\.hdr$"       # Matches _brdf_corrected_*.hdr
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str, time: Optional[str],
        suffix: str, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.suffix = suffix
        self.tile = tile  # Optional
        self.directional = directional  # True if "_directional" present

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceBRDFCorrectedENVIHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, suffix: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceBRDFCorrectedENVIHDRFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_brdf_corrected_{suffix}.hdr"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, suffix, tile, directional)

    @classmethod
    def find_in_directory(
        cls, directory: Path, suffix: Optional[str] = None
    ) -> List["NEONReflectanceBRDFCorrectedENVIHDRFile"]:
        files = super().find_in_directory(directory)
        return [f for f in files if suffix is None or f.suffix == suffix]




class NEONReflectanceBRDFMaskENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"                  # Optional time
        r"(?:_directional)?"                                     # Optional "_directional"
        r"_brdf_corrected_mask_(?P<suffix>[a-z]{3,4})\.img$"     # Matches .img files with "mask"
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str, time: Optional[str],
        suffix: str, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.suffix = suffix
        self.tile = tile  # Optional
        self.directional = directional  # True if "_directional" present

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceBRDFMaskENVIFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, suffix: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceBRDFMaskENVIFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_brdf_corrected_mask_{suffix}.img"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, suffix, tile, directional)

    @classmethod
    def find_in_directory(
        cls, directory: Path, suffix: Optional[str] = None
    ) -> List["NEONReflectanceBRDFMaskENVIFile"]:
        files = super().find_in_directory(directory)
        return [f for f in files if suffix is None or f.suffix == suffix]



class NEONReflectanceBRDFMaskENVIHDRFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"                   # Optional time
        r"(?:_directional)?"                                      # Optional "_directional"
        r"_brdf_corrected_mask_(?P<suffix>[a-z]{3,4})\.hdr$"      # Matches .hdr files with "mask"
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str, time: Optional[str],
        suffix: str, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.suffix = suffix
        self.tile = tile  # Optional
        self.directional = directional  # True if "_directional" present

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceBRDFMaskENVIHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, suffix: str, folder: Path,
        time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceBRDFMaskENVIHDRFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_brdf_corrected_mask_{suffix}.hdr"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, suffix, tile, directional)

    @classmethod
    def find_in_directory(
        cls, directory: Path, suffix: Optional[str] = None
    ) -> List["NEONReflectanceBRDFMaskENVIHDRFile"]:
        files = super().find_in_directory(directory)
        return [f for f in files if suffix is None or f.suffix == suffix]




class NEONReflectanceCoefficientsFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"                  # Optional time
        r"(?:_directional)?"                                     # Optional "_directional"
        r"_reflectance_(?P<correction>[a-z]+)_coeffs_(?P<suffix>[a-z0-9]+)\.json$"
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str, time: Optional[str],
        correction: str, suffix: str, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.correction = correction
        self.suffix = suffix
        self.tile = tile  # Optional
        self.directional = directional  # True if "_directional" present

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceCoefficientsFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, correction: str, suffix: str,
        folder: Path, time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceCoefficientsFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_reflectance_{correction}_coeffs_{suffix}.json"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, correction, suffix, tile, directional)

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




class NEONReflectanceResampledENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"          # Optional time
        r"(?:_directional)?"                             # Optional "_directional"
        r"_resampled_(?P<sensor>.+?)_(?P<suffix>[a-z0-9]+)\.img$"
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str, time: Optional[str],
        sensor: str, suffix: str, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.sensor = sensor
        self.suffix = suffix
        self.tile = tile
        self.directional = directional  # True if "_directional" was in filename

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceResampledENVIFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, sensor: str, suffix: str,
        folder: Path, time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceResampledENVIFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        sensor_safe = sensor.replace(" ", "_")
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_resampled_{sensor_safe}_{suffix}.img"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, sensor, suffix, tile, directional)

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
        return [
            f for f in files
            if (suffix is None or f.suffix == suffix)
        ]



class NEONReflectanceResampledHDRFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"         # Optional time
        r"(?:_directional)?"                            # Optional "_directional"
        r"_resampled_(?P<sensor>.+?)_(?P<suffix>[a-z0-9]+)\.hdr$"
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str, time: Optional[str],
        sensor: str, suffix: str, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.sensor = sensor
        self.suffix = suffix
        self.tile = tile
        self.directional = directional  # True if "_directional" was in filename

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceResampledHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, sensor: str, suffix: str,
        folder: Path, time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceResampledHDRFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        sensor_safe = sensor.replace(" ", "_")
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_resampled_{sensor_safe}_{suffix}.hdr"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, sensor, suffix, tile, directional)

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
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"         # Optional time
        r"(?:_directional)?"                            # Optional "_directional"
        r"_resampled_mask_(?P<sensor>.+?)_(?P<suffix>[a-z0-9]+)\.img$"
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str, time: Optional[str],
        sensor: str, suffix: str, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.sensor = sensor
        self.suffix = suffix
        self.tile = tile
        self.directional = directional  # True if "_directional" was in filename

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceResampledMaskENVIFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, sensor: str, suffix: str,
        folder: Path, time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceResampledMaskENVIFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        sensor_safe = sensor.replace(" ", "_")
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_resampled_mask_{sensor_safe}_{suffix}.img"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, sensor, suffix, tile, directional)

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
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})(?:_(?P<time>\d{6}))?"         # Optional time
        r"(?:_directional)?"                            # Optional "_directional"
        r"_resampled_mask_(?P<sensor>.+?)_(?P<suffix>[a-z0-9]+)\.hdr$"
    )

    def __init__(
        self, path: Path, domain: str, site: str, date: str, time: Optional[str],
        sensor: str, suffix: str, tile: Optional[str] = None, directional: bool = False
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.sensor = sensor
        self.suffix = suffix
        self.tile = tile
        self.directional = directional  # True if "_directional" was present

    @classmethod
    def from_filename(cls, path: Path) -> "NEONReflectanceResampledMaskHDRFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        groups = match.groupdict()
        directional = "_directional" in path.name
        return cls(path, directional=directional, **groups)

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, sensor: str, suffix: str,
        folder: Path, time: Optional[str] = None, tile: Optional[str] = None,
        directional: bool = False
    ) -> "NEONReflectanceResampledMaskHDRFile":
        tile_part = f"{tile}_" if tile else ""
        time_part = f"_{time}" if time else ""
        directional_part = "_directional" if directional else ""
        sensor_safe = sensor.replace(" ", "_")
        filename = (
            f"NEON_{domain}_{site}_DP1_{tile_part}{date}{time_part}"
            f"{directional_part}_resampled_mask_{sensor_safe}_{suffix}.hdr"
        )
        path = folder / filename
        return cls(path, domain, site, date, time, sensor, suffix, tile, directional)

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

class SpectralDataCSVFile(DataFile):
    pattern = re.compile(
        r"(?P<base>NEON_.*)_spectral_data\.csv$"
    )

    def __init__(self, path: Path, base: str):
        super().__init__(path)
        self.base = base

    @classmethod
    def from_raster_file(cls, raster_file: DataFile) -> "SpectralDataCSVFile":
        base = raster_file.path.stem  # removes .img
        filename = f"{base}_spectral_data.csv"
        return cls(raster_file.path.parent / filename, base=base)

    @classmethod
    def from_filename(cls, path: Path) -> "SpectralDataCSVFile":
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"Filename does not match SpectralDataCSVFile pattern: {path}")
        return cls(path, base=match.group("base"))


class MaskedSpectralCSVFile(DataFile):
    """Represents masked spectral data CSV files like NEON_D13_NIWO_DP1_*_with_mask_and_all_spectra.csv"""
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?:(?P<tile>L\d{3}-\d)_)?"
        r"(?P<date>\d{8})_(?P<time>\d{6})_.*_with_mask_and_all_spectra\.csv$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str, tile: Optional[str] = None):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
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
        r"endmembers_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_"
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
        r"model_best_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_"
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
        r"model_fractions_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_"
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
        r"model_rmse_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_"
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