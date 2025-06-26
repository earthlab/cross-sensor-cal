import re
import os
from pathlib import Path
from typing import Optional, List
from enum import Enum


class SensorType(str, Enum):
    LANDSAT_5_TM = "Landsat_5_TM"
    LANDSAT_7_ETM_PLUS = "Landsat 7 ETM+"
    LANDSAT_8_OLI = "Landsat 8 OLI"
    LANDSAT_9_OLI_2 = "Landsat 9 OLI-2"
    MICASENSE = "MicaSense"
    MICASENSE_MATCH_TM_ETM = "MicaSense-to-match TM and ETM+"
    MICASENSE_MATCH_OLI = "MicaSense-to-match OLI and OLI-2"


class DataFile:
    pattern: re.Pattern = re.compile('')  # Override in subclasses

    def __init__(self, path: Path):
        self.path = path

    @property
    def name(self):
        return self.path.name

    @property
    def directory(self) -> Path:
        return Path(os.path.dirname(self.path))

    @property
    def file_path(self) -> str:
        return str(self.path)

    @classmethod
    def match(cls, filename: str) -> Optional[re.Match]:
        return cls.pattern.match(filename)

    @classmethod
    def from_filename(cls, path: Path):
        match = cls.match(path.name)
        if not match:
            raise ValueError(f"{cls.__name__} could not parse {path.name}")
        return cls(path, **match.groupdict())

    @classmethod
    def find_in_directory(cls, directory: Path) -> List["DataFile"]:
        return [
            cls.from_filename(p)
            for p in directory.rglob("*")
            if p.is_file() and not p.name.startswith('.') and cls.match(p.name)
        ]


class MaskedFileMixin:
    @classmethod
    def masked_pattern(cls) -> re.Pattern:
        """
        Wrap the class's regex pattern to allow optional '_masked' before the suffix.
        Assumes the original pattern ends with: _<suffix>\.img$
        """
        base_pattern = cls.pattern.pattern
        modified_pattern = base_pattern.replace(r"\.img$", r"(?:_masked)?\.img$")
        return re.compile(modified_pattern)

    @property
    def is_masked(self) -> bool:
        return self.path.stem.endswith("_masked")

    @classmethod
    def match(cls, path: str) -> re.Match | None:
        return cls.masked_pattern().match(path)

    @property
    def masked_path(self) -> Path:
        """
        Return a new Path with '_masked' inserted before the file extension.
        """
        return self.path.with_name(f"{self.path.stem}_masked{self.path.suffix}")

    def masked_version(self):
        """
        Return a new instance of this class pointing to the masked version of the file.
        Assumes the class's __init__ can accept 'path' as a keyword argument.
        """
        return self.__class__.from_filename(path=self.masked_path, **{
            k: v for k, v in self.__dict__.items()
            if k != "path"
        })


class NEONReflectanceFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_(?P<date>\d{8})_(?P<time>\d{6})_reflectance\.h5$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time


class NEONReflectanceENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_(?P<date>\d{8})_(?P<time>\d{6})_reflectance(?:[-_])envi\.img$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time

    @classmethod
    def from_components(cls, domain: str, site: str, date: str, time: str, folder: Path) -> "NEONReflectanceENVIFile":
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_reflectance_envi.img"
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time)


class NEONReflectanceENVHDRFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_(?P<date>\d{8})_(?P<time>\d{6})_reflectance(?:[-_])envi\.hdr$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time

    @classmethod
    def from_components(cls, domain: str, site: str, date: str, time: str, folder: Path) -> "NEONReflectanceENVIHDRFile":
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_reflectance_envi.hdr"
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time)


class NEONReflectanceAncillaryENVIFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_(?P<date>\d{8})_(?P<time>\d{6})_reflectance(?:[-_])ancillary(?:[-_])envi\.img$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time

    @classmethod
    def from_components(cls, domain: str, site: str, date: str, time: str, folder: Path) -> "NEONReflectanceENVIFile":
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_reflectance_ancillary_envi.img"
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time)


class NEONReflectanceAncillaryENVIFileHeader(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_(?P<date>\d{8})_(?P<time>\d{6})_reflectance(?:[-_])ancillary(?:[-_])envi\.hdr$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time

    @classmethod
    def from_components(cls, domain: str, site: str, date: str, time: str, folder: Path) -> "NEONReflectanceAncillaryENVIFileHeader":
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_reflectance_ancillary_envi.img"
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time)


class NEONReflectanceConfigFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_(?P<date>\d{8})_(?P<time>\d{6})_config_(?P<suffix>[a-z]{3,4})\.json$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str, suffix: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.suffix = suffix

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, time: str, suffix: str, folder: Path
    ) -> "NEONReflectanceConfigFile":
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_config_{suffix}.json"
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time, suffix=suffix)

    @classmethod
    def find_in_directory(cls, directory: Path, suffix: str) -> List["NEONReflectanceConfigFile"]:
        return [f for f in super().find_in_directory(directory) if f.suffix == suffix]


class NEONReflectanceBRDFCorrectedENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?P<date>\d{8})_(?P<time>\d{6})_brdf_corrected_(?P<suffix>[a-z]{3,4})\.img$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str, suffix: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.suffix = suffix

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, time: str, suffix: str, folder: Path
    ) -> "NEONReflectanceBRDFCorrectedENVIFile":
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_brdf_corrected_{suffix}.img"
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time, suffix=suffix)

    @classmethod
    def find_in_directory(cls, directory: Path, suffix: str) -> List["NEONReflectanceBRDFCorrectedENVIFile"]:
        return [f for f in super().find_in_directory(directory) if f.suffix == suffix]


class NEONReflectanceBRDFCorrectedENVIHDRFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?P<date>\d{8})_(?P<time>\d{6})_brdf_corrected_(?P<suffix>[a-z]{3,4})\.hdr$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str, suffix: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.suffix = suffix

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, time: str, suffix: str, folder: Path
    ) -> "NEONReflectanceBRDFCorrectedENVIFile":
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_brdf_corrected_{suffix}.hdr"
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time, suffix=suffix)

    @classmethod
    def find_in_directory(cls, directory: Path, suffix: str) -> List["NEONReflectanceBRDFCorrectedENVIFile"]:
        return [f for f in super().find_in_directory(directory) if f.suffix == suffix]


class NEONReflectanceBRDFMaskENVIFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?P<date>\d{8})_(?P<time>\d{6})_brdf_corrected_mask_(?P<suffix>[a-z]{3,4})\.img$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str, suffix: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.suffix = suffix

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, time: str, suffix: str, folder: Path
    ) -> "NEONReflectanceBRDFMaskENVIFile":
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_brdf_corrected_mask_{suffix}.img"
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time, suffix=suffix)

    @classmethod
    def find_in_directory(cls, directory: Path, suffix: str) -> List["NEONReflectanceBRDFMaskENVIFile"]:
        return [f for f in super().find_in_directory(directory) if f.suffix == suffix]


class NEONReflectanceBRDFMaskENVIHDRFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?P<date>\d{8})_(?P<time>\d{6})_brdf_corrected_mask_(?P<suffix>[a-z]{3,4})\.hdr$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str, suffix: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.suffix = suffix

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, time: str, suffix: str, folder: Path
    ) -> "NEONReflectanceBRDFMaskENVIHDRFile":
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_brdf_corrected_mask_{suffix}.hdr"
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time, suffix=suffix)

    @classmethod
    def find_in_directory(cls, directory: Path, suffix: str) -> List["NEONReflectanceBRDFMaskENVIHDRFile"]:
        return [f for f in super().find_in_directory(directory) if f.suffix == suffix]


class NEONReflectanceCoefficientsFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?P<date>\d{8})_(?P<time>\d{6})_reflectance_"
        r"(?P<correction>[a-z]+)_coeffs_(?P<suffix>[a-z0-9]+)\.json$"
    )

    def __init__(
        self,
        path: Path,
        domain: str,
        site: str,
        date: str,
        time: str,
        correction: str,
        suffix: str
    ):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.correction = correction
        self.suffix = suffix

    @classmethod
    def from_components(
        cls,
        domain: str,
        site: str,
        date: str,
        time: str,
        correction: str,
        suffix: str,
        folder: Path
    ) -> "NEONReflectanceCoefficientsFile":
        filename = (
            f"NEON_{domain}_{site}_DP1_{date}_{time}_"
            f"reflectance_{correction}_coeffs_{suffix}.json"
        )
        path = folder / filename
        return cls(path, domain=domain, site=site, date=date, time=time, correction=correction, suffix=suffix)

    @classmethod
    def find_in_directory(
        cls,
        directory: Path,
        correction: Optional[str] = None,
        suffix: Optional[str] = None
    ) -> list["NEONReflectanceCoefficientsFile"]:
        # Use the base class find_in_directory to get all matching files
        all_files = super().find_in_directory(directory)
        
        # Filter based on correction and suffix if provided
        results = []
        for file_obj in all_files:
            if correction and file_obj.correction != correction:
                continue
            if suffix and file_obj.suffix != suffix:
                continue
            results.append(file_obj)
        
        return results

class NEONReflectanceResampledENVIFile(MaskedFileMixin, DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?P<date>\d{8})_(?P<time>\d{6})_resampled_"
        r"(?P<sensor>.+?)_(?P<suffix>[a-z0-9]+)\.img$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str, sensor: str, suffix: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.sensor = sensor
        self.suffix = suffix

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, time: str, sensor: str, suffix: str, folder: Path
    ) -> "NEONReflectanceResampledENVIFile":
        sensor_str = sensor.replace(" ", "_")
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_resampled_{sensor_str}_{suffix}.img"
        path = folder / filename
        return cls(path, domain, site, date, time, sensor, suffix)

    @classmethod
    def find_in_directory(
        cls, directory: Path, sensor: str, suffix: str
    ) -> List["NEONReflectanceResampledENVIFile"]:
        sensor_safe = sensor.replace(" ", "_")
        return [
            f for f in super().find_in_directory(directory)
            if f.sensor == sensor_safe and f.suffix == suffix
        ]

    @classmethod
    def find_all_sensors_in_directory(
            cls, directory: Path, suffix: str
    ) -> List["NEONReflectanceResampledENVIFile"]:
        return [
            f for f in super().find_in_directory(directory)
            if any(f.sensor == sensor.value.replace(' ', '_') and f.suffix == suffix for sensor in SensorType)
        ]


class NEONReflectanceResampledHDRFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?P<date>\d{8})_(?P<time>\d{6})_resampled_"
        r"(?P<sensor>.+?)_(?P<suffix>[a-z0-9]+)\.hdr$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str, sensor: str, suffix: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.sensor = sensor
        self.suffix = suffix

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, time: str, sensor: str, suffix: str, folder: Path
    ) -> "NEONReflectanceResampledHDRFile":
        sensor_str = sensor.replace(" ", "_")
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_resampled_{sensor_str}_{suffix}.hdr"
        path = folder / filename
        return cls(path, domain, site, date, time, sensor, suffix)

    @classmethod
    def find_in_directory(
        cls, directory: Path, sensor: str, suffix: str
    ) -> List["NEONReflectanceResampledHDRFile"]:
        sensor_safe = sensor.replace(" ", "_")
        return [
            f for f in super().find_in_directory(directory)
            if f.sensor == sensor_safe and f.suffix == suffix
        ]


class NEONReflectanceResampledMaskENVIFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?P<date>\d{8})_(?P<time>\d{6})_resampled_mask_"
        r"(?P<sensor>.+?)_(?P<suffix>[a-z0-9]+)\.img$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str, sensor: str, suffix: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.sensor = sensor
        self.suffix = suffix

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, time: str, sensor: str, suffix: str, folder: Path
    ) -> "NEONReflectanceResampledMaskENVIFile":
        sensor_str = sensor.replace(" ", "_")
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_resampled_mask_{sensor_str}_{suffix}.img"
        path = folder / filename
        return cls(path, domain, site, date, time, sensor, suffix)

    @classmethod
    def find_in_directory(
        cls, directory: Path, sensor: str, suffix: str
    ) -> List["NEONReflectanceResampledMaskENVIFile"]:
        sensor_safe = sensor.replace(" ", "_")
        return [
            f for f in super().find_in_directory(directory)
            if f.sensor == sensor_safe and f.suffix == suffix
        ]


class NEONReflectanceResampledMaskHDRFile(DataFile):
    pattern = re.compile(
        r"NEON_(?P<domain>D\d+?)_(?P<site>[A-Z]+?)_DP1_"
        r"(?P<date>\d{8})_(?P<time>\d{6})_resampled_mask_"
        r"(?P<sensor>.+?)_(?P<suffix>[a-z0-9]+)\.hdr$"
    )

    def __init__(self, path: Path, domain: str, site: str, date: str, time: str, sensor: str, suffix: str):
        super().__init__(path)
        self.domain = domain
        self.site = site
        self.date = date
        self.time = time
        self.sensor = sensor
        self.suffix = suffix

    @classmethod
    def from_components(
        cls, domain: str, site: str, date: str, time: str, sensor: str, suffix: str, folder: Path
    ) -> "NEONReflectanceResampledMaskHDRFile":
        sensor_str = sensor.replace(" ", "_")
        filename = f"NEON_{domain}_{site}_DP1_{date}_{time}_resampled_mask_{sensor_str}_{suffix}.hdr"
        path = folder / filename
        return cls(path, domain, site, date, time, sensor, suffix)

    @classmethod
    def find_in_directory(
        cls, directory: Path, sensor: str, suffix: str
    ) -> List["NEONReflectanceResampledMaskHDRFile"]:
        sensor_safe = sensor.replace(" ", "_")
        return [
            f for f in super().find_in_directory(directory)
            if f.sensor == sensor_safe and f.suffix == suffix
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