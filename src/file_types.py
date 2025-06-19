import re
import os
from pathlib import Path
from typing import Optional, Type, Dict, List

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


class NEONReflectanceENVIFile(DataFile):
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
    def from_components(cls, domain: str, site: str, date: str, time: str, folder: Path) -> "NEONReflectanceENVIFile":
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


class NEONReflectanceBRDFCorrectedENVIFile(DataFile):
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
        files = cls._iter_matching_files(directory)
        results = []

        for p in files:
            match = cls.match(p.name)
            if not match:
                continue
            if correction and match.group("correction") != correction:
                continue
            if suffix and match.group("suffix") != suffix:
                continue
            results.append(cls.from_filename(p))

        return results

class NEONReflectanceResampledENVIFile(DataFile):
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