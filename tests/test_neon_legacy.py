import h5py

from cross_sensor_cal.io.neon_legacy import detect_legacy_neon_schema, resolve_neon_paths


def _mk_h5(struct):
    f = h5py.File(b":memory:", "w")
    for path, payload in struct.items():
        grp = f
        parts = path.strip("/").split("/")
        for p in parts[:-1]:
            grp = grp.require_group(p)
        if isinstance(payload, (list, tuple)):
            grp.create_dataset(parts[-1], data=payload)
        else:
            grp.create_dataset(parts[-1], data=payload)
    return f


def test_detect_legacy_true():
    f = _mk_h5({
        "Reflectance/Reflectance_Data": [0],
        "Reflectance/Metadata/Spectral_Data/Wavelength": [1.0, 2.0],
        "Reflectance/Metadata/Spectral_Data/FWHM": [0.1, 0.1],
        "Reflectance/Metadata/to-sun_Zenith_Angle": [30.0],
    })
    try:
        assert detect_legacy_neon_schema(f) is True
        p = resolve_neon_paths(f)
        assert p.wavelength.endswith("Wavelength")
    finally:
        f.close()


def test_detect_legacy_false():
    f = _mk_h5({
        "Reflectance/Reflectance_Data": [0],
        "Reflectance/Metadata/Spectral_Data/wavelength": [1.0, 2.0],
        "Reflectance/Metadata/Spectral_Data/fwhm": [0.1, 0.1],
        "Reflectance/Metadata/to-sun_zenith_angle": [30.0],
    })
    try:
        assert detect_legacy_neon_schema(f) is False
        p = resolve_neon_paths(f)
        assert p.wavelength.endswith("wavelength")
    finally:
        f.close()
