# Naming Conventions

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
## Canonical Filename Pattern

All files produced by the pipeline use a common token order:

```
NEON_{site}_{YYYYMMDD}_{HHMMSS}_FL{line}_{product}{suffix}.ext
```

- `site` – NEON site code (e.g., `SJER`)
- `YYYYMMDD` and `HHMMSS` – acquisition date and time in UTC
- `FL{line}` – zero‑padded flight line identifier
- `product` – base product name (e.g., `NIS`)
- `suffix` – processing state (see table below)
- `ext` – file extension such as `.img` or `.hdr`

**Regex**

```
^NEON_[A-Z0-9]{4}_\d{8}_\d{6}_FL\d{3}_[A-Za-z0-9]+(?:_radiance|_ancillary|_corrected_envi|_reflectance)\.(?:img|hdr)$
```

## Standard Suffixes

| Suffix | Meaning |
|--------|---------|
| `_radiance` | Raw radiance from HDF5 conversion |
| `_ancillary` | Ancillary data produced with radiance |
| `_corrected_envi` | BRDF/TOPO corrected ENVI image |
| `_reflectance` | Final reflectance product |

## Directory Layout

```
site/
└── YYYYMMDD/
    └── FL###/
        ├── raw/
        │   ├── NEON_SITE_YYYYMMDD_HHMMSS_FL###_NIS_radiance.img
        │   └── NEON_SITE_YYYYMMDD_HHMMSS_FL###_NIS_ancillary.img
        └── derived/
            ├── NEON_SITE_YYYYMMDD_HHMMSS_FL###_NIS_corrected_envi.img
            └── NEON_SITE_YYYYMMDD_HHMMSS_FL###_NIS_corrected_envi.hdr
```

## Common Violations & Fixes

| Violation | Why it matters | Fix |
|-----------|----------------|-----|
| Missing flight line token | Downstream scripts cannot group files | Include `_FL###` before the suffix |
| Wrong suffix for directory (e.g., `_radiance` in `derived/`) | Causes processing confusion | Move file to `raw/` or rename with proper suffix |
| Lower‑case site code | Breaks regex patterns | Use upper‑case site codes |
| Spaces instead of underscores | Parsing fails | Replace spaces with `_` |
<!-- FILLME:END -->
