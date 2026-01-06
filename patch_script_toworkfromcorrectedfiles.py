from __future__ import annotations
from pathlib import Path
from spectralbridge.pipelines import pipeline
import inspect

# ========== CONFIGURATION (hardcoded paths) ==========
BASE_FOLDER = "NIWO_b91"
POLYGON_PATH = Path("/home/jovyan/data-store/cross-sensor-cal/aop_polygons_4_22_2025_analysis_ready_half_diam.geojson")
FLIGHTLINES = ["NEON_D13_NIWO_DP1_L005-1_20230815_reflectance"]
MAX_WORKERS = 1

# ========== SETUP ==========
base_folder = Path(BASE_FOLDER)
polygon_path = POLYGON_PATH
polygon_overwrite = False
polygon_min_overlap = 0.0
polygon_search_buffer_m = 0.0

# Validate polygon path
if polygon_path and not polygon_path.exists():
    print(f"⚠️  Warning: Polygon file not found: {polygon_path}")
    polygon_path = None

# ========== PATCH FUNCTIONS ==========
def stage_download_h5_patched(*args, **kwargs):
    fl = kwargs.get("flight_stem", "UNKNOWN")
    print(f"[carry_on] Skipping download for {fl}")

def stage_export_envi_from_h5_patched(*args, **kwargs):
    fl = kwargs["flight_stem"]
    d = base_folder / fl
    raw_img = d / f"{fl}_envi.img"
    raw_hdr = d / f"{fl}_envi.hdr"
    corr_img = d / f"{fl}_brdfandtopo_corrected_envi.img"
    corr_hdr = d / f"{fl}_brdfandtopo_corrected_envi.hdr"

    if not corr_img.exists() or not corr_hdr.exists():
        raise FileNotFoundError(f"[carry_on] Missing corrected ENVI for {fl}:\n  {corr_img}\n  {corr_hdr}")

    if not raw_img.exists():
        try:
            raw_img.symlink_to(corr_img)
        except OSError:
            raw_img.write_bytes(corr_img.read_bytes())
    if not raw_hdr.exists():
        try:
            raw_hdr.symlink_to(corr_hdr)
        except OSError:
            raw_hdr.write_bytes(corr_hdr.read_bytes())

    print(f"[carry_on] Using corrected ENVI for {fl}")
    return raw_img, raw_hdr

def stage_build_and_write_correction_json_patched(*args, **kwargs):
    import json
    fl = kwargs["flight_stem"]
    d = base_folder / fl
    json_path = d / f"{fl}_brdfandtopo_corrected_envi.json"
    if not json_path.exists():
        payload = {
            "note": "Stub JSON created by carry_on_my_wayward_son; actual BRDF/topo corrections were pre-applied.",
            "flight_stem": fl,
        }
        json_path.write_text(json.dumps(payload, indent=2))
        print(f"[carry_on] Stub correction JSON written for {fl}")
    else:
        print(f"[carry_on] Found existing correction JSON for {fl}")
    return json_path

def stage_apply_brdf_topo_correction_patched(*args, **kwargs):
    fl = kwargs["flight_stem"]
    d = base_folder / fl
    corr_img = d / f"{fl}_brdfandtopo_corrected_envi.img"
    corr_hdr = d / f"{fl}_brdfandtopo_corrected_envi.hdr"
    if not corr_img.exists() or not corr_hdr.exists():
        raise FileNotFoundError(f"[carry_on] Corrected ENVI not found for {fl}:\n  {corr_img}\n  {corr_hdr}")
    print(f"[carry_on] Using precomputed corrected ENVI for {fl}")
    return corr_img, corr_hdr

# ========== APPLY PATCHES ==========
patch_map = {
    "stage_download_h5": stage_download_h5_patched,
    "stage_export_envi_from_h5": stage_export_envi_from_h5_patched,
    "stage_build_and_write_correction_json": stage_build_and_write_correction_json_patched,
    "stage_apply_brdf_topo_correction": stage_apply_brdf_topo_correction_patched,
}

originals = {}
for name, fn in patch_map.items():
    originals[name] = getattr(pipeline, name)
    setattr(pipeline, name, fn)
    print(f"[carry_on] Patched {name}()")

# ========== RUN PIPELINE ==========
results = {}

try:
    for fl in FLIGHTLINES:
        print("\n" + "=" * 80)
        print(f"🚀 Processing {fl}")
        if polygon_path:
            print(f"🌐 Polygon extraction: ENABLED")
            print(f"   Polygon file: {polygon_path}")
        else:
            print(f"🌐 Polygon extraction: DISABLED")
        print("=" * 80 + "\n")

        try:
            kwargs = {
                "base_folder": base_folder,
                "site_code": "MULTI",
                "year_month": "0000-00",
                "product_code": "DP1.30006.001",
                "flight_lines": [fl],
                "max_workers": MAX_WORKERS,
                "engine": "thread",
            }

            # Check if polygon support exists
            sig = inspect.signature(pipeline.go_forth_and_multiply)
            has_polygon_support = "polygon_path" in sig.parameters
            
            if polygon_path and has_polygon_support:
                kwargs.update({
                    "polygon_path": polygon_path,
                    "extraction_mode": "polygon",
                    "polygon_min_overlap": polygon_min_overlap,
                    "polygon_overwrite": polygon_overwrite,
                    "polygon_search_buffer_m": polygon_search_buffer_m,
                })
            elif polygon_path and not has_polygon_support:
                print(f"⚠️  Warning: polygon_path provided but go_forth_and_multiply() doesn't support it.")
                print(f"   Continuing without polygon extraction...")

            res = pipeline.go_forth_and_multiply(**kwargs)
            results[fl] = ("success", res)
            print(f"\n✅ Successfully finished {fl}")
        except Exception as e:
            results[fl] = ("error", str(e))
            print(f"\n❌ Error on {fl}: {e}")
            import traceback
            traceback.print_exc()

finally:
    # Restore original functions
    for name, fn in originals.items():
        setattr(pipeline, name, fn)
        print(f"[carry_on] Restored original {name}()")

# ========== PRINT RESULTS ==========
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
for fl, (status, msg) in results.items():
    if status == "success":
        print(f"✅ {fl}: SUCCESS")
    else:
        print(f"❌ {fl}: ERROR - {msg}")
