# Configuration

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
The pipeline is configured through a `config.yaml` file that combines settings for every stage. Values shown below are the defaults unless marked as required.

### Schema

| Key                                   | Type            | Default                | Required |
|---------------------------------------|-----------------|------------------------|----------|
| `base_folder`                         | string          | `"output"`            | yes      |
| `download.site_code`                  | string          | —                      | yes      |
| `download.year_month`                 | string (YYYYMM) | —                      | yes      |
| `download.flight_lines`               | list[string]    | —                      | yes      |
| `download.product_code`               | string          | `"DP1.30006.001"`     | no       |
| `convert.export_ancillary`            | bool            | `true`                 | no       |
| `convert.export_brdf_config`          | bool            | `true`                 | no       |
| `topo_brdf.num_cpus`                  | int             | `8`                    | no       |
| `topo_brdf.file_type`                 | string          | `"envi"`              | no       |
| `topo_brdf.corrections`               | list[string]    | `["topo","brdf"]`    | no       |
| `topo_brdf.bad_bands`                 | list[int]       | `[]`                   | no       |
| `topo_brdf.anc_files`                 | map[string,str] | —                      | conditional† |
| `topo_brdf.export.output_dir`         | string          | `"./"`                | no       |
| `topo_brdf.export.suffix`             | string          | `"_corrected_envi"`   | no       |
| `topo_brdf.export.image`              | bool            | `true`                 | no       |
| `topo_brdf.export.masks`              | bool            | `true`                 | no       |
| `topo_brdf.export.coeffs`             | bool            | `true`                 | no       |
| `resample.method`                     | string          | `"convolution"`       | no       |
| `resample.sensors`                    | list[string]    | `["Landsat_8"]`       | no       |
| `mask.polygon_layer`                  | string          | —                      | no       |
| `mask.raster_crs_override`            | string\|int     | —                      | no       |
| `mask.polygons_crs_override`          | string\|int     | —                      | no       |
| `mask.plot_output`                    | bool            | `false`                | no       |
| `sort.remote_prefix`                  | string          | `""`                  | no       |
| `sort.sync_files`                     | bool            | `true`                 | no       |
| `postprocess.reflectance_offset`      | int             | `0`                    | no       |

† required when `topo_brdf.file_type` is `"envi"`.

### Example

```yaml
base_folder: output

download:
  site_code: NIWO
  year_month: "202008"
  flight_lines: ["FL1", "FL2"]
  product_code: DP1.30006.001

convert:
  export_ancillary: true
  export_brdf_config: true

topo_brdf:
  num_cpus: 8
  file_type: envi
  corrections: ["topo", "brdf"]
  bad_bands: []
  anc_files: {}
  export:
    output_dir: ./corrected
    suffix: _corrected_envi
    image: true
    masks: true
    coeffs: true

resample:
  method: convolution
  sensors: ["Landsat_8"]

mask:
  polygon_layer: polygons.geojson
  plot_output: false

sort:
  remote_prefix: ""
  sync_files: true

postprocess:
  reflectance_offset: 0
```

### CLI overrides

Command-line options override the corresponding entries in `config.yaml`:

- `bin/jefe.py BASE_FOLDER SITE YEAR_MONTH FL1,FL2` sets `base_folder`, `download.site_code`, `download.year_month` and `download.flight_lines`.
- `--polygon_layer_path` → `mask.polygon_layer`
- `--reflectance-offset` → `postprocess.reflectance_offset`
- `--remote-prefix` → `sort.remote_prefix`
- `--no-sync` sets `sort.sync_files` to `false`

<!-- FILLME:END -->
