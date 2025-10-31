# Schemas

> **When do I need this?** When validating outputs or writing downstream tooling that expects consistent columns and metadata.

## Purpose
Document the shape of artifacts emitted by Stage 5 (Parquet export) and Stage 6 (merge) so you can trust what [Outputs](../pipeline/outputs.md) deliver.

## Inputs
- Schema JSON files bundled with the project (see `schemas/` in the repo)
- Sample Parquet files from [Parquet export](../pipeline/stages.md#5-parquet-export) or [Merge](../pipeline/stages.md#6-duckdb-merge)

## Outputs
Validation reports confirming column presence, dtypes, and metadata blocks for ENVI-derived tables.

## Run it
```bash
python scripts/validate_schema.py parquet/demo_brdfandtopo_corrected_envi.parquet schemas/parquet_brdfandtopo.json
```

```python
import json
import pyarrow.parquet as pq

with open("schemas/parquet_brdfandtopo.json", "r", encoding="utf-8") as fp:
    spec = json.load(fp)
meta = pq.read_table("parquet/demo_brdfandtopo_corrected_envi.parquet")
missing = set(spec["columns"]) - set(meta.schema.names)
print(f"Missing columns: {sorted(missing)}")
```

## Pitfalls
- Always match schema files to the correct stage; merged tables include joined metadata absent in Stage 5 outputs.
- Case-sensitive column names can fail equality checks—normalize before comparing.
- When adding sensors, update both the schema and the [Troubleshooting](../troubleshooting.md) page to reflect new failure modes.
