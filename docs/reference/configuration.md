# Configuration

Most users can rely on default configuration, but the pipeline allows fine-grained control over processing stages, performance, and file handling.

This page documents the available configuration parameters and how they affect pipeline behavior.

---

## Configuration sources

1. Command-line options (highest priority)  
2. Environment variables  
3. Default internal settings  

---

## Common settings

### `base-folder`

Directory where:

- downloaded HDF5 tiles  
- ENVI exports  
- Parquet tables  
- QA artifacts  

are written.

---

### `engine`

Execution backend:

- `thread` (default)  
- `ray` (distributed/hyperparallel workflows)

---

### `max-workers`

Controls concurrency in:

- ENVI export  
- BRDF and topo correction  
- Parquet extraction  

Use cautiously when memory is limited.

---

### `start-at` and `end-at`

Define subsets of the pipeline to run.

Example:

```bash
--start-at brdf --end-at convolution
Environment variables
VariableMeaning
CSCAL_TMPDIROverride temporary directory
CSCAL_LOGLEVELSet logging verbosity
CSCAL_RAY_ADDRESSUse an existing Ray cluster
Advanced configuration
These settings primarily matter for large-scale workflows:
chunk sizes for Parquet extraction
memory thresholds for Ray worker processes
default CRS assignments
sensor SRF paths
Details of internal architecture appear in the Developer section.
Next steps
JSON schemas
Validation metrics
Pipeline stages

---
