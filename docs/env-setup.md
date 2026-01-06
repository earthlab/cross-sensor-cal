# Environment Setup

> DO NOT EDIT OUTSIDE MARKERS
<!-- FILLME:START -->
## Conda environment

An example environment file for [Conda](https://docs.conda.io/en/latest/) is
shown below. Save it as `environment.yaml` and create the environment with
`conda env create -f environment.yaml`.

```yaml
name: SpectralBridge
channels:
  - conda-forge
dependencies:
  - python=3.10
  - gdal
  - proj
  - pip
  - pip:
      - ray[default]
```

## uv / pip alternative

Instead of Conda you can install the project with
[`uv`](https://github.com/astral-sh/uv) or plain `pip`:

```bash
uv pip install -r requirements.txt
# or
pip install -r requirements.txt
```

## GDAL, PROJ, and Ray notes

- GDAL and PROJ require native libraries. Installing via the
  `conda-forge` channel usually resolves most platform issues.
- Ray makes heavy use of shared memory. If Ray reports `/dev/shm` errors,
  increase shared memory. For Docker containers use
  `--shm-size=8g` (adjust as needed).

## Known OS quirks

- **macOS**: Homebrew installations of GDAL/PROJ may conflict with Conda.
  Prefer the Conda packages or ensure `brew` paths come after Conda in `PATH`.
- **Windows**: enable long paths (`git config --system core.longpaths true`) to
  avoid checkout errors.

## Preview documentation locally

Run the MkDocs development server from the repository root:

```bash
mkdocs serve
```

Open <http://127.0.0.1:8000> in a browser to view the docs.
<!-- FILLME:END -->
