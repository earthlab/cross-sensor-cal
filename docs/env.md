# Environment

| Component | Known-good |
|---|---|
| Python   | 3.10–3.12 |
| OS       | macOS 13+, Ubuntu 22.04+ |
| Core libs| numpy, rasterio, gdal, ray, xarray, pandas |

## Setup (venv)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install spectralbridge
```

## Setup (conda)
```bash
conda create -n spectralbridge python=3.11 -y
conda activate spectralbridge
pip install -U pip
pip install spectralbridge
```
