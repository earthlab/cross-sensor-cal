from setuptools import setup, find_packages

setup(
    name="EarthLabSpectral",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.1',
        'spectral>=0.22',
        'geopandas>=0.8.0',  # Specify minimum versions based on your requirements
        'rasterio>=1.1.5',
        'pandas>=1.0.5',
        'matplotlib>=3.2.2',
        'scikit-learn>=0.23.1',  # For GradientBoostingRegressor
        'h5py>=2.10.0',
        'requests>=2.24.0',
        'ray>=1.0.0',  # Specify versions as needed
        # Add other dependencies as needed
    ],
    # Additional metadata about your package
    author="Ty Tuff, Erick Verleye",
    author_email="ty.tuff@colorado.edu",
    description="EarthLabSpectral provides tools for cross-sensor spectral calibration and resampling. "
                "Designed for remote sensing data analysis, it supports various satellite data formats, "
                "facilitating the comparison and integration of multi-sensor datasets for Earth observation research.",
    url="https://github.com/earthlab/cross-sensor-cal",
    # List additional URLs that are relevant to your project as a dict
    project_urls={
        "Documentation": "https://github.com/earthlab/cross-sensor-cal/blob/main/vignette.md",
        "Source Code": "https://github.com/earthlab/cross-sensor-cal",
        "Issue Tracker": "https://github.com/earthlab/cross-sensor-cal/issues",
    },
)

