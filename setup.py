from setuptools import setup, find_packages

setup(
    name="cross_sensor_cal",
    version="0.1",
    packages=find_packages(where="src"),  # Tells pip to look inside src/
    package_dir={"": "src"},              # Maps "src" to the package root
)
