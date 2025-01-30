from setuptools import setup, find_packages

setup(
    name="your_package_name",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "turtle",
        "numpy",
        "scipy",  
        "pandas",
        "tqdm",
        "h5py",
        "matplotlib",
        "pyvista",
        "pillow",
    ],
)