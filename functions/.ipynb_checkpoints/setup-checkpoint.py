from setuptools import setup, find_packages

setup(
    name="pyxtools",
    version="0.1",
    packages=find_packages(),
    install_requires=[
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