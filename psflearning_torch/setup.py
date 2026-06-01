from setuptools import setup, find_packages

setup(
    name="psflearning-torch",
    version="0.1.0",
    description="PSF learning toolbox using PyTorch (reimplementation of uiPSF)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "torch>=2.0",
        "matplotlib",
        "scikit-image",
        "tqdm",
        "h5py",
        "czifile",
        "omegaconf",
        "dotted_dict",
        "hdfdict",
        "Pillow",
    ],
)
