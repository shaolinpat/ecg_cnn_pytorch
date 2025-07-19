from setuptools import setup, find_packages

# Load __version__ from ecg_cnn/__init__.py
version_ns = {}
with open("ecg_cnn/__init__.py") as f:
    exec(f.read(), version_ns)

setup(
    name="ecg_cnn",
    version=version_ns["__version__"],
    description="CNN-based ECG classifier using PTB-XL and PyTorch",
    author="Patrick Beach",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.8",
)
