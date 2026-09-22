from setuptools import setup, find_packages
setup(
    name="pyto",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=["numpy>=2.0.0"],
)