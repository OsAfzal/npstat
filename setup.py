from setuptools import setup, find_packages

setup(
    name="npstat",
    version="0.1.3",
    author="OsAfzal",
    description="Statistical hypothesis testing package",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scipy",
    ],
    python_requires=">=3.6",
)