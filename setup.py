from setuptools import setup, find_packages

setup(
    name="npstat",
    version="0.3",
    author="OsAfzal",
    description="Statistical hypothesis testing package",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib"
    ],
    python_requires=">=3.6",

)




