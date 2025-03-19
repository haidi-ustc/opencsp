from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="opencsp",
    version="0.1.0",
    author="openCSP Development Team",
    author_email="example@example.com",
    description="An open-source crystal structure prediction software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/opencsp",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Materials Science",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "pymatgen": ["pymatgen>=2022.0.0"],
        "ase": ["ase>=3.22.0"],
        "ml": ["scikit-learn>=1.0.0", "torch>=1.9.0"],
        "docs": [
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
        "tests": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
        ],
        "all": [
            "pymatgen>=2022.0.0",
            "ase>=3.22.0",
            "scikit-learn>=1.0.0",
            "torch>=1.9.0",
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
        ],
    },
    #entry_points={
    #    "console_scripts": [
    #        "opencsp=opencsp.cli.main:main",
    #    ],
    #},
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/opencsp/issues",
        "Documentation": "https://opencsp.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/opencsp",
    },
)
