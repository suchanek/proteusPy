"""setuptools based setup script for proteusPy.

This uses setuptools which is now the standard python mechanism for
installing packages. If you have downloaded and uncompressed the
proteusPy source code, or fetched it from git, for the simplest
installation just type the command::

    python setup.py install

However, you would normally install the latest proteusPy release from
the PyPI archive with::

    python -m pip install proteusPy

For more in-depth instructions, see the installation section of the
proteusPy documentation, linked to from:

http://suchanek.github.io/proteusPy/


Author: Eric G. Suchanek, PhD., suchanek@mac.com
Last Modification: 2025-01-04 14:51:52
"""

# pylint: disable=E0011
# pylint: disable=W0611

import sys
from pathlib import Path

try:
    from setuptools import Command, Extension, find_packages, setup
except ImportError:
    sys.exit(
        "We need the Python library setuptools to be installed. "
        "Try running: python -m ensurepip"
    )

if "bdist_wheel" in sys.argv:
    try:
        import wheel  # pylint: disable=E0011
    except ImportError:
        sys.exit(
            "We need both setuptools AND wheel packages installed "
            "for bdist_wheel to work. Try running: pip install wheel"
        )


def can_import(module_name):
    """Check we can import the requested module."""
    try:
        return __import__(module_name)
    except ImportError:
        return None


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Get the version from the version file
version = {}
with open("proteusPy/_version.py") as fp:
    exec(fp.read(), version)

with open("requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

setup(
    name="proteusPy",
    version=version["__version__"],
    description="proteusPy - Protein Structure Analysis and Modeling Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suchanek/proteusPy/",
    author="Eric G. Suchanek, PhD",
    author_email="suchanek@mac.com",
    license="BSD",
    python_requires=">=3.12",  # corrected from 'requires_python'
    packages=find_packages(),
    keywords="proteusPy suchanek disulfide",
    tests_require=["pytest"],
    test_suite="tests",
    setup_requires=["setuptools", "pytest-runner", "wheel"],
    install_requires=requirements,
    include_package_data=True,
    extras_require={
        "dev": [
            "pytest",
            "pre-commit",
            "coverage",
            "twine",
            "pdoc",
            "wheel",
        ],
        "bio": [
            "Biopython",
        ],
        "pyqt5": [
            "pyqt5",
            "pyvistaqt",
        ],
        "all": [
            "pytest",
            "pre-commit",
            "coverage",
            "twine",
            "pdoc",
            "Biopython",
            "pyqt5",
            "pyvistaqt",
            "wheel",
        ],
    },
    project_urls={
        "Documentation": "https://suchanek.github.io/proteusPy/",
        "Source": "https://github.com/suchanek/proteusPy/",
        "Tracker": "https://github.com/suchanek/proteusPy/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    package_data={
        "proteusPy": [
            "README.md",
            "CHANGELOG.md",
            "data/pdb5rsa.ent",
            "data/ss_completed.txt",
            "data/ss_query.json",
            "data/ss_ids.txt",
            "data/SS_consensus_class_oct.pkl",
            "data/SS_consensus_class_32.pkl",
            "data/2q7q_seqsim.csv",
        ],
    },
    exclude_package_data={
        "proteusPy": [
            "data/PDB_all_ss.pkl",
            "data/PDB_SS_ALL_LOADER.pkl",
            "data/PDB_SS_SUBSET_LOADER.pkl",
        ]
    },
    entry_points={
        "console_scripts": [
            "proteusPy.qt5viewer = proteusPy.qt5viewer:main",
            "proteusPy.DisulfideExtractor = proteusPy.DisulfideExtractor_mp:main",
        ]
    },
    zip_safe=False,
)
