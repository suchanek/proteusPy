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

This code is in beta.

Eric G. Suchanek, PhD., suchanek@mac.com
"""

import sys

try:
    from setuptools import Command, Extension, setup
except ImportError:
    sys.exit(
        "We need the Python library setuptools to be installed. "
        "Try running: python -m ensurepip"
    )

if "bdist_wheel" in sys.argv:
    try:
        import wheel  # noqa: F401
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


from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Get the version from the version file
version = {}
with open("proteusPy/version.py") as fp:
    exec(fp.read(), version)


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
    requires_python="^3.11",
    packages=["proteusPy"],
    keywords="proteusPy suchanek disulfide",
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas==2.2.1",
        "pyvista[all]",
        "traitlets==5.9.0",
        "jupyter",
        "jupyter_server<2.0",
        "jupyterlab<4.0",
        "seaborn",
        "pillow",
        "tqdm",
        "plotly",
        "datetime",
        "jupyter_bokeh",
        "openai",
        "panel",
        "scikit-learn",
        "gdown",
        "ipykernel",
        "ipygany",
        "nodejs",
        "pytube",
        "grpcio",
        "pip",
        "wget",
        "vtk",
        "kaleido",
        "plotly_express",
        "trame-jupyter-extension",
        "jupyter_contrib_nbextensions",
        "ipywidgets",
        "imageio[ffmpeg]",
    ],
    source="https://github.com/suchanek/proteusPy/",
    project_urls={
        "Documentation": "https://suchanek.github.io/proteusPy/",
        "Source": "https://github.com/suchanek/proteusPy/",
        "Tracker": "https://github.com/suchanek/proteusPy/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3.11",
    ],
    include_package_data=True,
    package_data={
        "proteusPy": [
            "data/*.txt",
            "data/*.py",
            "data/*.json",
            "data/*.csv",
            "data/SS_consensus_class_sext.pkl",
        ]
    },
    exclude_package_data={
        "proteusPy": [
            "data/PDB_all_ss.pkl",
            "data/PDB_SS_ALL_LOADER.pkl",
            "data/PDB_all_ss_dict.pkl",
        ]
    },
    zip_safe=False,
)
