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
"""
import sys
import os

try:
    from setuptools import setup
    from setuptools import Command
    from setuptools import Extension
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

# Make sure we have the right Python version.
MIN_PY_VER = (3, 9)
if sys.version_info[:2] < MIN_PY_VER:
    sys.stderr.write(
        ("ERROR: proteusPy requires Python %i.%i or later. " % MIN_PY_VER)
        + ("Python %d.%d detected.\n" % sys.version_info[:2])
    )
    sys.exit(1)

def can_import(module_name):
    """Check we can import the requested module."""
    try:
        return __import__(module_name)
    except ImportError:
        return None

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

__version__ = "Undefined"
for line in open("proteusPy/__init__.py"):
    if line.startswith("__version__"):
        exec(line.strip())

setup(name='proteusPy',
      version=__version__,
      description='proteusPy - Protein Structure Analysis and Modeling Tools',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/suchanek/proteusPy/',
      author='Eric G. Suchanek, PhD',
      author_email='suchanek@mac.com',
      license='MIT',
      packages=['proteusPy'],
      keywords='proteus suchanek disulfide',
      install_requires=[],
      source='https://github.com/suchanek/proteusPy/',
      project_urls={
        "Documentation": "https://suchanek.github.io/proteusPy/",
        "Source": "https://github.com/suchanek/proteusPy/",
        "Tracker": "https://github.com/suchanek/proteusPy/issues",
      },
      classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3.9'],
      include_package_data=True,
      package_data={'proteusPy': ['data/*.txt', 'data/*.py', 'data/*.json', 'data/*.csv']},
      exclude_package_data={'proteusPy': ['data/PDB_all_ss.pkl', 'data/PDB_SS_ALL_LOADER.pkl', 'PDB_all_ss_dict.pkl']},
      python_requires=">=%i.%i" % MIN_PY_VER,
      zip_safe=False)
      