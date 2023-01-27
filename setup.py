# initialization for proteusPy pip setup
from setuptools import setup, find_packages
__version__ = '0.23dev'

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='proteusPy',
      version=__version__,
      description='proteusPy - protein structure analysis tools',
      long_description=long_description,
      url='https://github.com/suchanek/proteusPy/',
      author='Eric G. Suchanek, PhD',
      author_email='suchanek@mac.com',
      license='MIT',
      packages=['proteusPy'],
      keywords='proteus suchanek',
      install_requires=['pandas', 'numpy', 'matplotlib', 'pyvista'],
      source='https://github.com/suchanek/proteusPy/',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Programming Language :: Python :: 3.9'],
      include_package_data=True,
      package_data={'': ['data/*.txt', 'data/*.py']},
      zip_safe=False)
