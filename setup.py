# initialization for proteusPy pip setup
from setuptools import setup, find_packages
__version__ = '0.22dev'

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
      packages=find_packages(include=['proteusPy']),
      keywords='proteus suchanek',
      install_requires=['pandas', 'numpy', 'matplotlib', 'pyvista', 'biopython'],
      source='https://github.com/suchanek/proteusPy/',
      classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9'],
      include_package_data=True,
      package_data={'': ['data/*.csv', 'data/*.pkl', 'data/*.txt', 'data/*.py']},
      zip_safe=False)
