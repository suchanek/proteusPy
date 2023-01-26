# initialization for proteusPy pip setup
from setuptools import setup, find_packages
__version__ = '0.1dev'

setup(name='proteusPy',
      version=__version__,
      description='proteusPy - protein structure analysis tools',
      url='https://github.com/suchanek/proteusPy/',
      author='Eric G. Suchanek, PhD',
      author_email='suchanek@mac.com',
      license='MIT',
      packages=find_packages(include=['proteusPy']),
      keywords='proteus suchanek',
      install_requires=['pandas', 'numpy', 'matplotlib', 'pyvista', 'biopython', 'python>=3.9'],
      source='https://github.com/suchanek/proteusPy/',
      zip_safe=False)
