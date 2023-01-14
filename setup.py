# initialization for proteusPy pip setup
from distutils.core import setup
__version__ = '0.9dev'

setup(name='proteusPy',
      version=__version__,
      description='proteus structure analysis tools',
      url='https://github.com/suchanek/proteusPy/',
      author='Eric G. Suchanek, PhD',
      author_email='suchanek@mac.com',
      license='MIT',
      packages=['proteusPy'],
      zip_safe=False)
