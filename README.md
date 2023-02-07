*proteusPy* is a Python package specializing in the modeling and analysis of proteins of known structure with an emphasis on Disulfide Bonds. This package reprises my molecular modeling program [Proteus](https://doi.org/10.1021/bi00368a023), and relies on the [Turtle3D](https://suchanek.github.io/proteusPy/proteusPy/turtle3D.html) class. The turtle implements the functions ``Move``, ``Roll``, ``Yaw``, ``Pitch`` and ``Turn`` for movement in a three-dimensional space. The [Disulfide](https://suchanek.github.io/proteusPy/proteusPy/Disulfide.html) class implements methods to analyze the protein structure stabilizing element known as a *Disulfide Bond*. This class and its underlying methods are being used to perform a structural analysis of over 35,800 disulfide-bond containing proteins in the RCSB protein data bank.

### Virtual Environment Installation/Creation

1. Install Anaconda (<http://anaconda.org>)
   - Create a new environment using python 3.9
   - Activate the environment
2. Build the environment. At this point it's probably best to clone the repo via github since it contains all
   of the notebooks and test programs. Ultimately the distribution can be used from pyPi as a normal
   package.
   - Using pyPi:
     - python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ proteusPy
   - From gitHub:
     - Install git-lfs
       - https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage
       - From a shell prompt: git-lfs track "*.csv" "*.pkl" "*.mp4"
     - git clone https://github.com/suchanek/proteusPy/proteusPy.git
     - cd into the repo
     - pip install .
  

#### Publications
* https://doi.org/10.1021/bi00368a023
* https://doi.org/10.1021/bi00368a024
* https://doi.org/10.1016/0092-8674(92)90140-8
* http://dx.doi.org/10.2174/092986708783330566


*NB:* This distribution is actively being developed and will be difficult to implement locally unless the BioPython patch is applied. Also, if you're running on an M-series Mac then it's important to install Biopython first, since the generic release won't build on the M1. 1/26/23 -egs

Eric G. Suchanek, PhD. mailto:suchanek@mac.com

