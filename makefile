# makefile for proteusPy
CONDA = /home/suchanek/miniforge3/condabin/mamba

new:
	$(CONDA) env create --name proteusPy --file ppy.yml -y 

new2:
	$(CONDA) create --name proteusPy -y python=3.11.7
	
install:
	pip install . && cd ../biopython && pip install .
	touch inst.out
    

jup: inst.out
	sh jupyter.sh

