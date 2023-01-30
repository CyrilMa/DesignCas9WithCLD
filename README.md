## Computational design of novel Cas9 PAM-interacting domains using evolution-based modelling and structural quality assessment

[Graphical Abstract](ressources/graphical_abstract.png)

**Abstract** 
We present here an approach to protein design that combines evolutionary 
and physics-grounded modeling. Using a Restricted Boltzmann Machine, 
we learned a sequence model of a protein family and then explored of
the protein representation space using an empirical force field method
(FoldX). This method was applied to a domain of the Cas9 protein responsible
for recognition of a short DNA motif. We assessed the functionality of 71
variants exploring a range of RBM and FoldX energies. We show how a 
combination of structural and evolutionary information can identify 
functional variants with high accuracy. Sequences with as many as 
50 differences (20% of the protein domain) to the wild-type retained 
functionality. Interestingly, some sequences (6/71) produced by our 
method showed an improved activity in comparison with the original 
wild-type proteins sequence. These results demonstrate the interest 
of further exploring the synergies between machine-learning of proteins 
sequence representations and physics grounded modeling strategies informed
by structural information.

This repository is supporting the paper Malbranke et al. 2023 that you can 
find here.

In this repository, we provide :
- Scripts to format data from FASTA file in `data/` as well as all necessary data structure.
- Scripts to implement RBM and SSL-RBM using PyTorch in `torchpgm/`
- Scripts to perform Constrained Langevin Dynamics using RBM in `cld/`
- Tutorials and Notebooks to reproduce the results of the paper in `notebooks/` and to learn how to perform CLD using these scripts

To use these scripts please install all required Python libraries from `requirements.txt` as well as
[MMSEQS](https://github.com/soedinglab/MMseqs2) and [HH-suite](https://github.com/soedinglab/hh-suite). 
Also, download the working data (see below) or add yours easily following the tutorials. Once done, update
the `config.py` with your own absolute path to your data folder to avoid any bug.

### Data Availability

All heavy data are available to download in this [folder]() :
  
After downloading is done, extract it and set the path to different data in `config.py`
  
Please, should these links expires, don't hesitate to reach out for us.

### Building Data


### Training RBM and SSL-RBM

You learn how to train Restricted Boltzmann Machine and Semi-supervised Learning Restricted Boltzmann Machine
using the notebook available in [`notebooks/ModelingThroughRBM.ipynb`](notebooks/ModelingThroughRBM.ipynb)

### Constrained Langevin Dynamics

Constrained Langevin Dynamics and how to run it are available in [`notebooks/SamplingRBM-withCLD.ipynb`](notebooks/SamplingRBM-withCLD.ipynb).
We also displayed a study of the effect of the Semi-Supervised Learning RBM on how well the Constrained Langevin Dynamics perform
in the notebook.

The code are available in the `cld/` folder with some annotations.

### Sampling with RBM and SSQA

You can refer to [`notebooks/RBMSampling_With_SSQA.ipynb`](notebooks/RBMSampling_With_SSQA.ipynb) for detailled walktrough on sampling. 
Some pretrained weights for RBM are provided in `pfam/.../weights`.

### Experimental data

File containing informations about experimentally tested sequences is available at
[`data/ML.ipynb`](notebooks/RBMSampling_With_SSQA.ipynb)

### Contact

If you have any question please feel free to contact me at [cyril.malbranke@phys.ens.fr](mailto:cyril.malbranke@ens.fr)