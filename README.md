## Design of novel Cas9 PAM-interacting domains using evolution-based modelling and structural quality assessment

### Abstract

We present here an approach to protein design that combines (i) scarce functional information such as experimental data 
(ii) evolutionary information learned from a natural sequence variants and (iii) physics-grounded modeling. Using a 
Restricted Boltzmann Machine (RBM), we learn a sequence model of a protein family. We use semi-supervision to leverage
available functional information during the RBM training. We then propose a strategy to explore the protein 
representation space that can be informed by external models such as an empirical force-field method (FoldX). Our 
approach is applied to a domain of the Cas9 protein responsible for recognition of a short DNA motif. We experimentally 
assess the functionality of 71 variants generated to explore a range of RBM and FoldX energies. Sequences with as many 
as 50 differences (20% of the protein domain) to the wild-type retained functionality. Overall, 21/71 sequences designed 
with our method were functional. Interestingly, 6/71 sequences showed an improved activity in comparison with the original
wild-type protein sequence. These results demonstrate the interest in further exploring the synergies between machine-learning
of protein sequence representations and physics grounded modeling strategies informed by structural information.


Take a look at our [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011621)

### Content

- Scripts to train and use the RBM model in `torchpgm/` folder, examples about how to use it are available here in 
`notebooks/ModelingThroughRBM.ipynb`.
- You can reproduce the graphs from the paper using `notebooks/ExperimentalResults.ipynb` and using the data in 
`data/variants/ml-design-pid.csv`. 
- You can use `notebooks/ChimeraResults.ipynb` to reproduce the graphs from the paper 
about the chimeras. The sequences can be found in `data/variants/chimeras.csv`.
- Constrained Langevin Dynamic is implemented in `cld/` folder, examples about how to use it are available in 
`notebooks/SamplingRBMwithCLD.ipynb`.

### Data Availability

The protein sequences are from the PF16595 family of the Pfam database. The labels were collected from Vink et al. (2021).
The processed data can be found following this link: #TODO

### Contact

If you have any question please feel free to contact me at [cyril.malbranke@phys.ens.fr](mailto:cyril.malbranke@ens.fr)